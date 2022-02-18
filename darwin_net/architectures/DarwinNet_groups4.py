"""
Module for building ResNet Module according to different config file
"""
from torch.nn import Module, ModuleList
import torch
import torch.nn as nn
import numpy as np

from thop import profile
from copy import deepcopy


def conv2d(w_in, w_out, k, *, stride=1, groups=1, dilation=1, bias=False):
    """Helper for building a conv2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    s, p, g, b = stride, (k - 1) // 2, groups, bias
    return nn.Conv2d(w_in, w_out, k, stride=s, padding=p, groups=g, bias=b)


def norm2d(w_in):
    """Helper for building a norm2d layer."""
    return nn.BatchNorm2d(num_features=w_in, eps=1e-5, momentum=0.1)


def pool2d(_w_in, k, *, stride=1):
    """Helper for building a pool2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    return nn.MaxPool2d(k, stride=stride, padding=(k - 1) // 2)


def pool2d_average(_w_in, k, *, stride=1):
    """Helper for building a pool2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    return nn.AvgPool2d(k, stride=stride, padding=(k - 1) // 2)


def gap2d(_w_in):
    """Helper for building a gap2d layer."""
    return nn.AdaptiveAvgPool2d((1, 1))


def linear(w_in, w_out, *, bias=False):
    """Helper for building a linear layer."""
    return nn.Linear(w_in, w_out, bias=bias)


def activation():
    """Helper for building an activation layer."""
    activation_fun = "relu"  # cfg.MODEL.ACTIVATION_FUN.lower()
    if activation_fun == "relu":
        return nn.ReLU(inplace=True)
    elif activation_fun == "silu" or activation_fun == "swish":
        try:
            return torch.nn.SiLU()
        except AttributeError:
            return SiLU()
    else:
        raise AssertionError("Unknown MODEL.ACTIVATION_FUN: " + activation_fun)


def conv2d_flops(cx, w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Accumulates complexity of conv2d into cx = (h, w, flops, params, acts)."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    h, w, flops = cx["h"], cx["w"], cx["flops"]
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    flops += k * k * w_in * w_out * h * w // groups + (w_out if bias else 0)
    return {"h": h, "w": w, "flops": flops}


def conv2d_activations(cx, w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Accumulates complexity of conv2d into cx = (h, w, flops, params, acts)."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    h, w, activations = cx["h"], cx["w"], cx["activations"]
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    activations.append([w, h, w_out])
    return {"h": h, "w": w, "activations": activations}


def pool2d_flops(cx, w_in, k, *, stride=1):
    """Accumulates complexity of pool2d into cx = (h, w, flops, params, acts)."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    h, w, flops = cx["h"], cx["w"], cx["flops"]
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    return {"h": h, "w": w, "flops": flops}


def gap2d_flops(cx, _w_in):
    """Accumulates complexity of gap2d into cx = (h, w, flops, params, acts)."""
    flops = cx["flops"]
    return {"h": 1, "w": 1, "flops": flops}


def linear_flops(cx, w_in, w_out, *, bias=False):
    """Accumulates complexity of linear into cx = (h, w, flops, params, acts)."""
    h, w, flops = cx["h"], cx["w"], cx["flops"]
    flops += w_in * w_out + (w_out if bias else 0)
    return {"h": h, "w": w, "flops": flops}


def linear_activations(cx, w_in, w_out, *, bias=False):
    """Accumulates complexity of linear into cx = (h, w, flops, params, acts)."""
    h, w, activations = cx["h"], cx["w"], cx["activations"]
    activations.append([w_out])
    return {"h": h, "w": w, "activations": activations}


def get_transformation_function(func_name):
    "Returns the transformation function for ResNet Module"
    functions_available = {"basic": BasicTransform, "bottleneck": BottleneckTransform}
    try:
        func = functions_available[func_name]
        return func
    except:
        raise (ValueError("Function not available"))


class BasicTransform(Module):
    """Basic transformation: 3x3, BN, AF, 3x3, BN."""

    expansion: int = 1

    def __init__(
        self,
        inplanes,
        channels,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
    ):
        super(BasicTransform, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.stride = stride
        self.inplanes = inplanes
        self.channels = channels

        if stride != 1:
            self.avgpool = nn.AvgPool2d(2, stride=stride)
            self.conv1 = conv2d(self.inplanes, self.channels, 3, groups=4)
        else:
            self.conv1 = conv2d(
                self.inplanes, self.channels, 3, stride=self.stride, groups=4
            )

        # self.conv1 = conv2d(self.inplanes, self.channels, 3, stride=self.stride)
        self.bn1 = norm2d(self.channels)
        self.relu = activation()
        self.conv2 = conv2d(self.channels, self.channels, 3, groups=4)
        self.bn2 = norm2d(self.channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.stride != 1:
            out = self.avgpool(x)
            out = self.conv1(out)
        else:
            out = self.conv1(x)
        # out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def flops(self, cx):
        cx = conv2d_flops(cx, self.inplanes, self.channels, 3, stride=self.stride)
        cx = conv2d_flops(cx, self.channels, self.channels, 3)
        if self.downsample:
            cx = conv2d_flops(
                cx, self.inplanes, self.channels * self.expansion, 1, stride=self.stride
            )
        return cx

    def activations(self, cx):
        cx = conv2d_activations(cx, self.inplanes, self.channels, 3, stride=self.stride)
        cx = conv2d_activations(cx, self.channels, self.channels, 3)
        if self.downsample:
            cx = conv2d_activations(
                cx, self.inplanes, self.channels * self.expansion, 1, stride=self.stride
            )
        return cx

    def params(self):
        conv1_params = np.sum([np.prod(p.size()) for p in self.conv1.parameters()])
        bn1_params = np.sum([np.prod(p.size()) for p in self.bn1.parameters()])

        conv2_params = np.sum([np.prod(p.size()) for p in self.conv2.parameters()])
        bn2_params = np.sum([np.prod(p.size()) for p in self.bn2.parameters()])
        downsample_params = 0
        if self.downsample is not None:
            for module in self.downsample:
                downsample_params += np.sum(
                    [np.prod(p.size()) for p in module.parameters()]
                )
        return conv1_params + bn1_params + conv2_params + bn2_params + downsample_params


class BottleneckTransform(Module):
    """Bottleneck transformation: 1x1, BN, AF, 3x3, BN, AF, 1x1, BN."""

    expansion: int = 2

    def __init__(
        self,
        inplanes,
        channels,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
    ):
        super(BottleneckTransform, self).__init__()
        width = int(channels * (base_width / 64.0)) * groups
        self.stride = stride
        self.conv1 = conv2d(inplanes, width, 1, groups=1)
        self.bn1 = norm2d(width)

        if stride != 1:
            self.avgpool = nn.AvgPool2d(2, stride=stride)
            self.conv2 = conv2d(width, width, 3, stride=1, groups=4, dilation=dilation)
        else:
            self.conv2 = conv2d(
                width, width, 3, stride=stride, groups=4, dilation=dilation
            )

        # self.conv2 = conv2d(width, width, 3, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = norm2d(width)
        self.conv3 = conv2d(width, channels * self.expansion, 1, groups=1)
        self.bn3 = norm2d(channels * self.expansion)
        self.relu = activation()
        self.downsample = downsample
        self.stride = stride
        self.width = width
        self.inplanes = inplanes
        self.channels = channels
        self.groups = groups
        self.dilation = dilation
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.stride != 1:
            out = self.avgpool(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def flops(self, cx):
        cx = conv2d_flops(cx, self.inplanes, self.width, 1, stride=1)
        cx = conv2d_flops(
            cx, self.width, self.width, 3, stride=self.stride, groups=self.groups
        )
        cx = conv2d_flops(cx, self.width, self.channels * self.expansion, 1)
        if self.downsample:
            cx = conv2d_flops(
                cx, self.inplanes, self.channels * self.expansion, 1, stride=self.stride
            )
        return cx

    def activations(self, cx):
        cx = conv2d_activations(cx, self.inplanes, self.width, 1, stride=1)
        cx = conv2d_activations(
            cx, self.width, self.width, 3, stride=self.stride, groups=self.groups
        )
        cx = conv2d_activations(cx, self.width, self.channels * self.expansion, 1)
        if self.downsample:
            cx = conv2d_activations(
                cx, self.inplanes, self.channels * self.expansion, 1, stride=self.stride
            )
        return cx

    def params(self):
        conv1_params = np.sum([np.prod(p.size()) for p in self.conv1.parameters()])
        bn1_params = np.sum([np.prod(p.size()) for p in self.bn1.parameters()])

        conv2_params = np.sum([np.prod(p.size()) for p in self.conv2.parameters()])
        bn2_params = np.sum([np.prod(p.size()) for p in self.bn2.parameters()])

        conv3_params = np.sum([np.prod(p.size()) for p in self.conv3.parameters()])
        bn3_params = np.sum([np.prod(p.size()) for p in self.bn3.parameters()])
        downsample_params = 0
        if self.downsample is not None:
            for module in self.downsample:
                downsample_params += np.sum(
                    [np.prod(p.size()) for p in module.parameters()]
                )
        return (
            conv1_params
            + bn1_params
            + conv2_params
            + bn2_params
            + conv3_params
            + bn3_params
            + downsample_params
        )


class ResHead(Module):
    """ResNet head: AvgPool, 1x1."""

    def __init__(self, w_in, num_classes):
        super(ResHead, self).__init__()
        self.w_in = w_in
        self.num_classes = num_classes
        self.avg_pool = gap2d(w_in)
        self.fc = linear(w_in, num_classes, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def flops(self, cx):
        cx = gap2d_flops(cx, self.w_in)
        cx = linear_flops(cx, self.w_in, self.num_classes, bias=True)
        return cx

    def activations(self, cx):
        cx = linear_activations(cx, self.w_in, self.num_classes, bias=True)
        return cx

    def params(self):
        fc_params = np.sum([np.prod(p.size()) for p in self.fc.parameters()])
        return fc_params


def initialize_complexity(input_shape):
    cx = dict()
    cx["h"] = input_shape[0]
    cx["w"] = input_shape[1]
    cx["flops"] = 0
    cx["activations"] = []
    return cx


class DarwinStemIN(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(DarwinStemIN, self).__init__()
        self.w_in = w_in
        self.w_out = w_out

        self.relu = activation()

        self.conv1 = conv2d(w_in, w_out, 3, stride=2)
        self.bn1 = norm2d(w_out)
        self.conv2 = conv2d(w_out, w_out, 3, stride=1, groups=4)
        self.bn2 = norm2d(w_out)

        self.pool = pool2d_average(w_out, 3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)

        return x

    def flops(self, cx):
        cx = conv2d_flops(cx, self.w_in, self.w_out, 3, stride=2)
        cx = conv2d_flops(cx, self.w_out, self.w_out, 3, stride=1)
        cx = conv2d_flops(cx, self.w_out, self.w_out, 3, stride=1)
        cx = pool2d_flops(cx, self.w_out, 3, stride=2)
        return cx

    def activations(self, cx):
        cx = conv2d_activations(cx, self.w_in, self.w_out, 3, stride=2)
        cx = conv2d_activations(cx, self.w_out, self.w_out, 3, stride=1)
        cx = conv2d_activations(cx, self.w_out, self.w_out, 3, stride=1)
        return cx

    def params(self):
        conv2d1_params = np.sum([np.prod(p.size()) for p in self.conv1.parameters()])
        conv2d2_params = np.sum([np.prod(p.size()) for p in self.conv2.parameters()])
        conv2d3_params = np.sum([np.prod(p.size()) for p in self.conv3.parameters()])
        bn_params1 = np.sum([np.prod(p.size()) for p in self.bn1.parameters()])
        bn_params2 = np.sum([np.prod(p.size()) for p in self.bn2.parameters()])
        bn_params3 = np.sum([np.prod(p.size()) for p in self.bn3.parameters()])
        return (
            bn_params1
            + bn_params2
            + bn_params3
            + conv2d1_params
            + conv2d2_params
            + conv2d3_params
        )


class DarwinNetV2(Module):
    """DarwinNetV2 model."""

    def __init__(self, blockspecs, input_shape, num_classes, model_structure=[[None]]):
        super(DarwinNetV2, self).__init__()
        self.blockspecs = blockspecs
        self.channels = [max(4, b.channels - b.channels % 4) for b in self.blockspecs]
        self.model_depth = [b.depth for b in self.blockspecs]
        self.replace_stride_with_dilation = [False for i in range(len(self.channels))]
        self.module_types = ["basic", "basic", "bottleneck", "bottleneck"]

        self.num_channels = len(self.channels)
        self.blocks = []
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dilation = 1
        self.base_width = 64
        self.groups = 1
        assert self.num_channels == len(self.model_depth)
        self._construct_imagenet()

    def _construct_imagenet(self):
        depth = self.model_depth
        self.inplanes = 32
        self.stem = DarwinStemIN(self.input_shape[2], self.inplanes)

        for i in range(len(self.channels)):
            if i == 0:
                self.blocks.append(
                    self._make_layer(
                        get_transformation_function(self.module_types[i]),
                        self.channels[i],
                        stride=1,
                        depth=depth[i],
                    )
                )
            else:
                self.blocks.append(
                    self._make_layer(
                        get_transformation_function(self.module_types[i]),
                        self.channels[i],
                        stride=2,
                        depth=depth[i],
                        dilate=self.replace_stride_with_dilation[i - 1],
                    )
                )
        self.blocks = nn.ModuleList(self.blocks)
        self.head = ResHead(
            self.channels[self.num_channels - 1]
            * get_transformation_function(
                self.module_types[self.num_channels - 1]
            ).expansion,
            self.num_classes,
        )
        # model = nn.Sequential(*[self.stem, self.blocks, self.head])
        # return model

    def _make_layer(self, module_type, channels, stride, depth, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(2, stride=stride),
                conv2d(
                    self.inplanes,
                    channels * module_type.expansion,
                    1,
                    stride=1,
                    groups=1,
                ),
                norm2d(channels * module_type.expansion),
            )
        elif self.inplanes != channels * module_type.expansion:
            downsample = nn.Sequential(
                conv2d(
                    self.inplanes,
                    channels * module_type.expansion,
                    1,
                    stride=stride,
                    groups=1,
                ),
                norm2d(channels * module_type.expansion),
            )

        layers = []
        layers.append(
            module_type(
                self.inplanes,
                channels,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        )
        self.inplanes = channels * module_type.expansion
        for _ in range(1, depth):
            layers.append(
                module_type(
                    self.inplanes,
                    channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        for i in range(self.num_channels):
            x = self.blocks[i](x)
        x = self.head(x)
        return x

    def flops(self):
        flops = profile(
            deepcopy(self),
            inputs=(
                torch.zeros(
                    1, self.input_shape[2], self.input_shape[0], self.input_shape[1]
                ),
            ),
            verbose=False,
        )[0]
        return flops

    def flops2(self):
        """Computes model complexity. If you alter the model, make sure to update."""
        cx = initialize_complexity(self.input_shape)
        cx = self.stem.flops(cx)
        for i in range(self.num_channels):
            for module in self.blocks[i]:
                cx = module.flops(cx)
        cx = self.head.flops(cx)
        return cx["flops"]

    def activations(self):
        """Computes model complexity. If you alter the model, make sure to update."""
        cx = initialize_complexity(self.input_shape)
        cx = self.stem.activations(cx)
        for i in range(self.num_channels):
            for module in self.blocks[i]:
                cx = module.activations(cx)
        cx = self.head.activations(cx)
        return cx["activations"]

    def params(self):
        total_params = self.stem.params()
        for block in self.blocks:
            for module in block:
                total_params += module.params()

        total_params += self.head.params()
        return total_params
