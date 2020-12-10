from net.alexnet import *
from net.inception import *
from net.resnet import *
from net.vgg import *

model_factory = {
    'alexnet': alexnet,
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'inception_v3': inception_v3,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
}

input_sizes = {
    'alexnet': (1024, 3, 224, 224),
    'vgg11': (64, 3, 224, 224),
    'vgg13': (64, 3, 224, 224),
    'vgg16': (64, 3, 224, 224),
    'vgg19': (64, 3, 224, 224),
    'inception_v3': (32, 3, 300, 300),
    'resnet18': (256, 3, 224, 224),
    'resnet34': (128, 3, 224, 224),
    'resnet50': (64, 3, 224, 224),
    'resnet101': (32, 3, 224, 224),
    'resnet152': (16, 3, 224, 224),
}