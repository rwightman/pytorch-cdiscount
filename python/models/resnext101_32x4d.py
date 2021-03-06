""" Pytorch Resnet-101-32x4d impl
Sourced by running https://github.com/clcarwin/convert_torch_to_pytorch (MIT) on
https://github.com/facebookresearch/ResNeXt (BSD-3-Clause)
Pretrained weights are not being used as they are CC BY-NC 4.0 license.
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from .adaptive_avgmax_pool import *
from functools import reduce


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))


def resnext_101_32x4d_features(activation_fn=nn.ReLU()):
    features = nn.Sequential(  # Sequential,
        nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3), 1, 1, bias=False),
        nn.BatchNorm2d(64),
        activation_fn,
        nn.MaxPool2d((3, 3), (2, 2), (1, 1)),
        nn.Sequential(  # Sequential,
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(64, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(128),
                                  activation_fn,
                                  nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(128),
                                  activation_fn,
                              ),
                              nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                          ),
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                          ),
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(256, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(128),
                                  activation_fn,
                                  nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(128),
                                  activation_fn,
                              ),
                              nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(256, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(128),
                                  activation_fn,
                                  nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(128),
                                  activation_fn,
                              ),
                              nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
        ),
        nn.Sequential(  # Sequential,
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(256),
                                  activation_fn,
                                  nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(256),
                                  activation_fn,
                              ),
                              nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                          ),
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(256, 512, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                          ),
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(256),
                                  activation_fn,
                                  nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(256),
                                  activation_fn,
                              ),
                              nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(256),
                                  activation_fn,
                                  nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(256),
                                  activation_fn,
                              ),
                              nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(256),
                                  activation_fn,
                                  nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(256),
                                  activation_fn,
                              ),
                              nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
        ),
        nn.Sequential(  # Sequential,
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 1024, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                                  nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(512),
                                  activation_fn,
                              ),
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
        ),
        nn.Sequential(  # Sequential,
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                                  activation_fn,
                                  nn.Conv2d(1024, 1024, (3, 3), (2, 2), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(1024),
                                  activation_fn,
                              ),
                              nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(2048),
                          ),
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 2048, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(2048),
                          ),
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(2048, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                                  activation_fn,
                                  nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(1024),
                                  activation_fn,
                              ),
                              nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(2048),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(2048, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                                  activation_fn,
                                  nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                                  nn.BatchNorm2d(1024),
                                  activation_fn,
                              ),
                              nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(2048),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
        ),
    )
    return features


class ResNeXt101_32x4d(nn.Module):

    def __init__(self, num_classes=1000, activation_fn=nn.ReLU(), drop_rate=0, global_pool='avg'):
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.global_pool = global_pool
        super(ResNeXt101_32x4d, self).__init__()
        self.features = resnext_101_32x4d_features(activation_fn=activation_fn)
        self.num_features = 2048
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.fc = nn.Linear(2048, num_classes)

    def forward_features(self, x, pool=True):
        x = self.features(x)
        if pool:
            x = adaptive_avgmax_pool2d(x, self.global_pool)
            x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x, pool=True)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


def resnext101_32x4d(pretrained=False, num_classes=1000, **kwargs):
    model = ResNeXt101_32x4d(num_classes=num_classes, **kwargs)
    if pretrained:
        print('Warning: No pretrained weights setup.')
    return model
