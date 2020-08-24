# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from .VGGnet_train import VGGnet_train
from .VGGnet_test import VGGnet_test
from .VGGnet_wo_context_train import VGGnet_wo_context_train
from .VGGnet_wo_context_test import VGGnet_wo_context_test

from . import factory
