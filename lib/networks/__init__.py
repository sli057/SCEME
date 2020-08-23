# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from .VGGnet_train import VGGnet_train
from .VGGnet_test import VGGnet_test
from .VGGnet_wt_context_train import VGGnet_wt_context_train
from .VGGnet_wt_context_test import VGGnet_wt_context_test
from .VGGnet_context_ave_train import VGGnet_context_ave_train
from .VGGnet_context_ave_test import VGGnet_context_ave_test
from .VGGnet_context_maxpool_train import VGGnet_train as VGGnet_context_maxpool_train
from .VGGnet_context_maxpool_test import VGGnet_test as VGGnet_context_maxpool_test
from .VGGnet_classifier import VGGnet_classifier
#from .VGGnet_classifier_test import VGGnet_classifier_test
from . import factory
