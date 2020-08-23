# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

import networks.VGGnet_train
import networks.VGGnet_test
import networks.VGGnet_wo_context_train
import networks.VGGnet_wo_context_test



def get_network(name, data=None):
    """Get a network by name."""
    if name.split('_')[1] == 'test':
        return networks.VGGnet_test()
    elif name.split('_')[1] == 'train':
        return networks.VGGnet_train()
    elif name.split('_')[1] == 'wo' and name.split('_')[-1] == 'train':
        return networks.VGGnet_wo_context_train(data=data)
    elif name.split('_')[1] == 'wo' and name.split('_')[-1] == 'test':
        return networks.VGGnet_wo_context_test()
    elif name.split('_')[1] == 'classifier':# and name.split('_')[-1] == 'train':
        return networks.VGGnet_classifier()
    else:
        raise KeyError('Unknown dataset: {}'.format(name))
    

