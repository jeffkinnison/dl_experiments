from shadho import HyperparameterSearch
from shadho import choose, log2_randint, uniform

import copy

INITIALIZERS = choose(['zeros', 'ones', 'orthogonal', 'glorot_uniform',
                      'glorot_normal'])

REGULARIZERS = choose([None, 'l1', 'l2', 'l1_l2'])

CONSTRAINTS = choose([None, 'non_neg', 'unit_norm'])

LOSSES = choose(['squared_hinge', 'poisson', 'cosine_proximity'
                'kullback_liebler_divergence', 'mean_squared_error'])

DENSE = {
    'units': log2_randint(4, 10),
    'activation': choose(['elu', 'relu', 'selu', 'sigmoid', 'softmax', 'tanh']),
    'kernel_initializer': INITIALIZERS,
    'bias_initializer': INITIALIZERS,
    'kernel_regularizer': REGULARIZERS,
    'bias_regularizer': REGULARIZERS,
    'activity_regularizer': REGULARIZERS,
    'kernel_constraint': CONSTRAINTS,
    'bias_constraint': CONSTRAINTS,
    'dropout': {
        'optional': True,
        'rate': uniform(0.0, 1.0)
    }
}

layer1 = copy.deepcopy(DENSE)
DENSE['optional'] = True
layer2 = copy.deepcopy(DENSE)
layer3 = copy.deepcopy(DENSE)
layer4 = copy.deepcopy(DENSE)
layer5 = copy.deepcopy(DENSE)

layer1['next'] = layer2
layer2['next'] = layer3
layer3['next'] = layer4
layer4['next'] = layer5

space = {
    'optimizer': choose(['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam',
                        'nadam']),
    'lr': uniform(0.00001, 1.0),
    'loss': LOSSES,
    'layers': layer1
}


config = {
    'name': 'shadho',
    'port': 9000,
    'exclusive': True,
    'shutdown': True,
    'catalog': False,
    'logfile': 'v_to_sum.log',
    'debug': 'v_to_sum.debug',
    'files': [
        {
            'localpath': 'train.py',
            'remotepath': 'train.py',
            'type': 'input',
            'cache': True
        },
        {
            'localpath': 'shadhocmd.sh',
            'remotepath': 'shadhocmd.sh',
            'type': 'input',
            'cache': True
        },
        {
            'localpath': 'train.npz',
            'remotepath': 'train.npz',
            'type': 'input',
            'cache': True
        },
        {
            'localpath': 'out.tar.gz',
            'remotepath': 'out.tar.gz',

        }
    ]
}


if __name__ == "__main__":
    opt = HyperparameterSearch(space,
                               [],
                               config,
                               use_complexity=False,
                               use_priority=False,
                               timeout=600,
                               max_tasks=50)
    opt.optimize()
