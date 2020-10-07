'''
    Configuration of the scripts
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Óscar Apolinario Arzube <oscar.apolinarioa@ug.edu.ec>
    @author Rafael Valencia-Garcia <valencia@um.es>
'''

import sys


# @var max the Number of tweets to compile. Reduce for testing
max = sys.maxsize
# max = 1000


# IDs of the datasets
strategy = {
    'satire-2017-spain': 'satire-non-satire',
    'satire-2017-mexico': 'satire-non-satire',
    'satire-2017': 'satire-non-satire'
}


# Number of classes of each dataset
number_of_classes = {
    'satire-2017-spain': 1,
    'satire-2017-mexico': 1,
    'satire-2017': 1
}


# IDs of the datasets
ids = {
    'satire-2017-spain': {
        'satire-2017-spain': [9]
    },
    'satire-2017-mexico': {
        'satire-2017-mexico': [11]
    },
    'satire-2017': {
       'satire-2017': [9, 11]
    }
}
