"""
    Generate corpus features
    
    This script obtains the features for the corpus
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Óscar Apolinario Arzube <oscar.apolinarioa@ug.edu.ec>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import json
import pandas as pd
import random
import requests
import sys
import zipfile
import netrc
import os.path
import argparse
import config
import time

from pathlib import Path
from requests.auth import AuthBase



"""
   PLNAuth
   
   We use a custom authentication because the default authentication
   was adding Basic authentication and do not allowed to manually 
   introcuded the token
   
   @link https://requests.readthedocs.io/en/master/user/advanced/#custom-authentication
"""
class PLNAuth (AuthBase):
    def __init__(self, username):
        self.username = username

    def __call__(self, r):
        r.headers['Authorization'] = self.username
        return r
        
        
# Parser
parser = argparse.ArgumentParser (description='Generates the features.')
parser.add_argument ('--dataset', dest='dataset', default='satire-2017', help='satire-2017-spain|satire-2017-mexico|satire-2017')
parser.add_argument ('--format', dest='format', default='arff', help='arff|csv')

args = parser.parse_args ()
print (args.dataset)


# Read from the .netrc file in your home directory
secrets = netrc.netrc ()
email, account, password = secrets.authenticators ('collaborativehealth.inf.um.es')


# @var umutextstats_api_endpoint String
umutextstats_api_endpoint = 'https://collaborativehealth.inf.um.es/umutextstats/api/'


# @var certificate String
certificate = str (Path.home ()) + '/certificates/CA.pem'
print (certificate)



# @var reponse Response
print ("Loading credentials...")
response = requests.post (
    umutextstats_api_endpoint + 'login', 
    json={'email': email, 'password': password}, 
    verify=certificate
)


# Transform to JSON
response = response.json ()


# @var auth_token String
auth_token = str (response['data']['token'])
print ("Credentials acquired...")
print (auth_token)


# @var feature_sets Dictionary
feature_sets = {
    'sentence-embeddings': {
        'model': 'word-embeddings', 
        'class-strategy': config.strategy[args.dataset], 
        'endpoint': 'stats.' + args.format
    },
    'umutextstats': {
        'model': 'umutextstats', 
        'class-strategy': config.strategy[args.dataset], 
        'endpoint': 'stats.' + args.format
    }
}


# @var umucorpus_ids int|string The Corpus IDs
for key, umucorpus_ids in config.ids[args.dataset].items ():

    # @var filename_prepend String
    filename_prepend = key + '-'
        
    
    # Iterate over the feature sets
    for filename, payload in feature_sets.items ():
        
        # Start time
        start = time.time ()
        
        
        # @var zip_filename String 
        filename = './../features/' + filename_prepend + filename + '.' + args.format
        
        
        # Show log
        print ("Processing " + filename + " ...")
        
        
        # If file exists, skipe it
        if os.path.isfile (filename):
            print ("Skip file...")
            continue
        
        
        
        # @var request_payload Dictionary Prepare the request according to the features we want to retrieve
        request_payload = {
            'format': args.format,
            'umutextstats-config': 'default.xml',
            'source-provider': 'umucorpus',
            'max': config.max,
            'umucorpus': ','.join (str(x) for x in umucorpus_ids),
            'skip-first-row': 0,
            'labeled': 0,
            'corpus-merged': 0,
            'class-strategy': payload['class-strategy'],
            'balance-strategy': 'no-balance'
        }
        
        
        # Include model
        if 'model' in payload:
            request_payload['model'] = payload['model']
        
        
        
        # @var reponse Response
        response = requests.post (
            umutextstats_api_endpoint + payload['endpoint'], 
            json=request_payload, 
            verify=certificate,
            auth=PLNAuth (auth_token)
        )
        
        
        # Check response
        if (response.status_code != 200):
            print ("Authentication failed: " + str (response.status_code))
            print (response.text)
            sys.exit ()
        
        
        # UMUTextStats sends the results separtely
        if 'model' in payload and payload['model'] in ['word-embeddings', 'umutextstats']:
            
            # Get HTML result
            html_result = response.text
            
            
            # @var response Object Strip the values used for indicating progress
            json_response = json.loads (html_result[(html_result.index ('{')):])
            
            
            # Retrieve the file
            response = requests.get (
                umutextstats_api_endpoint + json_response['file'], 
                json=request_payload, 
                verify=certificate,
                headers={
                    'Authorization': auth_token
                }
            )


        done = time.time ()
        print (done - start)

        with open(filename, 'w') as f:
            print (response.text, file=f)