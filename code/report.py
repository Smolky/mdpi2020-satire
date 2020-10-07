"""
    Report
    
    This script generate the classification report for a deployed model
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Óscar Apolinario Arzube <oscar.apolinarioa@ug.edu.ec>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

# Load basic stuff
import random
import argparse
import pickle
import config
import os
import numpy as np
import tensorflow as tf
import talos as ta
import sys
import csv
import pandas as pd

from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report


# @var max_words Tokenizer max words
max_words = 50000


# @var maxlen The max number of tokens for each tweet. 
maxlen = 100



# Parser
parser = argparse.ArgumentParser (description='Evaluate hyperparameters.')
parser.add_argument ('--dataset', dest='dataset', default='satire-2017', help='satire-2017-spain|satire-2017-mexico|satire-2017')

args = parser.parse_args ()
print (args.dataset)


# @package Talos restored package
package = ta.Restore ("satire-2017-spain_model.zip")

"""
print (package.details)
print (package.model)
print (package.params)
print (package.results)
print (package.x)
print (package.y)
"""


# @var umucorpus_ids int|string The Corpus IDs
for key, umucorpus_ids in config.ids[args.dataset].items ():

    # @var train_filename String 
    train_filename = './../data/' + key + '.csv'


    # @var label_filename String 
    label_filename = '../results/binarizer/' + key + '.pickle'
    
    
    # @var token_filename String 
    token_filename = '../results/tokens/' + key + '_' + str (max_words) + '.pickle'


    # Restore tokenizer and laberizer
    with open (label_filename, 'rb') as handle:
        lb = pickle.load (handle)
        
        
    # Store tokenizer
    with open (token_filename, 'rb') as handle:
        tokenizer = pickle.load (handle)
    

    # Read tweets training and evaluating datasets
    df_texts = pd.read_csv (train_filename, names=['sentence', 'label'], sep='\t', header=None, quoting=csv.QUOTE_NONE)
    
    
    # Get features and class of the training data
    sentences_features = df_texts['sentence'];
    sentences_labels = df_texts['label'];
    print ("\t...Training datasets readed")


    # Binarize labels
    sentences_labels_binary = lb.transform (sentences_labels)
    

    # Update to tokens
    sentences_features = tokenizer.texts_to_sequences (sentences_features)


    # Add padding
    sentences_features = pad_sequences (sentences_features, padding='post', maxlen=maxlen)
    
    
    # Get predictions
    predictions = package.model.predict (sentences_features)
    
    
    # Print
    print (classification_report (sentences_labels_binary, predictions.argmax (axis=-1)))


