"""
    Generate Deep-Learning Models
    
    This script generate different deep-learning models 
    for the Opinion Mining module of CollaborativeHealth 
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Óscar Apolinario Arzube <oscar.apolinarioa@ug.edu.ec>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

# Load basic stuff
import random
import os
import numpy as np
import datetime
import sys
import pprint


# Configure seed prior other stuff
# @link https://stackoverflow.com/questions/45230448/how-to-get-reproducible-result-when-running-keras-with-tensorflow-backend
seed_value = 2
os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
random.seed (seed_value)
np.random.seed (seed_value)
pp = pprint.PrettyPrinter(indent=4)


# Import KERAS
import csv
import pickle
import argparse
import config
import time
import pandas as pd
import matplotlib.pyplot as plt
from pipelinehelper import PipelineHelper

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline



from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest



# Parser
parser = argparse.ArgumentParser (description='Evaluate statistical features.')
parser.add_argument ('--dataset', dest='dataset', default='satire-2017', help='satire-2017-spain|satire-2017-mexico|satire-2017')

args = parser.parse_args ()
print (args.dataset)



# Train ratio (float or int, default=None)
train_size = 0.8
# train_size = 1000


# Validation ratio
test_size = None
# test_size = 333



# @var umucorpus_ids int|string The Corpus IDs
for key, umucorpus_ids in config.ids[args.dataset].items ():

    print ("Processing " + key)
    

    # @var confussion_matrix String 
    confussion_matrix = '../results/confussion_matrix/{{ name }}.txt';


    # @var architecture_diagram_filename String 
    architecture_diagram_filename = '../results/architecture/{{ name }}.png';


    # @var plot_filename String 
    plot_filename = '../results/plots/{{ name }}.png';
    
   
    # @var train_filename String 
    train_filename = './../data/' + key + '.csv'
    
    
    # Read tweets training and evaluating datasets
    df_texts = pd.read_csv (train_filename, names=['sentence', 'label'], sep='\t', header=None, quoting=csv.QUOTE_NONE)
    
    
    # Get features and class of the training data
    sentences_features = df_texts['sentence'];
    sentences_labels = df_texts['label'];
    print ("\t...Training datasets readed")
    
    
    
    # Empty dataset
    if (0 == len (sentences_features)):
        continue;
        
    
    # Split dataset
    sss = StratifiedShuffleSplit (n_splits = 1, train_size = train_size, test_size = test_size, random_state = seed_value)
    sss.get_n_splits (sentences_features, sentences_labels)
    for train_index, test_index in sss.split (sentences_features, sentences_labels):
        sentences_features_train, sentences_features_val = sentences_features[train_index], sentences_features[test_index]
        sentences_labels_train, sentences_labels_val = sentences_labels[train_index], sentences_labels[test_index]
        print ("...")
    
   
    
    # Features
    features = TfidfVectorizer ()
    
    
    # Classifier to evaluate
    rf_classifier = RandomForestClassifier (bootstrap=False, max_features='auto', min_samples_leaf=2, min_samples_split=2)
    svm_classifier = SVC ()
    lr_classifier = LogisticRegression (random_state=seed_value)
    mnb_classifier = MultinomialNB ()
    k_classifier = KNeighborsClassifier (n_neighbors = 2)
    
    
    # Create pipeline with the feature selection and the classifiers
    pipe = Pipeline ([
        ('features', features),
        
        ('select', PipelineHelper ([
            ('vt', VarianceThreshold ()),
            ('skbest', SelectKBest ()),
        ])),
        
        ('classifier', PipelineHelper ([
            ('svm', svm_classifier),
            ('rf', rf_classifier),
        ]))
    ])
    
    
    # This are the TFIDF vectorizer options
    features_options = {
        'features__min_df': [1],
        'features__sublinear_tf': [True],
        'features__strip_accents': ['unicode'],
        'features__use_idf': [True],
    }
    
    classifier_hypermateters = {
        'svm__C': [1],
        'svm__kernel': ['poly'],
    }
    
    
    # Create the specific bag of word features from unigrams to trigrams
    features = {
        'features__analyzer': ['char_wb'],
        'features__ngram_range': [(4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10)]
    }
    
    
    # Mix the specific and generic parameters for the character n-grams and the word-grams
    features = {**features, **features_options}
    
     
    # Mix the features with the classifier parameters
    features['classifier__selected_model'] = pipe.named_steps['classifier'].generate (classifier_hypermateters)
    
    
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = features
    
    
    # Create a list where train data indices are -1 and validation data indices are 0
    split_index = [-1 if x in sentences_features_train.index else 0 for x in sentences_features.index]
    
    
    # Predefined split
    # @link https://stackoverflow.com/questions/31948879/using-explicit-predefined-validation-set-for-grid-search-with-sklearn
    ps = PredefinedSplit (test_fold=split_index)
    
    
    # Search space
    search = GridSearchCV (pipe, param_grid, n_jobs=1, cv=ps)
    
    
    # Fit over training data
    search.fit (sentences_features, sentences_labels)
    
    
    # Score
    df = pd.DataFrame (search.cv_results_)
    
    
    # Round decimal places for test metrics
    df = df.round ({'mean_test_score': 5})
    
    
    df["title"] = df["param_features__analyzer"].astype(str) + "-" + df["param_features__ngram_range"].astype(str)
    
    
    df = df.drop ('mean_fit_time', axis=1)
    df = df.drop ('std_fit_time', axis=1)
    df = df.drop ('mean_score_time', axis=1)
    df = df.drop ('std_score_time', axis=1)
    df = df.drop ('param_classifier__selected_model', axis=1)
    df = df.drop ('param_features__min_df', axis=1)
    df = df.drop ('param_features__strip_accents', axis=1)
    df = df.drop ('param_features__sublinear_tf', axis=1)
    df = df.drop ('param_features__use_idf', axis=1)
    df = df.drop ('params', axis=1)
    df = df.drop ('split0_test_score', axis=1)
    df = df.drop ('rank_test_score', axis=1)
    df = df.drop ('std_test_score', axis=1)
    df = df.drop ('param_features__analyzer', axis=1)
    df = df.drop ('param_features__ngram_range', axis=1)
    
    
    print (df.to_latex(index=False))