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
import tensorflow as tf
import datetime
import talos as ta
import sys

from tensorflow import keras

from keras.optimizers import Adam, Nadam, RMSprop
from talos.model.normalizers import lr_normalizer


# Configure seed prior other stuff
# @link https://stackoverflow.com/questions/45230448/how-to-get-reproducible-result-when-running-keras-with-tensorflow-backend
seed_value = 2
os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
random.seed (seed_value)
np.random.seed (seed_value)


# Import KERAS
from keras import backend as K

import csv
import pickle
import argparse
import config
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold

from keras.layers import SpatialDropout1D
from keras.layers import Layer
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import GlobalMaxPool1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.layers import concatenate
from keras.layers import GRU

from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# Parser
parser = argparse.ArgumentParser (description='Evaluate hyperparameters.')
parser.add_argument ('--dataset', dest='dataset', default='satire-2017', help='satire-2017-spain|satire-2017-mexico|satire-2017')

args = parser.parse_args ()
print (args.dataset)



# @var max_words Tokenizer max words
max_words = 50000


# @var minutes How many minutes to limit
minutes = 60 * 24 * 2
# minutes = 60 * 8


# @var maxlen The max number of tokens for each tweet. 
maxlen = 100


# For early stopping
patience = 15


# Train ratio (float or int, default=None)
train_size = 0.8
# train_size = 1000


# Validation ratio
test_size = None
# test_size = 333


"""
   create_embedding_matrix
   
   @param filepath
   @param word_index
   @param embedding_dim
   @param dataset
   @param encoding
   
   @link https://realpython.com/python-keras-text-classification/#your-first-keras-model
"""

def create_embedding_matrix (filepath, word_index, embedding_dim, dataset, encoding="utf8"):
    
    # pretrained_embeddings_dir
    pretrained_embeddings_dir = "./../assets/pretrained_embeddings/"

    
    # Cache file
    cache_file = "./../cache/" + dataset + "_" + filepath + ".npy"
    
    
    # Restore cache
    if (os.path.isfile (cache_file)):
        return np.load (cache_file)
    
    
    # Adding again 1 because of reserved 0 index
    vocab_size = len (word_index) + 1  
    embedding_matrix = np.zeros ((vocab_size, embedding_dim))

    
    # ...
    with open (pretrained_embeddings_dir + filepath, 'r', encoding=encoding, errors = 'ignore') as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array (vector, dtype=np.float32)[:embedding_dim]

    # Store in cache
    np.save (cache_file, embedding_matrix)

    return embedding_matrix
    

"""
   create_model
   
   * @param x_train
   * @param y_train
   * @param embedding_dim
   * @param embedding_matrix
   * @param params
"""
def create_model (x_train, y_train, x_val, y_val, params):

    # Show params
    for key in params:
        if (key == 'embedding_matrix'):
            print (key + ": " + str(params[key]['key']))
        else:
            print (key + ": " + str(params[key]))
            
            
    # Get last activation layer for binary or multiclass prediction
    if (config.number_of_classes[args.dataset] == 1):
        last_activation_layer = 'sigmoid'
        loss_function = 'binary_crossentropy'
        metric = keras.metrics.BinaryAccuracy (name="accuracy")
    
    # Multiclass
    else:
        last_activation_layer = 'softmax'
        loss_function = 'categorical_crossentropy'
        metric = keras.metrics.CategoricalAccuracy (name='accuracy')
        

    # Get the embedding dim
    embedding_dim = params['embedding_matrix']['weights'].shape[1] if params['embedding_matrix']['key'] != 'none' else 300 
    
    
    # Get the weights
    weights = [params['embedding_matrix']['weights']] if params['embedding_matrix']['key'] != 'none' else None
    

    # Input layer
    main_embeddings_input = Input (shape=(None,))
    
    
    # Embedding layer
    main_embeddings_1 = Embedding (
        input_dim=vocab_size, 
        output_dim=embedding_dim, 
        weights=weights, 
        input_length=params['maxlen'],
        trainable=params['trainable']
    )(main_embeddings_input)


    # First neuron size
    first_neuron = params['first_neuron']
    
    
    # Second neuron size
    if (params['reduce_dim']):
        second_neuron = first_neuron / 2;
    else:
        second_neuron = first_neuron;
    

    # Generate submodel
    # Dense
    if (params['word_embeddings_architecture'] == 'dense'):
        main_embeddings_layer = GlobalMaxPool1D ()(main_embeddings_1)
    
    
    # CNN
    if (params['word_embeddings_architecture'] == 'cnn'):
        main_embeddings_2 = SpatialDropout1D (params['dropout'])(main_embeddings_1)
        main_embeddings_3 = Conv1D (first_neuron, 5, activation=params['activation'])(main_embeddings_2)
        main_embeddings_layer = GlobalMaxPool1D ()(main_embeddings_3)
    
    # BiLSTM
    if (params['word_embeddings_architecture'] == 'bilstm'):
        main_embeddings_2 = SpatialDropout1D (params['dropout'])(main_embeddings_1)
        main_embeddings_layer = Bidirectional (LSTM (first_neuron, dropout=params['dropout'], recurrent_dropout=params['dropout']))(main_embeddings_2)

    # GRU
    if (params['word_embeddings_architecture'] == 'gru'):
        main_embeddings_2 = SpatialDropout1D (params['dropout'])(main_embeddings_1)
        main_embeddings_layer = Bidirectional (GRU (first_neuron, dropout=params['dropout'], return_sequences=False))(main_embeddings_2)
    
    
    # Concatenate Deep layers
    x = main_embeddings_layer
    for i in range (params['number_of_layers']):
        x = Dense (first_neuron if i == 1 else second_neuron, activation=params['activation'])(x)
        if (params['dropout']):
            x = Dropout (params['dropout'])(x)
    main_embeddings_layer = x
    
    
    # Inputs
    inputs = [main_embeddings_input]
        
    
    # Outputs
    outputs = Dense (config.number_of_classes[args.dataset], activation=last_activation_layer)(main_embeddings_layer)
    
    
    # Create model
    model = Model (inputs=inputs, outputs=outputs, name=name)
    
    
    # Define the metrics we want to obtain from our classifier
    metrics = [metric]
    
    
    # Optimizer
    optimizer = params['optimizer'](lr=lr_normalizer (params['lr'], params['optimizer']))
    

    # Compile model
    model.compile (optimizer=optimizer, loss=loss_function, metrics=metrics)

    
    # @var early_stopping Early Stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping (
        monitor='val_loss', 
        verbose=1,
        patience=patience,
        restore_best_weights=True
    )
    

    # Fit model
    history = model.fit (
        x=x_train, 
        y=y_train,
        validation_data=(x_val, y_val),
        batch_size = params['batch_size'],
        epochs = params['epochs'],
        callbacks = [early_stopping]
    )


    # finally we have to make sure that history object and model are returned
    return history, model


# @var umucorpus_ids int|string The Corpus IDs
for key, umucorpus_ids in config.ids[args.dataset].items ():

    print ("Processing " + key)
    
    
    # @var label_filename String 
    label_filename = '../results/binarizer/' + key + '.pickle'
    
    
    # @var token_filename String 
    token_filename = '../results/tokens/' + key + '_' + str (max_words) + '.pickle'
    
    
    # @var model_weights_filename String 
    model_weights_filename = '../results/models/{{ name }}.h5';
    
    
    # @var model_config_filename String 
    model_config_filename = '../results/models/{{ name }}.json';


    # @var confussion_matrix String 
    confussion_matrix = '../results/confussion_matrix/{{ name }}.txt';


    # @var architecture_diagram_filename String 
    architecture_diagram_filename = '../results/architecture/{{ name }}.png';


    # @var plot_filename String 
    plot_filename = '../results/plots/{{ name }}.png';
    
   
    # @var train_filename String 
    train_filename = './../data/' + key + '.csv'
    
    
    # Transform to binary
    lb = LabelBinarizer ()


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
    
    
    
    # Notice that labels are sorted alphabetically
    lb.fit (['non-satire', 'satire'])

    with np.printoptions (threshold=np.inf):
        print (lb.classes_)

    
    
    # Store binarizer
    with open (label_filename, 'wb') as handle:
        pickle.dump (lb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    # Fit dev in the same way as train set
    sentences_labels_binary = lb.transform (sentences_labels)
    sentences_labels_train_binary = lb.transform (sentences_labels_train)
    sentences_labels_val_binary = lb.transform (sentences_labels_val)
    print ("\t...Labels binarized")
    

    # @var Tokenizer
    tokenizer = Tokenizer (num_words = max_words)
    
    
    # Fit on trainin dataset
    tokenizer.fit_on_texts (sentences_features_train)


    # Remove words which do not happer too often in the training dataset
    # https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/49654
    count_thres = 3
    low_count_words = [w for w, c in tokenizer.word_counts.items() if c < count_thres]
    for w in low_count_words:
        del tokenizer.word_index[w]
        del tokenizer.word_docs[w]
        del tokenizer.word_counts[w]
    
    print ("\t...Found %d unique words in the training set: " % len (tokenizer.word_index))


    # Vocabulary
    tokens_ordered_by_frequency = {k: v for k, v in sorted (tokenizer.word_counts.items(), key=lambda item: item[1], reverse=True)}
    how_many = 20
    names = list (tokens_ordered_by_frequency.keys ())[:how_many]
    values = list (tokens_ordered_by_frequency.values ())[:how_many]
    
    print (names)
    print (values)
    """
    plt.bar (range(how_many),values,tick_label=names)
    plt.show ()
    sys.exit ()
    """
    
    
    # Store tokenizer
    with open (token_filename, 'wb') as handle:
        pickle.dump (tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # Update to tokens
    sentences_features_train = tokenizer.texts_to_sequences (sentences_features_train)
    sentences_features_val = tokenizer.texts_to_sequences (sentences_features_val)
    sentences_features = tokenizer.texts_to_sequences (sentences_features)


    # Get the vocab size
    vocab_size = len (tokenizer.word_index) + 1
    print ("\t...Tokenized. Size of the tokenizer: ", vocab_size)


    # Padding
    sentences_features_train = pad_sequences (sentences_features_train, padding='post', maxlen=maxlen)
    sentences_features_val = pad_sequences (sentences_features_val, padding='post', maxlen=maxlen)
    sentences_features = pad_sequences (sentences_features, padding='post', maxlen=maxlen)
    print ("\t...Padding generated. Shape: ", sentences_features_train.shape, sentences_features_val.shape)


    # Embeddings weights
    # gloVe
    # @link https://github.com/dccuchile/spanish-word-embeddings#fasttext-embeddings-from-spanish-wikipedia
    glove_embedding_matrix = create_embedding_matrix ("glove-sbwc.i25.vec", tokenizer.word_index, 300, args.dataset)
    nonzero_elements = np.count_nonzero (np.count_nonzero (glove_embedding_matrix, axis=1))
    print ("\t...Pretrained glove coverage: ", nonzero_elements / vocab_size)


    # Spanish Unannotated Corpora
    # @link https://github.com/dccuchile/spanish-word-embeddings#fasttext-embeddings-from-spanish-wikipedia
    suc_embedding_matrix = create_embedding_matrix ("embeddings-l-model.vec", tokenizer.word_index, 300, args.dataset)
    nonzero_elements = np.count_nonzero (np.count_nonzero (suc_embedding_matrix, axis=1))
    print ("\t...Pretrained fasttext (suc) coverage: ", nonzero_elements / vocab_size)


    # Word2Vec
    w2v_embedding_matrix = create_embedding_matrix ("SBW-vectors-300-min5.txt", tokenizer.word_index, 300, args.dataset)
    nonzero_elements = np.count_nonzero (np.count_nonzero (w2v_embedding_matrix, axis=1))
    print ("\t...pretrained word2vec coverage: ", nonzero_elements / vocab_size)
    
    
    # FastText
    fasttext_embedding_matrix = create_embedding_matrix ("cc.es.300.vec", tokenizer.word_index, 300, args.dataset)
    nonzero_elements = np.count_nonzero (np.count_nonzero (fasttext_embedding_matrix, axis=1))
    print ("\t...pretrained fast text coverage: ", nonzero_elements / vocab_size)
    

    # @var name String
    name = key
    
    
    # Parameter space
    p = {
        'epochs': [1000],
        'lr': (0.5, 2, 10),
        'optimizer': [Adam],
        'trainable': [True, False],
        'number_of_layers': [1, 2, 3, 4],
        'first_neuron': [8, 16, 48, 64, 128, 256],
        'reduce_dim': [True, False],
        'batch_size': [16, 32, 64],
        'dropout': [False, 0.2, 0.5, 0.8],
        'maxlen': [maxlen],
        'word_embeddings_architecture': ['cnn', 'bilstm', 'gru', 'dense'],
        'activation': ['relu', 'sigmoid', 'tanh', 'selu', 'elu'],
        'embedding_matrix': [
            {'key': 'none', 'weights': None}, 
            {'key': 'fastText', 'weights': fasttext_embedding_matrix}, 
            {'key': 'word2vec', 'weights': w2v_embedding_matrix}, 
            {'key': 'gloVe', 'weights': glove_embedding_matrix},  
            {'key': 'suc', 'weights': suc_embedding_matrix}
        ] 
    }


    # and run the experiment
    scan_object = ta.Scan (
        x=sentences_features_train, 
        y=sentences_labels_train_binary,
        x_val=sentences_features_val,
        y_val=sentences_labels_val_binary,
        model=create_model,
        params=p,
        experiment_name=args.dataset,
        time_limit=(datetime.datetime.now () + datetime.timedelta (minutes = minutes)).strftime ("%Y-%m-%d %H:%M"),
        reduction_metric='val_loss',
        minimize_loss=True,
        print_params=False,
        round_limit=200,
        seed=seed_value
    )
    
    
    print ("-------------------------");
    print (scan_object.data)
    print (scan_object.details)
    
    
    # get best model based on loss
    best_model = scan_object.best_model (metric='val_loss', asc=False)
    
    
    # accessing epoch entropy values for each round
    print ("")
    print ("learning entropy")
    print ("-------------------------");
    print (scan_object.learning_entropy)
    
    
    # Evaluate
    print ("")
    print ("evaluate")
    print ("-------------------------");
    evaluate_object = ta.Evaluate (scan_object)
    results = evaluate_object.evaluate (
        sentences_features_val, 
        sentences_labels_val_binary, 
        folds=10, 
        metric='val_loss',
        asc=True,
        task="binary"
    )
    print (results)
    
    
    print ("")
    print ("classification report")
    print ("-------------------------");
    p = ta.Predict (scan_object)
    predictions = p.predict (sentences_features, metric='val_loss', asc=True)
    print (classification_report (sentences_labels_binary, predictions.argmax (axis=-1)))
    
    
    
    # Deploy
    ta.Deploy (scan_object, args.dataset + '_model', metric='loss', asc=False)
    
    
    # Store scan results for further analysis
    scan_object.data.to_pickle ('../results/best_params/params-' + name + '.pkl')
    