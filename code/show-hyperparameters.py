# https://datatofish.com/correlation-matrix-pandas/
import pandas as pd
import numpy
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from pylab import rcParams


# Configure plot
sns.set (style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette (sns.color_palette (HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8



# Binarizer
lb = LabelBinarizer ()


# Read best parameters and results
df = pd.read_pickle ('../results/best_params/params-satire-2017.pkl')


# Configure panda according to the shape
pd.set_option ('display.max_rows', df.shape[0])
pd.set_option ('display.max_columns', df.shape[1])
pd.set_option ('display.width', 1000)


# For displaying the results, remove fixed parameters and training metrics
df = df.drop ('epochs', axis=1)
df = df.drop ('round_epochs', axis=1)
df = df.drop ('optimizer', axis=1)
df = df.drop ('loss', axis=1)
df = df.drop ('maxlen', axis=1)
df = df.drop ('accuracy', axis=1)
 

# Remove weights info from embeddings matrix
df['embedding_matrix'] = df['embedding_matrix'].apply (lambda x: x['key'])



# Round decimal places for test metrics
df = df.round ({'val_accuracy': 5, 'val_loss': 5})

pd.set_option ('display.max_rows', df.shape[0])
pd.set_option ('display.max_columns', df.shape[1])
pd.set_option ('display.width', 1000)

# To calculate the best parameters
print (df.sort_values(by=['val_accuracy']).tail(10))
print (df.sort_values(by=['val_accuracy']).head(10))
df.to_csv ('dp-satire-2017.csv')
sys.exit ()


"""
# To show by features
print (df.groupby (['word_embeddings_architecture']).agg (['count']))
print (df.groupby (['embedding_matrix']).agg (['count']))
sys.exit ()
"""



# Transform to binary the feature space
lb = LabelBinarizer ()
lb.fit (['gru', 'bilstm', 'cnn'])
df = pd.get_dummies (df, columns=['word_embeddings_architecture'])

lb = LabelBinarizer ()
# lb.fit (['none', 'word2vec', 'suc', 'gloVe', 'fastText'])
lb.fit (['custom-word2vec', 'custom-glove', 'custom-fasttext'])
df = pd.get_dummies (df, columns=['embedding_matrix'])
# print (df)
# sys.exit ()


# Rename the binazied columns
df.rename (columns={
    'word_embeddings_architecture_cnn': 'cnn', 
    'word_embeddings_architecture_bilstm': 'bilstm',
    'word_embeddings_architecture_gru': 'gru',
    'word_embeddings_architecture_dense': 'dense'
}, inplace=True)

df.rename (columns={
    'embedding_matrix_custom-word2vec': 'custom_word2vec', 
    'embedding_matrix_custom-glove': 'custom_glove',
    'embedding_matrix_custom-fasttext': 'custom_fasttext',
}, inplace=True)



def label (row):
   if row['gru'] == 1:
      return 'GRU'
   if row['bilstm'] == 1:
      return 'BiLSTM'
   if row['cnn'] == 1:
      return 'CNN'
   if row['dense'] == 1:
      return 'DENSE'
      
   return '-'
   
   
def order (row):
   if row['gru'] == 1:
      return 1
   if row['bilstm'] == 1:
      return 2
   if row['cnn'] == 1:
      return 3
   if row['dense'] == 1:
      return 4
      
   return 5



# Create title
df['title'] = df.apply (lambda row: label(row), axis=1)
df['order'] = df.apply (lambda row: order(row), axis=1)


# Reassign order
df = df[['first_neuron', 'activation', 'batch_size', 'dropout', 'lr', 'number_of_layers', 'reduce_dim', 'trainable', 'title', 'val_accuracy', 'order', 'custom_word2vec', 'custom_glove', 'custom_fasttext']]


# Sort values
df = df.sort_values (by='order', ascending=True)



df.drop(df.loc[df['title']=='delete'].index, inplace=True)

# df =  df.query('suc=="1" ')


tables = [
    df.query('custom_word2vec=="1"'),
    df.query('custom_glove=="1" '),
    df.query('custom_fasttext=="1" ')
]

for table in tables:
    print (table.to_csv(index=False))

# print (df.query('gru=="1" & fastText=="1" ').to_csv(index=False))



"""
df = df.drop ('order', axis=1)
df = df.drop ('suc', axis=1)

sns.heatmap (df.corr (), annot=True)
plt.show ()
"""