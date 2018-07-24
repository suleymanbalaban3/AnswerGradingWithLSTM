import numpy as np
import csv
import keras
import sklearn
import gensim
import random
import scipy
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense , Dropout , Activation , Flatten
from keras.layers import merge
from keras import backend as K
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding , LSTM
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC , SVC
from sklearn.naive_bayes import MultinomialNB
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec , TaggedDocument
from keras.utils.np_utils import to_categorical

# size of the word embeddings
embeddings_dim = 200

# maximum number of words to consider in the representations
max_features = 300000

# maximum length of a sentence
max_sent_len = 500

# percentage of the data used for model training
percent = 0.75

# number of classes
num_classes = 5

print ("")
print ("Reading pre-trained word embeddings...")
embeddings = dict( )
#embeddings = Word2Vec.load_word2vec_format( "GoogleNews-vectors-negative300.bin.gz" , binary=True ) 


#embeddings = gensim.models.KeyedVectors.load_word2vec_format('AllDataWithLastUpdatedVectors.txt', binary=False, unicode_errors='ignore')
embeddings = np.load('my_vector_dictionary.npy').item()


print ("Reading text data for classification and building representations...")
data = [ ( row["sentence"] , row["label"]  ) for row in csv.DictReader(open("answer270WithRandomShuffle.txt"), delimiter='\t', quoting=csv.QUOTE_NONE) ]
#random.shuffle( data )
print ("Reading text data for classification and building representations...")
data1 = [ ( row["sentence"] , row["label"]  ) for row in csv.DictReader(open("dataAllTester.txt"), delimiter='\t', quoting=csv.QUOTE_NONE) ]
#random.shuffle( data1 )

print(int(len(data)))
train_size = int(len(data))
test_size = int(len(data1))
train_texts = [ txt.lower() for ( txt, label ) in data[0:train_size] ]
test_texts = [ txt.lower() for ( txt, label ) in data1[0:test_size] ]

train_labels = [ label for ( txt , label ) in data[0:train_size] ]
test_labels = [ label for ( txt , label ) in data1[0:test_size] ]

num_classes = len( set( train_labels + test_labels ) )
print("Number of Classes :" + repr(num_classes))
print("Size of Train :" + repr(train_size))
print("Size of Test :" + repr(len(data) - train_size))

tokenizer = Tokenizer()
#texts = ["The sun is shining in June!","September is grey.","Life is beautiful in August.","I like it","This and other things?"]
tokenizer.fit_on_texts(train_texts)
#print(tokenizer.word_index)
#tokenizer = Tokenizer(nb_words=max_features, filters=keras.keras.preprocessing.text.base_filter(), lower=True, split=" ")
#tokenizer.fit_on_texts(train_texts)
train_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( train_texts ) , maxlen=max_sent_len )
test_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( test_texts ) , maxlen=max_sent_len )
train_matrix = tokenizer.texts_to_matrix( train_texts )
test_matrix = tokenizer.texts_to_matrix( test_texts )
embedding_weights = np.zeros( ( max_features , embeddings_dim ) )
for word,index in tokenizer.word_index.items():
  if index < max_features:
    try: embedding_weights[index,:] = embeddings[word]
    except: embedding_weights[index,:] = np.random.rand( 1 , embeddings_dim )
le = preprocessing.LabelEncoder( )
print("-------------------------- Train text ----------------------------\n" + repr(train_texts))
print("--------------------------- Test text ----------------------------\n" + repr(test_texts))
print("------------------------------------------------------------------\n")
le.fit( train_labels + test_labels )
train_labels = to_categorical( train_labels )
test_labels = to_categorical( test_labels )
#train_labels = le.transform( train_labels )
#test_labels = le.tra                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       nsform( test_labels )
print ("Classes that are considered in the problem : " + repr( le.classes_ ))


print ("Method = Stack of two LSTMs")
np.random.seed(0)
model = Sequential()
model.add(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=True, weights=[embedding_weights] ))
model.add(Dropout(0.25))
model.add(LSTM(activation='sigmoid', units = 200, recurrent_activation='hard_sigmoid', return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM( activation='sigmoid', units = 200, recurrent_activation='hard_sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(6))
model.add(Activation('sigmoid'))
if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='adam')
else: model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
model.fit( train_sequences , train_labels , nb_epoch=40, batch_size=16)
results = model.predict_classes( test_sequences )
print(test_sequences)
print(train_labels)
print(test_labels)
print(results)
model.save_weights('q270DataBatch16Epoch40Shuffled.h5')
K.clear_session()	
#print(accuracy_score(test_labels, results.round(), normalize=False))
#print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results.round() )  ))
#print (sklearn.metrics.classification_report( test_labels , results.round() ))
