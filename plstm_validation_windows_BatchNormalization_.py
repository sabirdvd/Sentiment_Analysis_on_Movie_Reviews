#!/usr/bin/python
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, getopt
import math
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.callbacks import Callback
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM
from phased_lstm_keras.PhasedLSTM import PhasedLSTM as PLSTM
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import adam
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2

    
class AccHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('acc'))

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

#read arguments from command line
if __name__ == "__main__":
    #set default values
    nb_epoch = 200
    # TODO Update
    DROPOUT = 0.6
    dryRun = False
    batch_size = 128
    nb_classes = 11#TODO ???
    np.random.seed(0)
    
    #load data
    train_df = pd.read_csv('train.tsv', sep='\t', header=0)
    test_df = pd.read_csv('test.tsv', sep='\t', header=0)
  
    raw_docs_train = train_df['Phrase'].values
    raw_docs_test = test_df['Phrase'].values
    sentiment_train = train_df['Sentiment'].values

    if(dryRun == True):
        maxIndex = int(math.ceil(len(raw_docs_train)*0.01))
        raw_docs_train=raw_docs_train[1:maxIndex]
        raw_docs_test=raw_docs_test[1:maxIndex]
        sentiment_train=sentiment_train[1:maxIndex]
   
    num_labels = len(np.unique(sentiment_train))

    #text pre-processing
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    stemmer = SnowballStemmer('english')

    print "pre-processing train docs..."
    processed_docs_train = []
    for doc in raw_docs_train:
       tokens = word_tokenize(doc)
       filtered = [word for word in tokens if word not in stop_words]
       stemmed = [stemmer.stem(word) for word in filtered]
       processed_docs_train.append(stemmed)
   
    print "pre-processing test docs..."
    processed_docs_test = []
    for doc in raw_docs_test:
       tokens = word_tokenize(doc)
       filtered = [word for word in tokens if word not in stop_words]
       stemmed = [stemmer.stem(word) for word in filtered]
       processed_docs_test.append(stemmed)

    processed_docs_all = np.concatenate((processed_docs_train, processed_docs_test), axis=0)

    dictionary = corpora.Dictionary(processed_docs_all)
    dictionary_size = len(dictionary.keys())
    print "dictionary size: ", dictionary_size 
    #dictionary.save('dictionary.dict')
    #corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    print "converting to token ids..."
    print "processed docs train", len(processed_docs_train)
    print "processed docs test", len(processed_docs_test)
    
    word_id_train, word_id_len = [], []
    for doc in processed_docs_train:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_train.append(word_ids)
        word_id_len.append(len(word_ids))
    
    print "processed word ids train", len(word_id_train)
    
    word_id_test, word_ids = [], []
    for doc in processed_docs_test:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_test.append(word_ids)
        word_id_len.append(len(word_ids))
 
    print "processed word ids test", len(word_id_test)
    
    seq_len = np.round((np.mean(word_id_len) + 2*np.std(word_id_len))).astype(int)

    #pad sequences
    word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen=seq_len)
    word_id_test = sequence.pad_sequences(np.array(word_id_test), maxlen=seq_len)
    y_train_enc = np_utils.to_categorical(sentiment_train, num_labels)
    
    #PLSTM        
    print "building PLSTM ..."
    model_PLSTM = Sequential()
    model_PLSTM.add(Embedding(dictionary_size, 512, dropout=0.6))
    model_PLSTM.add(PLSTM(512, consume_less='gpu'))
    model_PLSTM.add(Dense(5, activation='softmax'))
    #optimizer=RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
    #model_PLSTM.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #adam optimizer 
    optimizer = adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model_PLSTM.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'])
    model_PLSTM.add(Dense(64, input_dim=64, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))

    #model_PLSTM.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                        #metrics=['accuracy'])
    #Early stopping 
    #early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    #model_PLSTM.fit(word_id_train, y_train_enc, validation_split=0.2, callbacks=[early_stopping])


    model_PLSTM.summary()
    acc_PLSTM = AccHistory()
    loss_PLSTM = LossHistory()
    
    print "fitting PLSTM ..."
    history_plstm=model_PLSTM.fit(x=word_id_train, y=y_train_enc,
    nb_epoch=nb_epoch, batch_size=128, 
    shuffle=True, verbose=1, validation_split=0.25,
    callbacks=[acc_PLSTM, loss_PLSTM])

    #history_plstm=model_PLSTM.fit(x=word_id_train, y=y_train_enc,
    #nb_epoch=nb_epoch, batch_size=128, 
    #shuffle=True, verbose=1, validation_split=0.25,
    #callbacks=[early_stopping])
    
    ##save results to text file
    fileName="PLSTM_accuracy_and_loss_epochs_"+str(nb_epoch)+"_drop_out_"+str(DROPOUT)+".txt"
    index = range(1,1+len(np.asarray(loss_PLSTM.losses)))
    print index
    columns = ['accuracy_PLSTM','loss_PLSTM']
    print columns
    df = pd.DataFrame(
    {'accuracy_PLSTM': np.asarray(acc_PLSTM.losses),
    'loss_PLSTM': np.asarray(loss_PLSTM.losses)},
    index=index, columns=columns)
    print df
    df.to_csv(fileName, sep=';')

    # summarize history for accuracy
    plt.figure(1, figsize=(10, 10))
    plt.plot(history_plstm.history['acc'])
    plt.plot(history_plstm.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    figName = "PLSTM_accuracy_train_vs_validation_drop_out"+str(DROPOUT)+".svg"
    plt.savefig(figName, dpi=200)
    plt.clf()
    
    # summarize history for loss
    plt.figure(2, figsize=(10, 10))
    plt.plot(history_plstm.history['loss'])
    plt.plot(history_plstm.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    figName = "PLSTM_loss_train_vs_validation_drop_out"+str(DROPOUT)+".svg"
    plt.savefig(figName, dpi=200)
    plt.clf()

    #LSTM
    print "fitting LSTM ..."
    model_LSTM = Sequential()
    model_LSTM.add(Embedding(dictionary_size, 512, dropout=DROPOUT))
    #model_LSTM.add(Embedding(dictionary_size, 128, dropout=0.2))
    model_LSTM.add(LSTM(512, dropout_W=DROPOUT, dropout_U=DROPOUT))
    model_LSTM.add(Dense(num_labels))
    model_LSTM.add(BatchNormalization())
    model_LSTM.add(Activation('softmax'))
    optimizer = adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #model_LSTM.add(Dense(64, input_dim=64, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    model_LSTM.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'])
   

    ## def
    #keras.layers.normalization.BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, 
    #beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)
    
    #def mode 
    #model_LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Early stopping 
    #early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    #model_LSTM.fit(word_id_train, y_train_enc, validation_split=0.25, callbacks=[early_stopping])

    model_LSTM.summary()
    acc_LSTM = AccHistory()
    loss_LSTM = LossHistory()
        
    history_lstm=model_LSTM.fit(x=word_id_train, y=y_train_enc,
    nb_epoch=nb_epoch, batch_size=128, 
    shuffle=True, verbose=1, validation_split=0.25,
    callbacks=[acc_LSTM, loss_LSTM])
   
    ##Early stopping run 
    #history_lstm=model_LSTM.fit(x=word_id_train, y=y_train_enc,
    #nb_epoch=nb_epoch, batch_size=128, 
    #shuffle=True, verbose=1, validation_split=0.25,
    #callbacks=[early_stopping])

    #save results to text file
    #fileName="both_models_accuracy_and_loss_epochs_"+str(nb_epoch)+"_drop_out_"+str(DROPOUT)+".txt"
    #index = range(1,1+len(np.asarray(loss_PLSTM.losses)))
    #print index
    #columns = ['accuracy_PLSTM','loss_PLSTM','accuracy_LSTM','loss_LSTM']
    #print columns
    #df = pd.DataFrame(
    #{'accuracy_PLSTM': np.asarray(acc_PLSTM.losses),
    #'loss_PLSTM': np.asarray(loss_PLSTM.losses),
    #'accuracy_LSTM': np.asarray(acc_LSTM.losses),
    #'loss_LSTM': np.asarray(loss_LSTM.losses)    
    #},
    #index=index, columns=columns)
    #print df
    #df.to_csv(fileName, sep=';')
        
    # summarize history for accuracy
    plt.figure(3, figsize=(10, 10))
    plt.plot(history_lstm.history['acc'])
    plt.plot(history_lstm.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    figName = "LSTM_accuracy_train_vs_validation_drop_out"+str(DROPOUT)+".svg"
    plt.savefig(figName, dpi=200)
    plt.clf()
    
    # summarize history for loss
    plt.figure(4, figsize=(10, 10))
    plt.plot(history_lstm.history['loss'])
    plt.plot(history_lstm.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    figName = "LSTM_loss_train_vs_validation"+str(DROPOUT)+".svg"
    plt.savefig(figName, dpi=200)
    plt.clf() 
       
    # plot results for training dataset
    plt.figure(5, figsize=(10, 10))
    plt.title('Accuracy on validation set')
    plt.xlabel('Iterations, batch size ' + str(batch_size))
    plt.ylabel('accuracy')
    plt.plot(acc_LSTM.losses, color='k', label='LSTM')
    plt.hold(True)
    plt.plot(acc_PLSTM.losses, color='r', label='PLSTM')
    plt.legend(['LSTM', 'PLSTM'], loc='upper left')
    figName = 'sentiment_plstm_lstm_comparison_acc'+str(DROPOUT)+'.png'
    plt.savefig(figName, dpi=200)
    plt.clf()  
      
    plt.figure(6, figsize=(10, 10))
    plt.title('Loss on validation set')
    plt.xlabel('Iterations, batch size ' + str(batch_size))
    plt.ylabel('Categorical cross-entropy')
    plt.plot(loss_LSTM.losses, color='k', label='LSTM')
    plt.hold(True)
    plt.plot(loss_PLSTM.losses, color='r', label='PLSTM')
    plt.legend(['LSTM', 'PLSTM'], loc='upper left')
    figName = 'sentiment_plstm_lstm_comparison_loss'+str(DROPOUT)+'.png'
    plt.savefig(figName, dpi=200)
    plt.clf()

    #calculate predictions for the test set
    test_pred_PLSTM = model_PLSTM.predict_classes(word_id_test) 
    test_pred_LSTM = model_LSTM.predict_classes(word_id_test)
    
    if(dryRun == False):
        #result print in csv format (PLSTM) 
        test_df['Sentiment'] = test_pred_LSTM.reshape(-1,1) 
        header = ['PhraseId', 'Sentiment']
        fileName = 'Lstm_sentiment_prediction_results_drop_out_'+str(DROPOUT)+'.csv'
        test_df.to_csv(fileName, columns=header, index=False, header=True)
        
        #result print in csv format (LSTM)
        test_df['Sentiment'] = test_pred_PLSTM.reshape(-1,1) 
        header = ['PhraseId', 'Sentiment']
        fileName = 'Plstm_sentiment_prediction_results_drop_out_'+str(DROPOUT)+'.csv'
        test_df.to_csv(fileName, columns=header, index=False, header=True)    
     
