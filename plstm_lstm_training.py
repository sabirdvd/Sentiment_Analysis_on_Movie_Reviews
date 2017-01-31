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
def main(argv):
    #set default values
    nb_epoch = 1
    DROPOUT = 0
    dryRun = False
    try:
      opts, args = getopt.getopt(argv,"htd:e:",["dropout=","epochs="])
    except getopt.GetoptError:
      print 'usage:'
      print 'plstm_validation.py -d <drop-out rate> -e <nr of epochs> '
      print 'usage test run with 1% of data:'
      print 'plstm_validation.py -t -d <drop-out rate> -e <nr of epochs> '      
      sys.exit(2)
    for opt, arg in opts: 
      # -t: dry run where we only process part of the data 
      if opt == '-t':
         dryRun = True
      # -h: get help
      elif opt == '-h':
         print 'This python script predicts the sentiment of Rotten Tomatoes reviews using LTSM and PLTSM neural networks.'
         print 'Basic usage: plstm_validation.py -d <drop-out rate> -e <nr of epochs> '
         print 'Basic usage: plstm_validation.py -d <drop-out rate> -e <nr of epochs> '
         print 'Test run: plstm_validation.py -t -d <drop-out rate> -e <nr of epochs> '
         sys.exit()
      elif opt in ("-d", "--dropout"):
         if(float(arg)>=1.0 or float(arg)<0.0):
            print "Invalid value for drop-out ratio!"
            sys.exit(-2)
         DROPOUT = float(arg)
      elif opt in ("-e", "--epochs"):
         if(int(arg)<=0):
            print "Invalid value for number of epochs!"
            sys.exit(-2)        
         nb_epoch = int(arg)
    print 'Nr of epochs is ', nb_epoch
    print 'Drop-out rate is ', DROPOUT
    if(dryRun):
        print 'This is a test run with only 1% of the data.'
        
    batch_size = 128
    np.random.seed(0)
    
    #read the data sets from tab delimited text files
    train_df = pd.read_csv('train.tsv', sep='\t', header=0)
    test_df = pd.read_csv('test.tsv', sep='\t', header=0)
  
    #extract the raw values from the data sets  
    
    #The phrases (a sequence of words) of the training set
    raw_docs_train = train_df['Phrase'].values

    #The phrases (a sequence of words) of the test set    
    raw_docs_test = test_df['Phrase'].values
    
    #The response value, i.e. the output for the training data
    #For the test data set there are no labels available!
    sentiment_train = train_df['Sentiment'].values

    #In the dry run, we restrict the raw values to 1% of the original values
    if(dryRun == True):
        maxIndex = int(math.ceil(len(raw_docs_train)*0.01))
        raw_docs_train=raw_docs_train[1:maxIndex]
        raw_docs_test=raw_docs_test[1:maxIndex]
        sentiment_train=sentiment_train[1:maxIndex]
   
    #the number of labels (5)
    num_labels = len(np.unique(sentiment_train))

    #text pre-processing
    
    #Remove all words that are characterized as stopwords
    stop_words = set(stopwords.words('english'))
    
    #Also remove all interpunctuation marks.
    stop_words.update(['!','#','.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

    #Here we do "stemming":
    #replace nouns by their infinitive,
    #replace nouns by their singular.
    stemmer = SnowballStemmer('english')

    print "pre-processing of phrases from the training set..."

    processed_docs_train = []
    for doc in raw_docs_train:
       tokens = word_tokenize(doc)
       filtered = [word for word in tokens if word not in stop_words and word.isalpha()]
       stemmed = [stemmer.stem(word) for word in filtered]
       processed_docs_train.append(stemmed)
   
    print "pre-processing of phrases from the test set..."

    processed_docs_test = []
    for doc in raw_docs_test:
       tokens = word_tokenize(doc)
       filtered = [word for word in tokens if word not in stop_words and word.isalpha()]
       stemmed = [stemmer.stem(word) for word in filtered]
       processed_docs_test.append(stemmed)

    #Here we  first concatenate all processed words...
    processed_docs_all = np.concatenate((processed_docs_train, processed_docs_test), axis=0)

    #Then we transform the list of processed words into a dictionary.
    #A dictionary is a mapping between words and their frequency:
    #The word is represented by an integer Id.
    #Compare: http://radimrehurek.com/gensim/corpora/dictionary.html 
    #http://radimrehurek.com/gensim/tut1.html
    
    dictionary = corpora.Dictionary(processed_docs_all)
    dictionary_size = len(dictionary.keys())
    print "dictionary size: ", dictionary_size 
    
    #save the dictionary as a binary file
    dictionary.save('dictionary.dict')
    #save the dictionary as a text file (tab delimited)
    dictionary.save_as_text('dictionary.txt')
            
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
    
    #maximum sequence length
    seq_len = 500
    
    #pad sequences
    word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen=seq_len)
    word_id_test = sequence.pad_sequences(np.array(word_id_test), maxlen=seq_len)
    y_train_enc = np_utils.to_categorical(sentiment_train, num_labels)
    
    #PLSTM        
    print "building PLSTM ..."
    model_PLSTM = Sequential()
    model_PLSTM.add(Embedding(dictionary_size, 128))
    model_PLSTM.add(Dropout(DROPOUT))
    #model_PLSTM.add(PLSTM(64, consume_less='gpu'))
    model_PLSTM.add(PLSTM(64))
    model_PLSTM.add(Dense(5, activation='softmax'))
    model_PLSTM.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                        metrics=['accuracy'])

    model_PLSTM.summary()
    acc_PLSTM = AccHistory()
    loss_PLSTM = LossHistory()
    
    print "fitting PLSTM ..."
    history_plstm=model_PLSTM.fit(x=word_id_train, y=y_train_enc,
    nb_epoch=nb_epoch, batch_size=128, verbose=1, validation_split=0.2,
    callbacks=[acc_PLSTM, loss_PLSTM])
    
    #save results to text file
    fileName="PLSTM_accuracy_and_loss_epochs_"+str(nb_epoch)+"_drop_out_"+str(DROPOUT)+".txt"
    index = range(1,1+len(np.asarray(loss_PLSTM.losses)))
    columns = ['accuracy_PLSTM','loss_PLSTM']
    df = pd.DataFrame(
    {'accuracy_PLSTM': np.asarray(acc_PLSTM.losses),
    'loss_PLSTM': np.asarray(loss_PLSTM.losses)},
    index=index, columns=columns)
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
    model_LSTM.add(Embedding(dictionary_size, 128))
    model_LSTM.add(Dropout(DROPOUT))
    model_LSTM.add(LSTM(64))
    model_LSTM.add(Dense(num_labels))
    model_LSTM.add(Activation('softmax'))
    model_LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_LSTM.summary()
    acc_LSTM = AccHistory()
    loss_LSTM = LossHistory()
        
    history_lstm=model_LSTM.fit(x=word_id_train, y=y_train_enc,
    nb_epoch=nb_epoch, batch_size=128, verbose=1, validation_split=0.2,
    callbacks=[acc_LSTM, loss_LSTM])

    #save results to text file
    fileName="both_models_accuracy_and_loss_epochs_"+str(nb_epoch)+"_drop_out_"+str(DROPOUT)+".txt"
    index = range(1,1+len(np.asarray(loss_PLSTM.losses)))
    print index
    columns = ['accuracy_PLSTM','loss_PLSTM','accuracy_LSTM','loss_LSTM']
    print columns
    df = pd.DataFrame(
    {'accuracy_PLSTM': np.asarray(acc_PLSTM.losses),
    'loss_PLSTM': np.asarray(loss_PLSTM.losses),
    'accuracy_LSTM': np.asarray(acc_LSTM.losses),
    'loss_LSTM': np.asarray(loss_LSTM.losses)    
    },
    index=index, columns=columns)
    print df
    df.to_csv(fileName, sep=';')
        
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
    figName = "LSTM_loss_train_vs_validation_drop_out"+str(DROPOUT)+".svg"
    plt.savefig(figName, dpi=200)
    plt.clf() 
    
    # summarize history for accuracy combining both models
    plt.figure(5, figsize=(10, 10))
    plt.plot(history_lstm.history['acc'])
    plt.plot(history_lstm.history['val_acc'])
    plt.plot(history_plstm.history['acc'])
    plt.plot(history_plstm.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['LSTM train', 'LSTM validation','PLSTM train', 'PLSTM validation'], loc='upper left')
    figName = "Both_models_accuracy_drop_out"+str(DROPOUT)+".svg"
    plt.savefig(figName, dpi=200)
    plt.clf()
    
    # summarize history for loss combining both models
    plt.figure(6, figsize=(10, 10))
    plt.plot(history_lstm.history['loss'])
    plt.plot(history_lstm.history['val_loss'])
    plt.plot(history_plstm.history['loss'])
    plt.plot(history_plstm.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['LSTM train', 'LSTM validation','PLSTM train', 'PLSTM validation'],
    loc='upper left')
    figName = "Both_models_loss_drop_out"+str(DROPOUT)+".svg"
    plt.savefig(figName, dpi=200)
    plt.clf()
    
    # plot results for training dataset
    plt.figure(7, figsize=(10, 10))
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
      
    plt.figure(8, figsize=(10, 10))
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
       
if __name__ == "__main__":
    main(sys.argv[1:])    
   
