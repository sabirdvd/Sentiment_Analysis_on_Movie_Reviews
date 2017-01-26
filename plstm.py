import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

np.random.seed(0)


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

    batch_size = 128
    #nb_classes = 11
    nb_epoch = 1




if __name__ == "__main__":

    batch_size = 128
    nb_classes = 11
    nb_epoch = 300


    #load data
    train_df = pd.read_csv('train.tsv', sep='\t', header=0)
    test_df = pd.read_csv('test.tsv', sep='\t', header=0)

    raw_docs_train = train_df['Phrase'].values
    raw_docs_test = test_df['Phrase'].values
    sentiment_train = train_df['Sentiment'].values
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
    word_id_train, word_id_len = [], []
    for doc in processed_docs_train:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_train.append(word_ids)
        word_id_len.append(len(word_ids))

    word_id_test, word_ids = [], []
    for doc in processed_docs_test:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_test.append(word_ids)
        word_id_len.append(len(word_ids))
 
    seq_len = np.round((np.mean(word_id_len) + 2*np.std(word_id_len))).astype(int)

    #pad sequences
    word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen=seq_len)
    word_id_test = sequence.pad_sequences(np.array(word_id_test), maxlen=seq_len)
    y_train_enc = np_utils.to_categorical(sentiment_train, num_labels)

    #PLSTM
    print "fitting PLSTM ..."
    from phased_lstm_keras.PhasedLSTM import PhasedLSTM as PLSTM
  
   
    model_PLSTM = Sequential()
    model_PLSTM.add(Embedding(dictionary_size, 128, dropout=0.2))
    model_PLSTM.add(PLSTM(128, consume_less='gpu'))
    model_PLSTM.add(Dense(5, activation='softmax'))
    model_PLSTM.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                        metrics=['accuracy'])
    model_PLSTM.summary()
    acc_PLSTM = AccHistory()
    loss_PLSTM = LossHistory()
    model_PLSTM.fit(word_id_train, y_train_enc, nb_epoch=300, batch_size=128,
    callbacks=[acc_PLSTM, loss_PLSTM])

    #LSTM
    print "fitting LSTM ..."
    model_LSTM = Sequential()
    model_LSTM.add(Embedding(dictionary_size, 128, dropout=0.2))
    model_LSTM.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
    model_LSTM.add(Dense(num_labels))
    model_LSTM.add(Activation('softmax'))
    model_LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    ##model.fit(word_id_train, y_train_enc, nb_epoch=1, batch_size=256, verbose=1)
    
    model_LSTM.summary()
    acc_LSTM = AccHistory()
    loss_LSTM = LossHistory()
    model_LSTM.fit(word_id_train, y_train_enc ,  nb_epoch=300, batch_size=128,
                   callbacks=[acc_LSTM, loss_LSTM])


    test_pred_PLSTM = model_PLSTM.predict_classes(word_id_test) 
    test_pred_LSTM = model_LSTM.predict_classes(word_id_test)
   
   
    # plot results
    plt.figure(1, figsize=(10, 10))
    plt.title('Accuracy on training dataset')
    plt.xlabel('Iterations, batch size ' + str(batch_size))
    plt.ylabel('accuracy')
    plt.plot(acc_LSTM.losses, color='k', label='LSTM')
    plt.hold(True)
    plt.plot(acc_PLSTM.losses, color='r', label='PLSTM')
    plt.savefig('sentiment_plstm_lstm_comparison_acc.png', dpi=100)

    plt.figure(2, figsize=(10, 10))
    plt.title('Loss on training dataset')
    plt.xlabel('Iterations, batch size ' + str(batch_size))
    plt.ylabel('Categorical cross-entropy')
    plt.plot(loss_LSTM.losses, color='k', label='LSTM')
    plt.hold(True)
    plt.plot(loss_PLSTM.losses, color='r', label='PLSTM')
    plt.savefig('sentiment_plstm_lstm_comparison_loss.png', dpi=100)

   
    #result print in csv format (PLSTM) 
    test_df['Sentiment'] = test_pred_LSTM.reshape(-1,1) 
    header = ['PhraseId', 'Sentiment']
    test_df.to_csv('.lstm_sentiment_result.csv', columns=header, index=False, header=True)

    
    #result print in csv format (LSTM)
    test_df['Sentiment'] = test_pred_PLSTM.reshape(-1,1) 
    header = ['PhraseId', 'Sentiment']
    test_df.to_csv('.Plstm_sentiment_result.csv', columns=header, index=False, header=True)



