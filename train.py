import numpy as np
import pandas as pd

from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM

np.random.seed(0)

if __name__ == "__main__":

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
   

    print "converting to token ids..."
    word_id_train, word_id_len = [], []
    for doc in processed_docs_train:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_train.append(word_ids)
        word_id_len.append(len(word_ids))
    # test
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
    print('word_id_train:', word_id_train.shape)
    print(' word_id_test:', word_id_test.shape)

    #LSTM
    print "fitting LSTM ..."
    #first layer in a Sequential model
    model = Sequential()
    model.add(Embedding(dictionary_size, 128, dropout=0.2))
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    print('train...') 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(word_id_train, y_train_enc, nb_epoch=200, batch_size=256, verbose=1)
    
    test_pred = model.predict_classes(word_id_test)
     

    history = model.fit(word_id_train, word_id_train, validation_split=0.33, nb_epoch=200, batch_size=10, verbose=0)
    #list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



    #make a submission
    test_df['Sentiment'] = test_pred.reshape(-1,1)
    header = ['PhraseId', 'Sentiment']
    test_df.to_csv('./lstm_sentiment.csv', columns=header, index=False, header=True)
