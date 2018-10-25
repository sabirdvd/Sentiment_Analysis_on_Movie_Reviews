# Sentiment Analysis based on Movie Reviews using Recurrent Neural Networks (LSTM and PLSTM)
> Dependencies

- python 2.7 
- keras 
- gensim 
- nltk 
- panda 

> Install ``PLSTM``
```
python setup.py build
python setup.py install  
```
> Start ``training``

After making the script ``plstm_lstm_training.py`` executable it can be called with the following flags: ``./plstm_lstm_training.py -d -e ``For more information type: ``./plstm_lstm_training.py -h``

# Background

Kaggle hosted a sentiment analysis competition in February of 2014 for the machine learning community to benchmark their ideas using the Rotten Tomatoes movie review dataset; which is a corpus of movie reviews. The goal was to label phrases on a scale of five values: negative, somewhat negative, neutral, somewhat positive, positive. Obstacles like sentence negation, sarcasm, terseness, language ambiguity, and many others make this task very challenging.

# Challenge 

Regardless of words like cleverness, intelligent, kind and humor are positives words, the phrase is still negative overall. That is why the order of words and the sentence structure must be taken into account to not loose information.
Technical details

The dataset is divided into training data and validation data. The sizes of the validation data is 20%. Each subset is pre-processed by the following steps:

Tokenize each row so that we only have tokens Remove all the stopwords Use a stemmer (SnowballStemmer) to generalize the word Create an id for each token Connect each id with the correct label Kind of one-hot encoding

The preprocessed data is arrays of length 12. Each element corresponds to a token, with a number with the token-id or a zero (which means that there is no token there). In this way each sentence that is being classified is a vector consisting of tokens that each build a sentence. The zero-padding is from left to right, so that the vector contains the information in the end, which benefits the LSTM-/PLSTM-layers. The label for each vector is represented as a 5-element binary vector (one-hot encoding). The input training data consists of the following:

- Dictionary: 13759 different tokens (both training and test) · There are 8544 sentences · There are 156060 sentences, phrases and single tokens · Labels for each of the types above

The input test data consists of the following:

- Dictionary: 13759 tokens (both training and test) · There are 3311 sentences · There are 66292 sentences, phrases and single tokens

Single tokens are classified with the label 2. The output data is the classified label.

The models that have been used are 4 different models, where each model has been tested with different hyperparameters. The models that have been built are the following:

{Embedding-layer,LSTM-layer,Dense-layer,Softmax-layer} {Embedding-layer,PLSTM-layer,Dense-layer,Softmax-layer} {Embedding-layer,LSTM-layer,LSTM-layer,Dense-layer,Softmax-layer} {Embedding-layer,PLSTM-layer,PLSTM-layer,Dense-layer,Softmax-layer}

The objective for calculation of the loss that has been used is the categorical crossentropy, also known as the multiclass logloss. We also shuffle the data-samples for each epoch.

The both optimizers uses moving average which, compared to SGD, allows the algorithms to take bigger steps and therefore they converge faster.
