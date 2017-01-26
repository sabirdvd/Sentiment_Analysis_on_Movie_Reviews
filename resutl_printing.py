    # printing the result
    test_df['Sentiment'] = test_pred.reshape(-1,1)
    header = ['PhraseId', 'Sentiment']
    test_df.to_csv('./lstm_output.csv', columns=header, index=False, header=True)
