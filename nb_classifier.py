from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def classifier_train(data, train_idx):
    """simple multinomial naive bayes classifier for sentiment analysis"""
    cv = CountVectorizer()
    X = cv.fit_transform(data['tweet_processed'])

    x_train = X[0:train_idx]
    x_test = X[train_idx+1:, :]
    y = data['sentiment'].values[0:train_idx]

    model = MultinomialNB()
    model.fit(x_train, y)
    y_pred = model.predict(x_test)

    return y_pred
