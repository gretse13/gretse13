import math
from math import log

import pandas as pd
import os
import nltk
import numpy as np
import nltk.classify.util

from nltk.stem.porter import *

import regressor
from nb_classifier import classifier_train
from scraper import scrape_coindesk

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = f'{ROOT}\\data_raw'


def remove_pattern(input_txt, pattern):
    """
     remove pattern from input txt
    """
    input_txt = str(input_txt)
    pattern = str(pattern)
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


def text_preprocessing(data, name):
    """
        preprocess tweets for sentiment analysis
    """
    # remove mentions
    data['tweet_processed'] = np.vectorize(remove_pattern)(data['tweet'], "@[\w]*")
    # remove punctuations, numbers, special characters
    data.tweet_processed = data.tweet_processed.str.replace("[^a-zA-Z#]", " ", regex=True)
    # remove short words (3 characters or fewer) that may remain
    data.tweet_processed = data.tweet_processed.apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
    # tokenize tweets and add a new column to data
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    data['tweet_processed'] = data['tweet_processed'].apply(lambda x: tokenizer.tokenize(x))
    # get english stopwords
    stopwords = nltk.corpus.stopwords.words("english")
    # remove stopwords from tokenized text
    data['tweet_processed'] = data['tweet_processed'].apply(lambda x: [w for w in x if w.lower() not in stopwords])
    # also remove whitespace and links that may remain in text
    cleaning = ["", "https", ]
    data['tweet_processed'] = data['tweet_processed'].apply(lambda x: [w for w in x if w not in cleaning])

    # normalize text
    stemmer = PorterStemmer()
    data.tweet_processed = data.tweet_processed.apply(lambda x: [stemmer.stem(i) for i in x])

    # make "tweet_processed" column string from list
    data['tweet_processed'] = data['tweet_processed'].apply(lambda x: ' '.join(x))
    # for i in range(len(tweet_processed)):
    #     tmp = tweet_processed[i]
    #     tweet_processed[i] = ' '.join(tweet_processed[i])
    # data.loc['tweet_processed'] = tweet_processed

    # empty string to nan value
    data['tweet_processed'] = data['tweet_processed'].replace(r'^\s*$', np.nan, regex=True)
    data.dropna(inplace=True)

    data.to_csv(path_or_buf=name, index=False)

    return data


def get_data(name):
    """
        get dataframe from csv by name or create csv from raw data and save it
    """
    try:
        # if data.csv exists, simply read it and return to main
        data = pd.read_csv(name)
    except FileNotFoundError:
        # if we can't find the file, create it
        # initializes empty dataframe to concat data to
        data = pd.DataFrame()
        # get list of filenames under data directory to read
        files = os.listdir(DATA_RAW)
        # read raw tweets' csvs and concat them into the data dataframe
        for file in files:
            df = pd.read_csv(f'{DATA_RAW}\\{file}', sep=';')
            data = pd.concat([data, df])
    # save data csv for later use
    data.to_csv(path_or_buf=name, index=False)
    return data


def get_crypto_tweets(data, name):
    """
        return only data that contains crypto keywords
    """
    # initialize empty dataframe
    crypto_only = pd.DataFrame()
    # lowercase tweets
    data['tweet'] = data['tweet'].apply(lambda y: y.lower())
    # find occurrences of crypto names in tweets and save them in new dataframe
    for x in ['crypto', 'cryp', 'bitcoin', 'btc', 'ethereum', 'eth', 'tether', 'usdt', 'block', 'blockchain', 'coin',
              'mine']:
        crypto_only = pd.concat([crypto_only, data[data['tweet'].str.contains(x)]])
    crypto_only.to_csv(path_or_buf=name, index=False)
    return crypto_only


def get_volatility(data):
    """
        calculate volatility from data
    """
    # Date column to datetime object
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    # get unique crypto symbols
    symbols = data['Symbol'].unique()
    # initialize result dataframe
    result = pd.DataFrame()
    # iterate
    for symbol in symbols:
        # get cryptocoin data
        coin_df = data.loc[data['Symbol'] == symbol]
        # natural logarithms
        coin_df['ln_high'] = coin_df['High'].apply(lambda x: log(x))
        coin_df['ln_low'] = coin_df['Low'].apply(lambda x: log(x))
        coin_df['ln_open'] = coin_df['Open'].apply(lambda x: log(x))
        coin_df['ln_close'] = coin_df['Close'].apply(lambda x: log(x))
        close_mean = coin_df['Close'].mean()
        # find volatility
        coin_df.reset_index(inplace=True, drop=True)
        # for i in coin_df.axes[0].values:
        #     if i == 0:
        #         coin_df.loc[0, 'volatility'] = np.nan
        #     else:
        #         # coin_df.loc[i, 'volatility'] = coin_df.loc[i, 'ln_close'] - coin_df.loc[i-1, 'ln_close']
        # coin_df.dropna(inplace=True)
        coin_df['volatility'] = coin_df['Close'].apply(lambda x: close_mean-x)
        coin_df['volatility'] = coin_df['volatility'].apply(lambda x: x**2)
        coin_df[f'vol_direction'] = np.nan
        for i in coin_df.axes[0].values[1:]:
            if coin_df.loc[i-1, 'volatility'] > coin_df.loc[i, f'volatility']:
                coin_df.loc[i, f'vol_direction'] = -1
            elif coin_df.loc[i-1, f'volatility'] < coin_df.loc[i, f'volatility']:
                coin_df.loc[i, f'vol_direction'] = 1
            else:
                coin_df.loc[i, f'vol_direction'] = 0
        coin_df.dropna(inplace=True)
        # append to result dataframe
        result.reset_index(drop=True, inplace=True)
        result = pd.concat([result, coin_df])

    return result


def match_influencers_to_volatility(influencers_data, volatility_data):
    """
        insert volatility into influencers sentiment data
    """
    symbols = volatility_data['Symbol'].unique()
    influencers_data['created_at(utc)'] = pd.to_datetime(influencers_data['created_at(utc)'])
    influencers_data['created_at(utc)'] = influencers_data['created_at(utc)'].apply(lambda t: t.replace(second=0))
    volatility_data['Datetime'] = pd.to_datetime(volatility_data['Datetime'])
    volatility_data['Datetime'] = volatility_data['Datetime'].apply(lambda t: t.replace(second=0))
    influencers_data = influencers_data.loc[influencers_data['created_at(utc)'] >= volatility_data.at[0, 'Datetime']]
    volatility_data.reset_index(inplace=True, drop=True)
    last_day = volatility_data.at[volatility_data.axes[0].values[-1], 'Datetime']
    influencers_data = influencers_data.loc[influencers_data['created_at(utc)'] <= last_day]
    influencers_data.sort_values('created_at(utc)', inplace=True)
    volatility_data.sort_values('Datetime', inplace=True)

    volatility_data = volatility_data.rename(columns={'Datetime': 'time'})
    influencers_data = influencers_data.rename(columns={'created_at(utc)': 'time'})

    for symbol in symbols:
        coin_df = volatility_data.loc[volatility_data['Symbol'] == symbol]
        for i in influencers_data.axes[0].values:
            date = influencers_data.loc[i, 'time']
            date = date.date()
            today_vol = coin_df.loc[coin_df['time'].dt.date == date, ['time', 'volatility', 'vol_direction']]
            today_vol.reset_index(inplace=True, drop=True)
            try:
                min_dif = abs((today_vol.loc[0, 'time'] - influencers_data.loc[i, 'time']).total_seconds())
            except KeyError:
                min_dif = abs((today_vol.loc[0, 'time'] - influencers_data.loc[i, 'time']).total_seconds())
            idx = 0
            for j in today_vol.axes[0].values[1:]:
                time_dif = abs((today_vol.loc[j, 'time'] - influencers_data.loc[i, 'time']).total_seconds())
                if time_dif < min_dif:
                    min_dif = time_dif
                    idx = j
            influencers_data.loc[i, f'{symbol}_volatility'] = today_vol.at[idx, 'volatility']
            influencers_data.loc[i, f'{symbol}_vol_direction'] = today_vol.at[idx, 'vol_direction']
        print()

    return influencers_data


if __name__ == '__main__':
    # download and/or update nltk packages
    # nltk.download(
    #     ["names", "stopwords", "state_union", "twitter_samples", "movie_reviews", "averaged_perceptron_tagger",
    #      "vader_lexicon", "punkt", 'wordnet', 'omw-1.4'])

    # preprocessing and naive bayes
    try:
        # see if data is ready
        influencer_data = pd.read_csv('influencer_data_labeled.csv')
    except FileNotFoundError:
        # if not, create it
        # get raw influencer data
        influencer_data = get_data('influencer_data.csv')
        # keep columns we need
        influencer_data = influencer_data.loc[:, ['author_id', 'username', 'created_at(utc)', 'tweet']]
        # keep only tweets that cointain crypto
        influencer_data = get_crypto_tweets(influencer_data, 'influencer_data_crypto.csv')
        # preprocess data for classifier
        influencer_data = text_preprocessing(influencer_data, 'influencer_data_cleaned.csv')

        # get raw training data (hand labeled)
        training_data = pd.read_excel('labeled_train.xlsx')
        # keep columns we need
        training_data = training_data.loc[:, ['author_id', 'username', 'created_at(utc)', 'tweet', 'sentiment']]
        # drop rows without sentiment label
        training_data.dropna(subset=['sentiment'], inplace=True)
        # preprocess data for classifier
        training_data = text_preprocessing(training_data, 'training_data_cleaned.csv')

        # get amount of training data
        # this will show us how to split merged dataset to train and test, using our hand labeled data for training
        train_idx = training_data.values.shape[0]
        # concat influencer data to training data
        data = pd.concat([training_data, influencer_data])
        # train classifier and return prediction (label for influencer data)
        y_pred = classifier_train(data, train_idx)
        # add label to influencer data
        influencer_data['sentiment'] = pd.Series(y_pred, name='sentiment')
        # save to csv
        influencer_data.to_csv(path_or_buf='influencer_data_labeled.csv', index=False)

    try:
        # try to get scraped crypto data
        data = pd.read_csv('raw_df.csv')
    except FileNotFoundError:
        # if file does not exist, create it using scraper
        scrape_coindesk()
        data = pd.read_csv('raw_df.csv')

    try:
        data = pd.read_csv('regression_data_new.csv')
    except FileNotFoundError:
        # calculate volatility
        data = get_volatility(data)
        # save data
        data.to_csv(path_or_buf='crypto_volatility.csv', index=False)

        # add volatility to influencer data
        influencer_data = match_influencers_to_volatility(influencer_data, data)
        influencer_data.to_csv(path_or_buf='regression_data.csv', index=False)
        influencer_data = influencer_data.drop_duplicates(subset='tweet', keep='first')
        influencer_data.to_csv(path_or_buf='regression_data_new.csv', index=False)

    regressor.le_regress(data)
    data = pd.read_csv('regression_data_new_lag.csv')
    regressor.le_regress(data, lag=True)
