import requests
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta


def scrape_coindesk():
    """scrape coindesk for Open, Close, Low, High data for each coin in crypto_list"""
    crypto_list = ['BTC', 'ETH', 'USDT']

    raw_df = pd.DataFrame()
    for coin in crypto_list:
        coin_df = pd.DataFrame()
        temp = pd.DataFrame(index=[0])

        starting_point = datetime(2021, 7, 31, 0, 0)
        end_point = datetime(2022, 7, 31, 0, 0)

        while len(temp) > 0:
            if end_point == starting_point:
                break
            start_dt = end_point - relativedelta(days=1)
            url = 'https://production.api.coindesk.com/v2/price/values/' + coin + '?start_date=' + start_dt.strftime(
                "%Y-%m-%dT%H:%M") + '&end_date=' + end_point.strftime("%Y-%m-%dT%H:%M") + '&ohlc=true'
            temp_data_json = requests.get(url)
            temp_data = temp_data_json.json()
            temp = pd.DataFrame(temp_data['data']['entries'])
            temp.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close']

            temp = temp.drop(['Timestamp'], axis=1)
            temp['Datetime'] = [end_point - relativedelta(minutes=len(temp) - i) for i in range(0, len(temp))]
            coin_df = temp.append(coin_df)
            end_point = start_dt
        coin_df['Symbol'] = coin
        raw_df = raw_df.append(coin_df)
    raw_df = raw_df[['Datetime', 'Symbol', 'Open', 'High', 'Low', 'Close']].reset_index(drop=True)
    raw_df.to_csv('raw_df.csv', index=False)
