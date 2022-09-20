import pandas as pd
import matplotlib.pyplot as plt

""" simple script to plot Influencer Sentiment Frequency"""

df = pd.read_csv('regression_data_new.csv')
fig, ax = plt.subplots()

df['sentiment'].value_counts().plot(ax=ax, kind='bar')
plt.title('Sentiment Frequency in Influencer Dataset')
plt.xticks(rotation=45)
plt.show()
