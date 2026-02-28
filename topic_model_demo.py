# joblib is for saving and loading objects
from joblib import load, dump
import pandas as pd
import numpy as np

# gensim is for LDA
import lda
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

import pandas as pd
import pprint as pprint

import requests

from sklearn.feature_extraction.text import CountVectorizer

stopwords = requests.get('https://raw.githubusercontent.com/stopwords-iso/stopwords-en/refs/heads/master/stopwords-en.txt')

stopwords = stopwords.text.split('\n')

vulgarity = ['fuck', 'shit', 'piss', 'bitch']
[stopwords.append(x) for x in vulgarity]

stopword_pattern = r'\b(?:{})\b'.format('|'.join(stopwords))

songs = pd.read_feather(
  '/Users/sberry5/Documents/teaching/UDA/code/lyrics_scrape_python/complete_lyrics_2025.feather'
)

songs = songs.dropna()

songs['lyrics'] = songs['lyrics'].astype('str')

songs['lyrics'] = songs['lyrics'].str.replace('([a-z])([A-Z])', '\\1 \\2', regex=True)

songs['lyrics'] = songs['lyrics'].str.replace('\\[.*?\\]', ' ', regex=True)

songs['lyrics'] = songs['lyrics'].str.replace('[Ee]mbeded|Embed', ' ', regex=True)

songs['lyrics'] = songs['lyrics'].str.replace('Contributed', ' ', regex=True)

songs = songs[songs['lyrics'].str.contains("This song is an instrumental") == False]

songs['genre'] = songs['week'].str.extract('((?<=charts/).*(?=/[0-9]{4}))')

songs['date'] = songs['week'].str.extract('([0-9]{4}-[0-9]{2}-[0-9]{2})')

country_only = songs[songs['genre'] == 'country-songs']

country_only.info()

country_only.describe()

country_only.drop_duplicates(subset=['joiner'], inplace=True)

country_only.info()

country_only['lyrics'] = country_only['lyrics'].str.lower()

country_only['lyrics_clean'] = country_only['lyrics'].str.replace(stopword_pattern, '', regex=True)

country_only['lyrics_clean'] = country_only['lyrics_clean'].str.replace(r"[\(\).,?!;:'-]", '', regex=True)

country_only['lyrics_clean'] = country_only['lyrics_clean'].str.replace(' +', ' ', regex=True)

country_only['word_count'] = country_only['lyrics_clean'].apply(lambda x: len(x.split()))
country_only['word_count'].describe()
country_only = country_only[country_only['word_count'] < 175]

country_only.info()

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(country_only['lyrics_clean'])

vocab = vectorizer.get_feature_names_out()

model = lda.LDA(n_topics=5, n_iter=500, random_state=1)

model.fit(X)

topic_word = model.topic_word_

n_top_words = 10

top_words = []

for i, topic_dist in enumerate(topic_word):
  topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
  top_words.append(' '.join(topic_words))
  print('Topic {}: {}'.format(i, ' '.join(topic_words)))

model.doc_topic_

country_only['top_topic'] = np.argmax(model.doc_topic_, axis=1)

country_only['top_prob'] = [max(model.doc_topic_[x]) for x in range(len(model.doc_topic_))]

country_only['top_topic'].value_counts()

country_only['date'] = pd.to_datetime(country_only['date'])

country_only['year'] = country_only['date'].dt.year

country_only[['topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']] = model.doc_topic_

aggregated = country_only.groupby(['top_topic', 'year'])['top_prob'].mean().to_frame(name = 'prob').reset_index()

sns.lineplot(x='year', y='prob', data=aggregated, hue='top_topic')
plt.show()

mf = smf.mixedlm("top_prob ~ year", country_only, groups=country_only["top_topic"])

mdf = mf.fit()

print(mdf.summary())
country_only.columns
mf_t1 = smf.ols("topic_1 ~ year", country_only)

mdf_t1 = mf_t1.fit()

print(mdf_t1.summary())

mdf_t1_preds = mdf_t1.get_prediction()

plt.plot(
  country_only['year'], 
  mdf_t1_preds.predicted_mean, 
  color='red')
plt.title('Topic 1')
plt.show()

mf_t2 = smf.ols("topic_2 ~ year", country_only)

mdf_t2 = mf_t2.fit()

mdf_t2_preds = mdf_t2.get_prediction()

mf_t3 = smf.ols("topic_3 ~ year", country_only)

mdf_t3 = mf_t3.fit()

mdf_t3_preds = mdf_t3.get_prediction()

mf_t4 = smf.ols("topic_4 ~ year", country_only)

mdf_t4 = mf_t4.fit()

mdf_t4_preds = mdf_t4.get_prediction()

mf_t5 = smf.ols("topic_5 ~ year", country_only)

mdf_t5 = mf_t5.fit()

mdf_t5_preds = mdf_t5.get_prediction()

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(country_only['year'], mdf_t1_preds.predicted_mean, label="T1")
ax.plot(country_only['year'], mdf_t2_preds.predicted_mean, label="T2")
ax.plot(country_only['year'], mdf_t3_preds.predicted_mean, label="T3")
ax.plot(country_only['year'], mdf_t4_preds.predicted_mean, label="T4")
ax.plot(country_only['year'], mdf_t5_preds.predicted_mean, label="T5")
legend = ax.legend(loc="best")

plt.show()