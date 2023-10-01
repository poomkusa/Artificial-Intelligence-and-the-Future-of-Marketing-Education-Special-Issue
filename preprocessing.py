# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import string
from datasketch import MinHash, LeanMinHash
import xxhash
from tqdm import tqdm

#data = pd.read_csv('G:/My Drive/shared_folder/paper/tweets_v84.csv')
data = pd.read_csv('/home/poom/Desktop/tweets_v172.csv')
#temp = data.head(100)
#data.dtypes

# check and remove empty tweets
#data.info()
#(data.isnull().sum() / len(data)) * 100
#temp1 = data[~data['text'].isnull()]
#temp2 = data[data['text'].isnull()]
data = data[~data['text'].isnull()]

# describe each column
#for (columnName, columnData) in data.iteritems():
#    series_value = pd.Series(columnData.values)
#    print('====================================================================')
#    print('Colunm Name : ', columnName)
#    print('Type: ', type(columnData.values[0]))
#    print(series_value.describe())
#    print('number of nan', series_value.isnull().sum())

# remove bot (top 1% tweets)
top_users = data['user_name'].value_counts().rename_axis('user_name').reset_index(name='counts')
n_blacklisted_users = int(len(top_users)*0.01)
#temp = data[data["user_name"] == top_users['user_name'][23]]
#temp['user_description'].unique()
blacklisted_users = top_users.head(n_blacklisted_users)
data['bot'] = np.where(data['user_name'].isin(blacklisted_users['user_name']), True, False)
#data['bot'].sum()
#blacklisted_users['counts'].sum()
data = data[~data.bot]
data = data.drop(['bot'], axis=1)

# filter for tweets with education word
#temp = data[-data['text'].str.lower().str.contains('chatgpt')]
# data = data[data['text'].str.contains('chatgpt', case=False)]
keep_list = ['school','college','university','education','student','teacher','learning'\
             ,'curriculum','class','exam','homework','teaching','academia','academic']
pat = '|'.join(r"\b{}\b".format(x) for x in keep_list)
data = data[data['text'].str.contains(pat, case=False)] #\b to avoid match cat in words bobcat is nice, matching cat is nice
# data = data[data["text"].str.contains("|".join(keep_list), case=False)]
data = data[data["text"].apply(lambda x: any(k in x for k in keep_list))]
#data = data[data['text'].str.contains('school|college|university|education|student|teacher|learning|curriculum|class|exam|homework|teaching|academia|academic', case=False)]

# remove duplicates & near duplicates (We find near-duplicates by hashing the texts of tweets after lowercasing and stripping
# punctuation. Hashing is performed using MinHash (Broder, 1997), with 16 permutations.)
# check for duplicate tweets
#temp = data['text'].value_counts()
#temp1 = data[data['text'].str.contains('he Next ChatGPT Revolution: Intelligent Document Processing')]
#temp2 = temp1['text'].value_counts()
#data["user+text"] = data["user_name"] + data["text"]
#temp = data['user+text'].value_counts()
data = data.drop_duplicates(subset=['text'], keep='first')
# https://github.com/google-research/deduplicate-text-datasets
# https://github.com/cardiffnlp/timelms/blob/main/scripts/preprocess.py
def hash_tweet(tweet, num_perm=16):
    def normalize_text(text):
        text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        text = text.lower()
        return text
    def minhash(seq):
        # https://skeptric.com/minhash/
        m = MinHash(num_perm=num_perm, hashfunc=xxhash.xxh64_intdigest)
        for s in seq:
            m.update(s.encode('utf8'))
        return LeanMinHash(m)
    tokens = normalize_text(tweet).split()  # whitespace tokenization
    return minhash(tokens)
written_hashes = set()
data['duplicate'] = False
# for index in range(len(data)):
#   tweet_hash = hash_tweet(data['text'].iloc[index])
#   if tweet_hash in written_hashes:
#     data['duplicate'].iloc[index] = True
#   else: data['duplicate'].iloc[index] = False
#   if not data['duplicate'].iloc[index]:
#     written_hashes.add(tweet_hash)
for index in tqdm(range(len(data))):
    tweet_hash = hash_tweet(data['text'].iloc[index])
    if tweet_hash in written_hashes:
        data['duplicate'].iloc[index] = True
    else: data['duplicate'].iloc[index] = False
    if not data['duplicate'].iloc[index]:
        written_hashes.add(tweet_hash)
    
# occupation
#preprocessed texts in this column by removing URLs, the hashtag symbol, emojis, stop words,
#digits, and punctuations, as well as expanding abbreviations, expanding contractions, applying low-
#ercase, and applying lemmatization. We eventually obtained unigrams and bigrams of the preprocessed
#texts which are used for matching the occupationl ist.







# data.reset_index(drop=True).to_feather("/home/poom/Desktop/tweets_v172_cleaned.csv")
data.to_csv('/home/poom/Desktop/tweets_v172_cleaned_v3.csv', index=False)
# sentiment analysis
# replacing user handles and URL links with generic placeholders (@user and http)
# user mentions are replaced with a generic placeholder (@user), except for verified users.
def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')    
    new_text = []
    for t in text.split():
        t = '@user' if t.startswith('@') and len(t) > 1 and t.replace('@','') not in verified_users else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return ' '.join(new_text)

# topic modeling
# removing URLs, user handles, and emojis (maybe not).
def remove_mentions_and_links(text):
    new_text = []
    for t in text.split(" "):
        t = "" if (t.startswith('@') or t.startswith('#') ) and len(t) > 1 else t
        new_text.append(t)

    new_text = re.sub(r'http\S+', '', " ".join(new_text))
    return new_text
from tqdm import tqdm
# from tqdm.auto import tqdm  # for notebooks
# from tqdm.notebook import tqdm
# Create new `pandas` methods which use `tqdm` progress (can use tqdm_gui, optional kwargs, etc.)
tqdm.pandas()
# Now you can use `progress_apply` instead of `apply`
data["prep"] = data.text_rt.progress_apply(remove_mentions_and_links)
data = data.reset_index(drop=True)










#https://github.com/cardiffnlp/timelms/blob/main/scripts/preprocess.py
def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')    
    new_text = []
    for t in text.split():
        t = '@user' if t.startswith('@') and len(t) > 1 and t.replace('@','') not in verified_users else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return ' '.join(new_text)

#https://github.com/twitter-tuebingen/ChatGPT_project/blob/main/code/analysis/Topic%20and%20Sentiment.ipynb
import re
def remove_mentions_and_links(text):
    new_text = []
    for t in text.split(" "):
        t = "" if (t.startswith('@') or t.startswith('#') ) and len(t) > 1 else t
        new_text.append(t)

    new_text = re.sub(r'http\S+', '', " ".join(new_text))
    return new_text
data["prep"] = data.text_rt.apply(remove_mentions_and_links)
data = data.reset_index(drop=True)
def basic_preprocessing(texts):
    texts = texts.str.replace("https:","")
    texts = texts.str.replace("t.co","")
    # remove tel and email
    texts = texts.str.replace("<email>","")
    texts = texts.str.replace("<tel>","")
    texts = texts.str.replace("<link>","")
    texts = [re.sub(r'anon\d*',"",t) for t in texts]
    # Remove new line characters
    texts = [re.sub('\s+', ' ', t) for t in texts]
    # Remove single quotes
    texts = [re.sub("\'", "", sent) for sent in texts]
    # remove some punctuation and numbers, emoji
    texts = [words_only(t.lower()).strip() for t in texts]
    return texts
data.prep = basic_preprocessing(data.prep)
data.prep = data.prep.str.replace("chatgpt","")





