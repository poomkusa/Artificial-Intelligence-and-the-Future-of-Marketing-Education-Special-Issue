# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

data = pd.read_csv('G:/My Drive/shared_folder/paper/tweets_v84.csv')
temp = data.head(100)
data.dtypes

# check and remove empty tweets
data.info()
(data.isnull().sum() / len(data)) * 100
temp1 = data[~data['text'].isnull()]
temp2 = data[data['text'].isnull()]
data = data[~data['text'].isnull()]

# describe each column
for (columnName, columnData) in data.iteritems():
    series_value = pd.Series(columnData.values)
    print('====================================================================')
    print('Colunm Name : ', columnName)
    print('Type: ', type(columnData.values[0]))
    print(series_value.describe())
    print('number of nan', series_value.isnull().sum())

# remove bot (top 1% tweets)
    
    
    
    
    
    
    
    
    
    
    
# remove duplicates & near duplicates (We find near-duplicates by hashing the texts of tweets after lowercasing and stripping
# punctuation. Hashing is performed using MinHash (Broder, 1997), with 16 permutations.)
# https://github.com/google-research/deduplicate-text-datasets
# https://github.com/cardiffnlp/timelms/blob/main/scripts/preprocess.py
    
    
    
    
    
    
    
    
    
    
    
    
    
# check for duplicate tweets
temp = data['text'].value_counts()
temp1 = data[data['text'].str.contains('he Next ChatGPT Revolution: Intelligent Document Processing')]
temp2 = temp1['text'].value_counts()
#data["user+text"] = data["user_name"] + data["text"]
#temp = data['user+text'].value_counts()
result_df = data.drop_duplicates(subset=['user_name', 'text', 'date'], keep='first')

# filter for tweets with education word
#ChatGPT + school; ChatGPT + college; ChatGPT + university;
#ChatGPT + education; ChatGPT + student; ChatGPT + teacher;
#ChatGPT + learning; ChatGPT + curriculum; ChatGPT + class;
#ChatGPT + exam; ChatGPT + homework; ChatGPT + teaching;
#ChatGPT + academia; ChatGPT + academic.













# occupation








data.to_feather("Desktop/tweets_v84.feather")
# sentiment analysis
# replacing user handles and URL links with generic placeholders (@user and http)
# user mentions are replaced with a generic placeholder (@user), except for verified users.

# topic modeling
# removing URLs, user handles, and emojis.



















