# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 21:26:22 2023

@author: Poom
"""
import os
import re
import sys
import platform
import shutil
import enchant, difflib
import io
import collections

import pandas as pd
import math
import multiprocessing
import os.path
import numpy as np
from tqdm import tqdm

########################################################################
# Edit job titles
########################################################################

d = enchant.DictWithPWL("en_US", 'G:/My Drive/shared_folder/__paper/data/myPWL.txt')

#...............................................#
# This python module cleans titles
# (1.) substitute word-by-word: includes plural => singular, abbrevations...
# (2.) substitute phrases
# (3.) general plural to singular transformation
#...............................................#

def WordSubstitute(InputString, word_substitutes):
    # This function makes word-by-word substitutions (See: word_substitutes.csv)
    # For each row, everything in the second to last column will be substituted with the first column
    # Example, one row reads "assistant | assistants | asst | asst. | assts"
    # If any word is "assistants", "asst." or "assts" is found, it will be substituted with simply "assistant"    

    InputTokens = [w for w in re.split('\s|-', InputString.lower()) if not w=='']
    
    ListBase = [re.split(',', w)[0] for w in word_substitutes] # list of everything in the first column
    
    RegexList = ['|'.join(['\\b'+y+'\\b' for y in re.split(',', w)[1:] if not y=='']) for w in word_substitutes]
    # regular expressions of everyhing in the second to last column
    
    OutputTokens = InputTokens[:] #copying the output from input
    
    for tokenInd in range(0,len(OutputTokens)):
        token = OutputTokens[tokenInd] # (1) For each word...
        for regexInd in range(0,len(RegexList)):   
            regex = RegexList[regexInd] # (2) ...for each set of regular expressions...
            baseForm = ListBase[regexInd] 
            if re.findall(re.compile(regex),token): # (3) ...if the word contains in the set of regular expressions... 
                OutputTokens[tokenInd] = baseForm # (4) ...the word becomes that baseForm = value of the first column.
    return ' '.join(OutputTokens)

#...............................................#

def PhraseSubstitute(InputString, phrase_substitutes):
    # This function makes phrases substitutions (See: phrase_substitutes.csv)
    # The format is similar to word_substitutes.csv 
    # Example: 'assistant tax mgr' will be substituted with 'assistant tax manager'
    
    ListBase = [re.split(',',w)[0] for w in phrase_substitutes]
    RegexList = ['|'.join(['\\b'+y+'\\b' for y in re.split(',',w)[1:] if not y=='']) for w in phrase_substitutes]
    
    OutputString = InputString.lower()

    # Unlike WordSubstitute(.) function, this one looks at the whole InputString and make substitution.

    for regexInd in range(0,len(RegexList)):
        regex = RegexList[regexInd]
        baseForm = ListBase[regexInd]
        if re.findall(re.compile(regex),InputString):
            OutputString = re.sub(re.compile(regex),baseForm,InputString)
    return OutputString

#...............................................#

def SingularSubstitute(InputString):
    # This function performs general plural to singular transformation
    # Note that several frequently appeared words would have been manually typed in "word_substitutes.csv" 

    InputTokens = [w for w in re.split(' ', InputString.lower()) if not w=='']
    OutputTokens = InputTokens[:] #initialize output to be exactly as input
    
    for tokenInd in range(0,len(OutputTokens)):
        
        token = OutputTokens[tokenInd]
        corrected_token = ''

        if d.check(token): # To be conservative, only look at words that d.check(.) is true
            if re.findall('\w+ies$',token):
                # if the word ends with 'ies', changes 'ies' to 'y'
                corrected_token = re.sub('ies$','y',token) 
            elif re.findall('\w+ches$|\w+ses$|\w+xes|\w+oes$',token):
                # if the word ends with 'ches', 'ses', 'xes', 'oes', drops the 'es'
                corrected_token = re.sub('es$','',token)
            elif re.findall('\w+s$',token):
                # if the word ends with 's' BUT NOT 'ss' (this is to prevent changing words like 'business')
                if not re.findall('\w+ss$',token): 
                    corrected_token = re.sub('s$','',token) # drop the 's'
            
        if len(corrected_token) >= 3 and d.check(corrected_token):
            #finally, make a substitution only if the word is at least 3 characters long...
            # AND the correction actually has meanings! 
            OutputTokens[tokenInd] = corrected_token
        
    return ' '.join(OutputTokens)

#...............................................#

def substitute_titles(InputString,word_substitutes,phrase_substitutes):
    # This is the main function
    
    # (1.) Initial cleaning:
    CleanedString = re.sub('[^A-Za-z- ]','',InputString)
    CleanedString = re.sub('-',' ',CleanedString.lower())
    CleanedString = ' '.join([w for w in re.split(' ', CleanedString) if not w==''])

    # (2.) Three types of substitutions:

    if len(CleanedString) >= 1:
        CleanedString = PhraseSubstitute(CleanedString, phrase_substitutes)
        CleanedString = WordSubstitute(CleanedString, word_substitutes)
        CleanedString = SingularSubstitute(CleanedString)
        CleanedString = PhraseSubstitute(CleanedString, phrase_substitutes)

    # (3.) Get rid of duplicating words:
    # This step is to reduce dimensions of the title.
    # for example, "sale sale engineer sale " would be reduced to simply "sale engineer"
    
    ListTokens = [w for w in re.split(' ',CleanedString) if not w=='']
    FinalTokens = list()

    for token in ListTokens: # for each word...
        if not token in FinalTokens: # ...if that word has NOT appeared before...
            FinalTokens.append(token) # ...append that word to the final result. 
            
    return ' '.join(FinalTokens)  

#...............................................#

# import files for editing titles
word_substitutes = io.open('G:/My Drive/shared_folder/__paper/data/word_substitutes.csv','r',encoding='utf-8',errors='ignore').read()
word_substitutes = ''.join([w for w in word_substitutes if ord(w) < 127])
word_substitutes = [w for w in re.split('\n',word_substitutes) if not w=='']
 
phrase_substitutes = io.open('G:/My Drive/shared_folder/__paper/data/phrase_substitutes.csv','r',encoding='utf-8',errors='ignore').read()
phrase_substitutes = ''.join([w for w in phrase_substitutes if ord(w) < 127])
phrase_substitutes = [w for w in re.split('\n',phrase_substitutes) if not w=='']

########################################################################
# Map job titles
########################################################################

title2SOC_filename = 'G:/My Drive/shared_folder/__paper/data/title2SOC.txt'
names = ['title','original_title','soc']

# title: The edited title, to be matched with newspaper titles.
# original_title: The original titles from ONET website. 
# soc: Occupation code.
 
# import into pandas dataframe
title2SOC = pd.read_csv(title2SOC_filename, sep = '\t', names = names)

data = pd.read_csv('G:/My Drive/shared_folder/__paper/data/tweets_v172_final_final.csv')
tqdm.pandas()

# map all job titles from user description
for i, row in tqdm(data.iterrows(), total=data.shape[0]):
    if pd.isnull(row['user_description']):
        data.loc[i, 'job_title'] = None
        continue
    text = substitute_titles(row['user_description'],word_substitutes,phrase_substitutes)
    tmp = title2SOC[title2SOC.title.apply(lambda x: x in text)]
    if(len(tmp)==0):
        continue
    data.loc[i, 'job_title'] = ', '.join(tmp['original_title'].tolist())
    data.loc[i, 'soc'] = ', '.join(tmp['soc'].tolist())
    data.loc[i, 'num_matched'] = len(tmp)

counts = data['num_matched'].value_counts()
data['num_matched'].isna().sum()

data.to_csv('C:/Users/Poom/Desktop/tweets_v172_final_final.csv', index=False)

# map only the first job mentioned in user description
data['refined_job'] = np.nan
data['refined_soc'] = np.nan
data['tmp_job'] = np.nan
data['tmp_soc'] = np.nan
data['unique_count'] = np.nan
for i, row in tqdm(data.iterrows(), total=data.shape[0]):
    if(row['num_matched']==1 or pd.isnull(row['job_title'])):
       data.loc[i, 'refined_job'] = row['job_title']
       data.loc[i, 'refined_soc'] = row['soc']
       continue
    text = substitute_titles(row['user_description'],word_substitutes,phrase_substitutes)
    for j in range(len(row['user_description'].split()) + 1):
        stext = ' '.join(text.split()[:j])
        tmp = title2SOC[title2SOC.title.apply(lambda x: x in stext)]
        if(len(tmp)!=0):
            data.loc[i, 'refined_job'] = tmp['original_title'].iloc[0]
            data.loc[i, 'refined_soc'] = tmp['soc'].iloc[0]
            data.loc[i, 'unique_count'] = tmp['soc'].astype(str).str[:2].nunique()
            if(len(tmp)>1):
                data.loc[i, 'tmp_job'] = ', '.join(tmp['original_title'].tolist())
                data.loc[i, 'tmp_soc'] = ', '.join(tmp['soc'].tolist())
            break

counts = data[data['unique_count'] > 1]['tmp_job'].value_counts().rename_axis('unique_values').reset_index(name='counts')
data[data['job'] == 'Director']['soc_code']
data['user_description'][4392]

# temp = counts.merge(data[['tmp_job', 'tmp_soc']], left_on='unique_values', right_on='tmp_job', how='left')
# temp.drop_duplicates(subset=['unique_values'], keep='first', inplace=True)
# temp2 = pd.read_csv('C:/Users/Poom/Desktop/multi_job.csv')
# temp2 = temp2[['unique_values', 'job', 'soc']]
# temp3 = temp.merge(temp2, on='unique_values', how='left')

# data = data[data['job_title'].notna()]
# data = data[data['num_matched']!=1]
# counts = data['job_title'].value_counts()
# counts.to_csv('C:/Users/Poom/Desktop/occupation_human_expert.csv')
# counts = pd.read_csv('C:/Users/Poom/Desktop/occupation_human_expert.csv')
# temp = counts.merge(data[['user_name', 'user_description', 'job_title', 'soc']], on='job_title', how='left')
# temp.drop_duplicates(subset=['job_title'], inplace=True)
# temp.to_csv('C:/Users/Poom/Desktop/occupation_human_expert.csv', index=False)

# first job mention still have multiple titles, use human judgement to refine further
data = pd.read_csv('C:/Users/Poom/Desktop/tweets_v172_final_final.csv')
multi_job = pd.read_csv('C:/Users/Poom/Desktop/multi_job.csv')

data['job'] = np.nan
data['soc_code'] = np.nan
for i, row in tqdm(data.iterrows(), total=data.shape[0]):
    if (pd.isnull(row['num_matched'])):
        continue
    elif (row['num_matched']==1 or row['unique_count']==1):
        data.loc[i, 'job'] = data.loc[i, 'refined_job']
        data.loc[i, 'soc_code'] = data.loc[i, 'refined_soc']    
    elif (~pd.isnull(multi_job[multi_job['unique_values'] == row['tmp_job']]['counts'].item())):
        if (multi_job[multi_job['unique_values'] == row['tmp_job']]['counts'].item() > 1):
            data.loc[i, 'job'] = multi_job[multi_job['unique_values'] == row['tmp_job']]['job'].item()
            data.loc[i, 'soc_code'] = multi_job[multi_job['unique_values'] == row['tmp_job']]['soc'].item()
        elif (multi_job[multi_job['unique_values'] == row['tmp_job']]['counts'].item() == 1):
            if (pd.isnull(multi_job[multi_job['unique_values'] == row['tmp_job']]['counts'].item())):
                data.loc[i, 'job'] = row['refined_job']
                data.loc[i, 'soc_code'] = row['refined_soc']
            else:
                data.loc[i, 'job'] = multi_job[multi_job['unique_values'] == row['tmp_job']]['job'].item()
                data.loc[i, 'soc_code'] = multi_job[multi_job['unique_values'] == row['tmp_job']]['soc'].item()
                
data['soc_code'] = data['soc_code'].astype('Int64').astype('str')
data['soc_code'] = np.where(data['job'] == 'Leader', '51101100', data['soc_code'])
data.drop(['job_title', 'soc', 'num_matched', 'refined_job', 'refined_soc', 'tmp_job', 'tmp_soc', 'unique_count'], axis=1, inplace=True)

# data[data['job'] == 'Leader']['soc_code']
# data[data['job'] == 'Leader']['soc_code'].nunique()

# define major groups
soc_group = pd.read_csv('C:/Users/Poom/Desktop/soc_group.csv')
data['soc'] = data['soc_code'].str[:2]
data = data.merge(soc_group, on='soc', how='left')
data.drop(['soc'], axis=1, inplace=True)


























