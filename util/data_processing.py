import os
import pandas as pd
import numpy as np
import string
import nltk
import re
import contractions

def fetch_data():
    data_file_path = './dataset'
    data_file_list = os.listdir(data_file_path)

    swda_df = pd.concat([pd.read_csv(data_file_path+'/'+f) for f in data_file_list ], ignore_index=True)
    return swda_df


def remove_unwanted_params(df, required_params):
    return df[required_params]


def mark_backchannels(df):
    is_backchannel = np.where(df.act_tag == 'b', 1, 0)
    df['is_backchannel'] = is_backchannel
    return df

def preprocess_data(df):
    df['text'] =  df['text'].str.lower()

    punctuation = string.punctuation
    df['text'] = df['text'].apply(lambda x:' '.join(word for word in x.split() if word not in punctuation))
    
    #Remove non-alphabetical character function by using regex
    df['text']=df['text'].apply(lambda x: " ".join([re.sub('[^A-Za-z]+','', x) for x in nltk.word_tokenize(x)]))
    df['text']=df['text'].apply(lambda x: re.sub(' +', ' ', x))
    df['text']=df['text'].str.replace('\d+', '')

    #Performing word contraction using library
    df['no_contract'] = df['text'].apply(lambda x: [contractions.fix(word) for word in x.split()])
    df['text'] = [' '.join(map(str, l)) for l in df['no_contract']]
    df = df.drop(['no_contract', 'act_tag'], axis='columns')
    df = df.dropna()

    return df