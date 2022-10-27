import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Libraries and packages for text (pre-)processing 
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import scikitplot as skplt

import pickle
import streamlit as st


BANNER_URL = "banner.jpg"
# Set page title and favicon.
st.set_page_config(
    page_title="Fake News Detection App", page_icon=BANNER_URL,
)
## Main Panel

st.image(BANNER_URL, width=1080)

with open('./models/vect.p', 'rb') as f:
    vect = pickle.load(f)

with open('./models/model.p', 'rb') as f:
    model = pickle.load(f)
    
#initialise lemmatizer
lemmatizer = WordNetLemmatizer()
stpwrds = list(stopwords.words('english'))
    
#function for preprocess
def preprocess(news):
    corpus =[]
    news = re.sub(r'https?://\S+|www\.\S+', '', news) #remove URL
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});") #define html tags
    news = re.sub(html, "", news) #remove html tags
    news = re.sub(r'[^\x00-\x7f]','', news) #remove non-ascii 
    news = re.sub(r'[^a-zA-Z\s]', '', news) #remove non-letters
    news = news.lower() #convert all characters to lower caps
    news = nltk.word_tokenize(news) #covert string of text into token within list
    for y in news :
        if y not in stpwrds : #remove stopwords
            corpus.append(lemmatizer.lemmatize(y))
    news = ' '.join(corpus) #rejoin words into strings
    
    return news

import streamlit as st
st.title('Fake News Detection System')

def fakenewsdetection():
    user = st.text_area("Enter Any News: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        preprocess(user)
        data = vect.transform([sample]).toarray()
        result = model.predict(data)
        if result == 1:
            st.title('Fake News!')
        elif result == 0:
            st.title('Real News!')
        #st.title(a)
fakenewsdetection()