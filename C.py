#!/usr/bin/env python
# coding: utf-8

# In[2]:

#Kanoume import tis libraries pou tha xrhsimopoihsoume
import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

#Fortonoume to dataset mas
dataset = pd.read_csv('onion-or-not.csv' , sep =',')

#Kanoume seperate ta variables oso afora auta pou theloume na mantepsoume
dataset_x = dataset.iloc[:,0]
dataset_y = dataset.iloc[:,1].values

#Pinakas gia na baloume ta tokens
tokenized_text = []

#Kanoume tokenize to text kai to apothikeuoume ston pinaka
for i in range(dataset_x.size):
	tokenized_text.append(word_tokenize(dataset_x[i]))

#Dhmiourgoume to list pou tha baloume ta stemmed words, tis listes twn stemmed words kai filtered_stemming kai to stemmer
stop_words=set(stopwords.words("english"))
filtered_stemming = []
stemmer = PorterStemmer()

#Kanoume stemming me thn xrhsh tou Algorithmou tou Porter Stemmer kai afairoume stop words
for i in tokenized_text:
	for k in i:
		if k not in stop_words:
			k = (stemmer.stem(k))
			filtered_stemming.append(k)
	
#Afairoume ta symvola
symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
for i in symbols:
    filtered_stemming = np.char.replace(filtered_stemming, i, ' ')
	
#Twra bazoume th lista me tis telikes le3eis twn titlwn se ena tf-idf vector
tf_idf_vectorizer = TfidfVectorizer()
tf_idf_vectors = tf_idf_vectorizer.fit_transform(filtered_stemming)
feature_names = tf_idf_vectorizer.get_feature_names()
dense_vectors = tf_idf_vectors.todense()
title_list = dense_vectors.tolist()

title_dataframe = pd.DataFrame(title_list, columns=feature_names)

#twra afou exoume to dataset mas tha xrhsimopoihsoume ena dyktio me perceptrons gia na paroume thn ektimhsh afou kanoume prwta split to dataset
dataset_x_new = title_dataframe.iloc[:,:].values
dataset_y_new = np.reshape(dataset_y,[np.size(dataset_y),1])

Train_X, Test_X, Train_X, Test_Y = train_test_split(dataset_x_new,dataset_y_new,test_size=0.25,train_size=0.75,random_state= 10)

perceptrons = MLPClassifier(hidden_layer_sizes=(4,2),max_iter=50)
perceptrons.fit(Train_X,Train_Y)

perceptrons_predict = perceptrons.predict(Test_X)

#kanoume print ta metrics mas
print(classification_report(Test_Y,perceptrons_predict))

os.system("PAUSE")

