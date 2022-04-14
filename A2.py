#!/usr/bin/env python
# coding: utf-8

# In[3]:



import os
#Kanoume import to pandas gia na diavasoume to csv file
import pandas as pd
#Kanoume import to Scikit-Learn to opoio tha xrhsimopoihsoume gia split alla kai gia test tou montelou mas
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

#Fortonoume to dataset mas
dataset = pd.read_csv('winequality-red.csv' , sep =',')

#Kanoume seperate ta variables oso afora auta pou theloume na mantepsoume
dataset_x = dataset.drop('quality', axis = 1)
dataset_y = dataset['quality']

#gia veltiosh twn apotelesmatwn mas tha kanoume kanonikopoihsh me thn xrhsh enws scaler
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
dataset_x = min_max_scaler.fit_transform(dataset_x)

#Kanoume split se train kai test to dataset mas me analogia 75-25
#Pairname kai ena seed wste na pairnoume panta ta idia Train kai Test gia na testaroume thn apodwsh tou montelou argotera
Train_X, Test_X, Train_Y, Test_Y = train_test_split(dataset_x, dataset_y, test_size = 0.25, train_size = 0.75, random_state=10)

#Ftiaxnoume kai kanoume train to support vector machine mas
vector_machine = svm.SVC()
vector_machine.fit(Train_X, Train_Y)
pred_vector_machine = vector_machine.predict(Test_X)

#Kanoume print ta metrics mas
print(classification_report(Test_Y,pred_vector_machine))

os.system("PAUSE")


# In[ ]:




