#!/usr/bin/env python
# coding: utf-8

# In[10]:



import os
#Kanoume import to pandas gia na diavasoume to csv file
import numpy as np
import pandas as pd
#Kanoume import to Scikit-Learn to opoio tha xrhsimopoihsoume gia split alla kai gia test tou montelou mas
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

#Fortonoume to dataset mas
dataset = pd.read_csv('winequality-red.csv' , sep =',')    

#Kanoume seperate ta variables oso afora auta pou theloume na mantepsoume
dataset_x = dataset.drop('quality', axis = 1)
dataset_y = dataset['quality']

#gia veltiosh twn apotelesmatwn mas tha kanoume kanonikopoihsh me thn xrhsh enws scaler
standard_scaler = StandardScaler()
dataset_x = standard_scaler.fit_transform(dataset_x)

#Kanoume split se train kai test to dataset mas me analogia 75-25
#Pairname kai ena seed wste na pairnoume panta ta idia Train kai Test gia na testaroume thn apodwsh tou montelou argotera
Train_X, Test_X, Train_Y, Test_Y = train_test_split(dataset_x, dataset_y, test_size = 0.25, train_size = 0.75, random_state=10)

#Edw afairoume(kanoume null) apo to training set mas to 33% twn timwn tou ph
rows = Train_X.shape[0]
size = rows/3
for x in range(int(size)):
    Train_X[x,8]=np.nan

#Twra kanoume handle tis elleipes times xrhsimopoiontas logistic regresion
#Tha 3anaxwrisoume to dataset ws e3hs: 
#to Train_X_new tha einai ola ta columns xwris to ph pou den periexoun NaN
#to Train_Y_new tha einai ta pH ta opoia den periexoun NaN
#to Test_X_new tha einai ola ta columns xwris to ph pou gia auta to ph perileambanei NaN
#to Test_Y_new tha einai ta ph ta opoia exoun NaN
#Epeita tha prospathisoume na kanoume predict to Test_Y_new me logistic regresion
Train_X_new = Train_X[int(size):,:]
Train_X_new = np.delete(Train_X_new, 8, 1)

Train_Y_new = Train_X[int(size):,8]

Test_X_new = Train_X[0:int(size),:]
Test_X_new = np.delete(Test_X_new, 8, 1)

Test_Y_new = Train_X[0:int(size),8]

#Kanoume train afou dhmiourgisoume to logistic regresion mas
logistic_regression = LogisticRegression()
logistic_regression.fit(Train_X_new.astype(int), Train_Y_new.astype(int))

#Kanoume predict tis times
pred_logistic_regresion = logistic_regression.predict(Test_X_new)

#Anadomoume to arxiko Train_X set
Train_X[0:int(size),8] = pred_logistic_regresion

#Ftiaxnoume kai kanoume train to support vector machine mas
vector_machine = svm.SVC()
vector_machine.fit(Train_X, Train_Y)
pred_vector_machine = vector_machine.predict(Test_X)

#Kanoume print ta metrics mas
print(classification_report(Test_Y,pred_vector_machine))

os.system("PAUSE")





