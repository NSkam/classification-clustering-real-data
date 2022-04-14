#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
#Kanoume import to pandas gia na diavasoume to csv file
import numpy as np
import pandas as pd
#Kanoume import to Scikit-Learn to opoio tha xrhsimopoihsoume gia split alla kai gia test tou montelou mas
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.cluster import KMeans
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

#Tha xwrisoume to Train_X se train_x_1 pou tha periexei ola ta columns ektos apo to ph
#Tha tre3oume to kmeans xrhsimopoiwntas to train_x_1 kai meta afou paroume ta cluster gia kathe shmeio tha broume to mean twn ph kathe cluster(akoma kai an den kaname clustering xrhsimopowntas thn column ph)
#kai tha symplhrwsoume ta nulls sto Train_X
train_x_1 = Train_X[int(size):,:]
train_x_1 = np.delete(train_x_1, 8, 1) 
kmeans = KMeans(n_clusters=8)
kmeans.fit(train_x_1)
k_means_predict = kmeans.predict(train_x_1)

#listes gia kratame kathe shmeio pou anhkei se kathe cluster
cluster_0 = []
cluster_1 = []
cluster_2 = []
cluster_3 = []
cluster_4 = []
cluster_5 = []
cluster_6 = []
cluster_7 = []

#Twra exoume to to k_means_predict pou mas leei se poio cluster anhkei kathe shmeio ara mporoume na broume to means tou ph kathe cluster kai meta na kanoume fill
#in ta shmeia sto idio cluster me null
for i in range(int(k_means_predict.shape[0])):
    if(k_means_predict[i] == 0):
        cluster_0.append(i)
    if(k_means_predict[i] == 1):
        cluster_1.append(i)
    if(k_means_predict[i] == 2):
        cluster_2.append(i)
    if(k_means_predict[i] == 3):
        cluster_3.append(i)
    if(k_means_predict[i] == 4):
        cluster_4.append(i)
    if(k_means_predict[i] == 5):
        cluster_5.append(i)
    if(k_means_predict[i] == 6):
        cluster_6.append(i)
    if(k_means_predict[i] == 7):
        cluster_7.append(i)

#Briskoume ta means tou ph gia ta kathe cluster
cluster_0_mean = np.nanmean(Train_X[cluster_0,8], axis=0)
cluster_1_mean = np.nanmean(Train_X[cluster_1,8], axis=0)
cluster_2_mean = np.nanmean(Train_X[cluster_2,8], axis=0)
cluster_3_mean = np.nanmean(Train_X[cluster_3,8], axis=0)
cluster_4_mean = np.nanmean(Train_X[cluster_4,8], axis=0)
cluster_5_mean = np.nanmean(Train_X[cluster_5,8], axis=0)
cluster_6_mean = np.nanmean(Train_X[cluster_6,8], axis=0)
cluster_7_mean = np.nanmean(Train_X[cluster_7,8], axis=0)

#Twra brsikoume ta shmeia pou exoun null sto ph kai ta kanoume isa me to mean tou cluster
#cluster 0
for k in cluster_0:
    if(np.isnan(Train_X[k,8])):
        Train_X[k,8]=cluster_0_mean

#cluster 1
for k in cluster_1:
    if(np.isnan(Train_X[k,8])):
        Train_X[k,8]=cluster_1_mean

#cluster 2
for k in cluster_2:
    if(np.isnan(Train_X[k,8])):
        Train_X[k,8]=cluster_2_mean

#cluster 3
for k in cluster_3:
    if(np.isnan(Train_X[k,8])):
        Train_X[k,8]=cluster_3_mean
        
#cluster 4
for k in cluster_4:
    if(np.isnan(Train_X[k,8])):
        Train_X[k,8]=cluster_4_mean

#cluster 5
for k in cluster_5:
    if(np.isnan(Train_X[k,8])):
        Train_X[k,8]=cluster_5_mean
        
#cluster 6
for k in cluster_6:
    if(np.isnan(Train_X[k,8])):
        Train_X[k,8]=cluster_6_mean

#cluster 7
for k in cluster_7:
    if(np.isnan(Train_X[k,8])):
        Train_X[k,8]=cluster_7_mean

#Ftiaxnoume kai kanoume train to support vector machine mas
vector_machine = svm.SVC()
vector_machine.fit(Train_X, Train_Y)
pred_vector_machine = vector_machine.predict(Test_X)

#Kanoume print ta metrics mas
print(classification_report(Test_Y,pred_vector_machine))

os.system("PAUSE")


# In[ ]:




