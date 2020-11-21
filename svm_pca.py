#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 20:20:31 2020

@author: sharontan
"""

import pandas as pd
import numpy as np
import struct
import wave
import sys
import pydub
from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub.silence import split_on_silence
import sklearn
from sklearn import cluster, preprocessing
from sklearn.preprocessing import normalize, scale
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
from sklearn import svm, decomposition, cluster
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
import keras
from keras import backend as bkend
import tensorflow as tf
from tensorflow.python.client import device_lib

import scipy
from scipy.io import wavfile
import pickle
import random
from scipy.spatial.distance import sqeuclidean
import matplotlib.dates as mdates
import datetime
from datetime import timedelta
from plotnine import *
import plotnine
import matplotlib.pyplot as plt
import os
import importlib


# init data
wave_dir_1='/Users/sharontan/Documents/autoListener/machine/data/ToyConveyor/train/'
wave_dir_2='/Users/sharontan/Documents/autoListener/machine/data/ToyConveyor/test/'
chunk_dir='/Users/sharontan/Documents/autoListener/machine/chunk/svm/chunk/20201115/'
csv_dir='/Users/sharontan/Documents/autoListener/machine/svm/csv/20201115/'
log_dir='/Users/sharontan/Documents/autoListener/machine/svm/log/20201115/'
graph_dir='/Users/sharontan/Documents/autoListener/machine/svm/graph/line/20201115/'
graph_dir_1='/Users/sharontan/Documents/autoListener/machine/svm/graph/loss/20201115/'
graph_dir_2='/Users/sharontan/Documents/autoListener/machine/svm/graph/error/20201115/'

currentTime=datetime.datetime.now()

os.environ["KERAS_BACKEND"] = "tensorflow"
importlib.reload(bkend)
print(device_lib.list_local_devices())

file_format='.wav'

sample_count=8
m=50
PopulationSize=10800
PredictSize=3200
GapSize=0

n0=random.randint(0,3000)
#n0=0
n1=n0+PopulationSize
n2=n0+PopulationSize+GapSize
n3=n0+PopulationSize+GapSize+PredictSize

evaluation_rate=0.05

currentTime=datetime.datetime.now()

timeSequence=str(object=currentTime)[20:26]


class data_preprocessing():
    
    def _init_(self):
        self.df=df
        
    def file_alias():
        
        file_string_a='normal_id_01_000000'
        file_string_b='anomaly_id_01_000000'
        

        file_name_a=[]
        file_name_b=[]
        for i in range(0,sample_count):
            file_number=random.randint(1,99)
            if file_number<10:
                file_number='0'+str(object=file_number)
            else:
                file_number=str(object=file_number)
            file_name_a.append(file_string_a+file_number)
            file_name_b.append(file_string_b+file_number)
            i+=1
        return file_name_a, file_name_b
    

        
    def populationInit():


        #file_1=AudioSegment.from_wav(wave_dir_1+file_name_1+file_format)
        global m
        file_name_a, file_name_b=data_preprocessing.file_alias()
        i=0
        df_a=[]

        df_b=[]

        for i in range(0,int(sample_count)):
            print(i,file_name_a[i])
            file_alias=wave_dir_1+file_name_a[i]+file_format
            Fs, audioData=wavfile.read(file_alias)
            n=audioData.size
            t=round(Fs/10)
            m=round(n/t)
            #print(m)
            
            wavFile=wave.open(file_alias)
            audioString=wavFile.readframes(wavFile.getnframes())
            audioText=struct.unpack('%ih' % (wavFile.getnframes()*wavFile.getnchannels()),audioString)
            audioText=[float(val)/pow(2,15) for val in audioText]
            #print(len(audioText)/m,round(len(audioText)/m))
            audio_rows=round(len(audioText)/m)
            audio_cols=m
            textArray=np.array_split(audioText,round(len(audioText)/m))
            df=pd.DataFrame(data=textArray)
            df.to_csv(csv_dir+file_name_a[i]+'.csv')
            problem=[]
            for j in range (round(len(audioText)/m)):
                problem.append(0)
                j+=1
            df['IssueOrNot']=problem
            df_a.append(df)
            


            file_alias_b=wave_dir_2+file_name_b[i]+file_format
            Fs_b, audioData_b=wavfile.read(file_alias_b)
            n_b=audioData_b.size
            t_b=round(Fs_b/10)
            m_b=round(n_b/t_b)
            #print(m)
            
            wavFile_b=wave.open(file_alias_b)
            audioString_b=wavFile_b.readframes(wavFile_b.getnframes())
            audioText_b=struct.unpack('%ih' % (wavFile_b.getnframes()*wavFile_b.getnchannels()),audioString_b)
            audioText_b=[float(val_b)/pow(2,15) for val_b in audioText_b]
            #print(len(audioText_b)/m_b,round(len(audioText_b)/m_b))
            textArray_b=np.array_split(audioText_b,round(len(audioText_b)/m_b))
            df_=pd.DataFrame(data=textArray_b)
            df_.to_csv(csv_dir+file_name_b[i]+'.csv')
            problem_b=[]
            for i in range (round(len(audioText_b)/m_b)):
                problem_b.append(1)
                i+=1
            df_['IssueOrNot']=problem_b
            df_b.append(df_)
            #print(i)
            
            i+=1


        df_a=pd.concat(df_a,axis=0)
        df_b=pd.concat(df_b,axis=0)
        df=df_a.append(df_b)
        print(len(df_a))
        print(len(df_b))
        print(len(df))

        feature=[]
        df_consolidated=pd.DataFrame()
        for i in range(0,m):
            feature_=df[i]
            feature_name='Feature_'+str(object=i)
            feature_=scale(feature_)
            feature_=normalize(np.array(np.reshape(feature_,(-1,1))))
            feature_=pd.Series(np.reshape(feature_,(-1)))
            issue_or_not=df['IssueOrNot']
            issue_or_not=pd.Series(np.reshape(np.array(issue_or_not),(-1)))
            df_consolidated.loc[:,feature_name]=feature_

            i+=1
        df_consolidated['IssueOrNot']=issue_or_not
        #print(df_consolidated)
        
        df_=np.array(df_consolidated.values)


        print(n0,n2,n3)
        

        x_train=df_[n0:n1,:][:,-51:-1]
        y_train=df_[n0:n1,:][:,-1:]
        x_test=df_[n2:n3,:][:,-51:-1]
        y_test=df_[n2:n3,:][:,-1:]
        x=df_[n0:n3,:][:,-51:-1]


        #print('x_train is', x_train)
        #print('y_train is', y_train)
        #print('x_test is', x_test)
        #print('y_test is', y_test)
        #print(y_train.shape[0],y_train.shape[1])
        #print(len(x_train),x_train.shape[0],x_train.shape[1])
        print (len(df_consolidated.columns))
        features=len(df_consolidated.columns)-1
        #print(features)
        y_train=y_train.astype('int')
        x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1]))
        y_train=np.reshape(y_train,(y_train.shape[0]))
        x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1]))
        x=np.reshape(x,(x.shape[0],x.shape[1]))
        #print(x_test)
        #y_train=np.reshape(y_train,(y_train.shape[0],1))
        #print(y_train)
        return x_train,y_train,x_test,y_test,features,df_consolidated,x




class SVMAnalysis():
    def _init_(self):
        self.df_consolidated=df_consolidated
        


    def myPCA():
        x_train,y_train,x_test,y_test,features,df_consolidated,x=data_preprocessing.populationInit()
        clf_pca=PCA(n_components=7)
        clf_pca.fit(x)
        x_pca=clf_pca.transform(x)
        x_train_pca=clf_pca.transform(x_train)
        x_test_pca=clf_pca.transform(x_test)


        
        return x_test_pca,y_test,x_train_pca,y_train,features,df_consolidated,x_pca
    
    def myLDA():
        x_test_pca,y_test,x_train_pca,y_train,features,df_consolidated,x_pca=SVMAnalysis.myPCA()
        clf_lda=LDA(n_components=2)
        clf_lda.fit(x_train_pca,y_train)
        y=clf_lda.predict(x_pca)
        print(y[0])
        y=np.reshape(y,(y.shape[0],1))
        
        return x_test_pca,y,y_train,y_test,features,y


    
    def myKmeans():
        global n_class
        x_test_pca,y_test,x_train_pca,y_train,features,df_consolidated,x_pca=SVMAnalysis.myPCA()

        
        n_class=2
        k_means = cluster.KMeans(init='k-means++',n_clusters=n_class,max_iter=500000,n_init=50000, random_state=50)
        k_means.fit(x_pca)
        y=k_means.labels_
        print(y)
        print(y[n0:n1])
        print(y[n2-n0:n3-n0])
        
        return x_test_pca,y,y_train,y_test,features,y

    def myDBSCAN():
        global n_class
        x_test_pca,y_test,x_train_pca,y_train,features,df_consolidated,x_pca=SVMAnalysis.myPCA()
        #x_train,y_train,x_test,y_test,features,df_consolidated,x=data_preprocessing.populationInit()

        
        
        dbscan = DBSCAN(eps=2.7, min_samples=m)
        print(m)
        y=dbscan.fit_predict(x_pca)
        for i in range (len(y)):
            if y[i:i+1]==-1:
                y[i:i+1]=0
            else:
                y[i:i+1]=1
                
            i+=1
        print(y[n2-n0:n3-n0])
        print(y[n0:n1-n0])

        #y_predict= dbscan.labels_
        print(y)
        
        return x_test_pca,y,y_train,y_test,features,y
        #return x_test,y,y_train,y_test,features,y

    def myAgglomerativeClustering():

        x_test_pca,y_test,x_train_pca,y_train,features,df_consolidated,x_pca=SVMAnalysis.myPCA()
        #x_train,y_train,x_test,y_test,features,df_consolidated,x=data_preprocessing.populationInit()

        
        n_class=2
        clustering = AgglomerativeClustering()
        clustering.fit(x_pca)
        y=clustering.labels_
        print(y)
        print(y[n0:n1])
        print(y[n2-n0:n3-n0])
        
        return x_test_pca,y,y_train,y_test,features,y
        #return x_test,y,y_train,y_test,features,y
    
        
    def mySVM():
        x_test_pca,y_test,x_train_pca,y_train,features,df_consolidated,x_pca=SVMAnalysis.myPCA()
        #clf=svm.SVC(kernel='rbf', gamma=0.25, C=1.0)
        clf=svm.SVC(gamma='scale')
        print(x_train_pca)
        print(y_train)
        clf.fit(x_train_pca,y_train)
        y=clf.predict(x_pca)
        print(y.shape[0])
        y=np.reshape(y,(y.shape[0],1))
        
        return x_test_pca,y_test,x_train_pca,y_train,y
    
    def myVisualize():
        #x_test_pca,y_test,x_train_pca,y_train,y=SVMAnalysis.mySVM()
        x_test_pca,y,y_train,y_test,features,y=SVMAnalysis.myKmeans()
        #x_test_pca,y,y_train,y_test,features,y=SVMAnalysis.myLDA()
        #x_test_pca,y,y_train,y_test,features,y=SVMAnalysis.myAgglomerativeClustering()
        #x_test_pca,y,y_train,y_test,features,y=SVMAnalysis.myDBSCAN()
        y_predict=y[n2-n0:n3-n0]
        Diff=[]
        count=0
        totalCount=0
        predict=[]
        actual=[]
        
        for i in range(len(y_test)):
            Diff.append(y_test[i]-y_predict[i])
            predict.append(y_predict[i])
            actual.append(y_test[i])
            if Diff[i]==0:
                count=count+1
            else:
                count=count
            totalCount=totalCount+1
            i+=1
        
        print('fitness =', count/totalCount)
        cm=confusion_matrix(y_test,y_predict)
        print(cm)
        print('Accuracy'+str(object=accuracy_score(y_test,y_predict)))
        mse= mean_squared_error(actual,predict,multioutput='raw_values')
        print('mse = ',mse)
        
        f= open(log_dir+'svm_log.txt','a') 
        f.write('----------------------------------------------------\n')
        f.write('binary fitness={}\n'.format(count/totalCount))
        f.write('mse={}\n'.format(mse))
        f.close()
        
        
        #d=np.concatenate((y_test,y_predict),axis=1)
        #df_output=pd.DataFrame(data=d)
        df_output = pd.DataFrame.from_records({'Actual':y_test,'Predict':predict},index='Actual')
        df_output.to_csv(csv_dir+'slpcnn_output_'+timeSequence+'.csv')
        
        plt.plot(y_predict,color='red',label='prediction')
        plt.plot(y_test,color='blue',label='actual')
        plt.xlabel('Counts')
        plt.ylabel('Validity')
        plt.legend()
        fig = plt.gcf()
        fig.set_size_inches(15,7)
        #plt.show()
        png_name_svm = 'prediction_svn_line_'+str(object=n2)+'_'+timeSequence+'.png'
        plt.savefig(graph_dir+png_name_svm)
        plt.close()
        '''
        x,y,k_means=SVMAnalysis.myKmeans()
        sse=0
        f= open(log_dir+'kmeans_log.txt','w')
        f.write('----------------------------------------------------\n') 
        for i in range(0,n_class):
            f.write('class{}\n'.format(i))
            #print('class',i,':',)    
            ssep=0
            for j in range(0,len(x)):
                if k_means.labels_[j]==i:
                    #print(player[j],',',points[j],',')
                    f.write('parameters={}\n'.format(x[j]))
                    ssep +=sqeuclidean(k_means.cluster_centers_[i],x[j])
                    sse +=sqeuclidean(k_means.cluster_centers_[i],x[j])
            f.write('SSE P:{}\n'.format(ssep))
            #print('SSE P:',ssep)
            #print('\n')
        f.write('SSE T_test:{}\n'.format(k_means.inertia_))
        f.write('SSE T_test_2:{}\n'.format(sse))
        f.close()
        #print('SSE T_test:',k_means.inertia_)
        #print('\n')
        #print('SSE T_test_2:',sse)
        '''

        del x_test_pca
        del y_test
        #del x_train_pca
        del y_train
        del y_predict
        #del x
        #del y
        #del k_means

        
if __name__=='__main__':
    x=SVMAnalysis
    #x.myKmeans()
    #x.mySVM()
    x.myVisualize()
