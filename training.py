import numpy as np
import matplotlib
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from tensorflow.keras.models import load_model
import time
import copy
import os
from scipy import stats
import pickle 
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

df1 = pd.read_csv('database/p3t5s10000v25da.csv')
df2 = pd.read_csv('database/p3t5s10000v23da.csv')
times = 5

class operation:
    @staticmethod
    def scalarMulti(c, vector):
        vector0 = vector.copy()
        for i in range(len(vector)):
            vector0[i] = vector[i].copy()
            if (np.ndim(vector[i]) == 1):
                for j in range(np.size(vector[i])):
                    vector0[i][j] = vector[i][j] * c     
            elif (np.ndim(vector[i]) == 2):
                for j in range(np.shape(vector[i])[0]):
                    for k in range(np.shape(vector[i])[1]):
                        vector0[i][j][k] = vector[i][j][k] * c
            else:
                print('Error!')
                return -1

        return vector0
    
    @staticmethod
    def vectorSum (vector1, vector2, subtraction = 0):
        vector = vector1.copy()
        for i in range(len(vector)):
            vector[i] = vector1[i].copy()
            if (np.ndim(vector[i]) == 1):
                for j in range(np.size(vector[i])):
                    if(subtraction == 1):
                        vector[i][j] = vector1[i][j] - vector2[i][j]
                    else:
                        vector[i][j] = vector1[i][j] + vector2[i][j]
            elif (np.ndim(vector[i]) == 2):
                for j in range(np.shape(vector[i])[0]):
                    for k in range(np.shape(vector[i])[1]):
                        if (subtraction == 1):
                            vector[i][j][k] = vector1[i][j][k] - vector2[i][j][k]
                        else:
                            vector[i][j][k] = vector1[i][j][k] + vector2[i][j][k]
        return vector
    
    @staticmethod
    def vectorRandom(vector):
        vector0 = vector.copy()
        for i in range(len(vector)):
            vector0[i] = vector[i].copy()
            if (np.ndim(vector[i]) == 1):
                for j in range(np.size(vector[i])):
                    vector0[i][j] = np.random.random()*0.5 
            elif (np.ndim(vector[i]) == 2):
                for j in range(np.shape(vector[i])[0]):
                    for k in range(np.shape(vector[i])[1]):
                        vector0[i][j][k] = np.random.random()*0.5     
            else:
                print('Error!')
                return -1

        return vector0    
    
    @staticmethod
    def displacementUpdate (displacement, vector, vmax = 0.1, mode = 0):
        if mode == 0:
            for i in range(len(vector)):
                if (np.ndim(vector[i]) == 1):
                    for j in range(np.size(vector[i])):
                        if (vector[i][j] > vmax):
                            vector[i][j] = vmax
                            displacement[i][j] = displacement[i][j] + vmax
                        elif (vector[i][j] < -vmax):
                            vector[i][j] = -vmax
                            displacement[i][j] = displacement[i][j] - vmax
                        else:
                            displacement[i][j] = displacement[i][j] + vector[i][j]   

                elif (np.ndim(vector[i]) == 2):
                    for j in range(np.shape(vector[i])[0]):
                        for k in range(np.shape(vector[i])[1]):
                            if (vector[i][j][k] > vmax):
                                vector[i][j][k] = vmax
                                displacement[i][j][k] = displacement[i][j][k] + vmax
                            elif (vector[i][j][k] < -vmax):
                                vector[i][j][k] = -vmax
                                displacement[i][j][k] = displacement[i][j][k] - vmax
                            else:
                                displacement[i][j][k] = displacement[i][j][k] + vector[i][j][k]  
                                
        elif mode == 1:
            for i in range(len(vector)):
                if (np.ndim(vector[i]) == 1):
                    for j in range(np.size(vector[i])):
                        if (vector[i][j] > vmax[i][j]):
                            vector[i][j] = vmax[i][j]
                            displacement[i][j] = displacement[i][j] + vmax[i][j]
                        elif (vector[i][j] < -vmax[i][j]):
                            vector[i][j] = -vmax[i][j]
                            displacement[i][j] = displacement[i][j] - vmax[i][j]
                        else:
                            displacement[i][j] = displacement[i][j] + vector[i][j]   

                elif (np.ndim(vector[i]) == 2):
                    for j in range(np.shape(vector[i])[0]):
                        for k in range(np.shape(vector[i])[1]):
                            if (vector[i][j][k] > vmax[i][j][k]):
                                vector[i][j][k] = vmax[i][j][k]
                                displacement[i][j][k] = displacement[i][j][k] + vmax[i][j][k]
                            elif (vector[i][j][k] < -vmax[i][j][k]):
                                vector[i][j][k] = -vmax[i][j][k]
                                displacement[i][j][k] = displacement[i][j][k] - vmax[i][j][k]
                            else:
                                displacement[i][j][k] = displacement[i][j][k] + vector[i][j][k]              

                                
    @staticmethod
    def distance(displacement1, displacement, vmax = 0.1, mode = 0):
        displacement = operation.vectorSum(displacement1, displacement2,subtraction = 1)
        dsum = 0
        for i in range(len(displacementx)):
            if (np.ndim(displacement[i]) == 1):
                for j in range(np.size(vector[i])):
                        dsum = dsum + displacement[i][j]*displacement[i][j]
            elif (np.ndim(displacement[i]) == 2):
                for j in range(np.shape(displacement[i])[0]):
                    for k in range(np.shape(displacement[i])[1]):
                        dsum = dsum + displacement[i][j][k]*displacement[i][j][k]
        return dsum

class Particle:
    
    def __init__(self, ID,arguNumber = [50,25], epoch_size = 20, batch_size = 50, model = 0, displacement = 0, scale = 0.1, countmax = 500,\
                 loss = 10, pbestDense = 0,pbestLoss = 0, v = 0, historyLoss = 10, lossProcess = [],count = 0, vmax = 0, random = 0,trainsize = 10000, testsize = 10000):
        self.ID = ID
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.model = model
        self.displacement = displacement
        self.loss = loss
        self.pbestDense = pbestDense
        self.pbestLoss = pbestLoss
        self.v = v
        self.lossProcess = lossProcess
        self.historyLoss = historyLoss
        self.count = count
        self.vmax = vmax
        self.model = model
        self.arguNumber = arguNumber
        self.scale = scale
        self.random = random
        self.trainx = np.array(df2.iloc[:,0:5])
        self.trainy = np.array(df2.iloc[:,5:int(2000/times)+5]) + np.random.normal(0,scale,np.shape(np.array(df2.iloc[:,5:int(2000/times)+5])))
        self.testx = np.array(df1.iloc[:,0:5])
        self.testy = np.array(df1.iloc[:,5:int(2000/times)+5]) + np.random.normal(0,scale,np.shape(np.array(df1.iloc[:,5:int(2000/times)+5])))
        self.countmax = countmax
        
        self.x_train,self.x_val_test,self.y_train,self.y_val_test = \
        train_test_split(self.trainx,self.trainy,test_size = 0.1, random_state = 25)
        self.scaler = MinMaxScaler(feature_range=(-1,1))
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_val_test = self.scaler.transform(self.x_val_test)
        self.x_test = self.scaler.transform(self.testx)
        
        self.trainPredict = 0
        self.trainLoss = 0
        self.testPredict = 0
        self.testLoss = 0
        self.trainsize = trainsize
        self.testsize = testsize
        
    def modelInit(self):
        #根据已有模型初始化
        self.historyLoss = np.zeros(self.countmax*self.epoch_size)
        self.train()
        self.v = operation.scalarMulti(0,self.displacement.copy())
        self.pbestDense = self.displacement.copy()
        self.pbestLoss = self.loss
        
        self.train()
        self.v = operation.scalarMulti(0,self.displacement.copy())
        self.pbestDense = self.displacement.copy()
        self.pbestLoss = self.loss
        
    def initialize(self,times = 5):
        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(self.arguNumber[0],input_dim=int(2000/times),activation='relu'))
        for i in range(len(self.arguNumber)-1):
            self.model.add(layers.Dense(self.arguNumber[i+1],activation='relu'))
            self.model.add(Dropout(0.01))
        self.model.add(layers.Dense(5,activation='relu'))
        self.model.compile(optimizer='adam',loss='mse')
        if self.random == 1:
            self.displacement = self.model.get_weights() 
            vr = operation.vectorRandom(self.displacement)
            self.displacement = vr
            self.model.set_weights(self.displacement)
            
        self.historyLoss = np.zeros(self.countmax*self.epoch_size)
        self.train()
        self.v = operation.scalarMulti(0,self.displacement.copy())
        self.pbestDense = self.displacement.copy()
        self.pbestLoss = self.loss
        
        
        
    def pbestUpdate(self):
        if (self.loss<self.pbestLoss):
            self.pbestDense = self.displacement.copy()
            self.pbestLoss = self.loss
        
    def train(self):
        history=self.model.fit(self.y_train,self.x_train,batch_size=self.batch_size,epochs=self.epoch_size,\
                               verbose=0, validation_data=(self.y_val_test,self.x_val_test))
        self.loss = history.history['loss'][-1]
        self.historyLoss[self.count*self.epoch_size:(self.count+1)*self.epoch_size] = history.history['loss']
        self.displacement = self.model.get_weights() 
        self.pbestUpdate()
        self.count = self.count + 1
        
    def psoUpdate(self, gbestDense, c1 = 2, c2 = 2,omega = 0.4, vmax = 0.1):
        omega = 0.4
        for i in range(self.epoch_size):
            fst = operation.scalarMulti(omega,self.v)
            scd = operation.scalarMulti(c1*np.random.random(),operation.vectorSum(self.pbestDense, self.displacement,subtraction = 1))
            trd = operation.scalarMulti(c2*np.random.random(),operation.vectorSum(gbestDense, self.displacement,subtraction = 1))
            self.v =  operation.vectorSum(operation.vectorSum(fst,scd),trd)
            operation.displacementUpdate(self.displacement, self.v, vmax = vmax)
            self.model.set_weights(self.displacement)
            self.loss = self.model.evaluate(self.trainy,self.trainx,verbose = 0)
            self.pbestUpdate()
            self.historyLoss[self.count*self.epoch_size+i] = self.loss
        self.count = self.count+1
        
    def plotLoss(self, mode = 0,begin = 0, end = 0,):
        if mode == 1:
            plt.plot(np.arange(begin*self.epoch_size,end*self.epoch_size), self.historyLoss[begin*self.epoch_size:end*self.epoch_size])
        else:
            plt.plot(np.arange(self.count*self.epoch_size), self.historyLoss[0:self.count*self.epoch_size])
#         plt.legend(["Particle {0}".format(self.ID)])
#         plt.xticks(np.arange(1,self.count*self.epoch_size+1))
    
    def test(self):
        self.trainPredict = self.model.predict(np.array(self.trainy))
        self.trainPredictOriginal = self.scaler.inverse_transform(self.trainPredict)
        self.trainLoss = mean_squared_error(self.trainPredictOriginal, self.trainx)
        self.testPredict = self.model.predict(np.array(self.testy))
        self.testPredictOriginal = self.scaler.inverse_transform(self.testPredict)
        self.testLoss = mean_squared_error(self.testPredictOriginal, self.testx)
        self.trainCor = stats.pearsonr(np.reshape(self.trainx,[1,np.size(self.trainx)])[0],\
                                       np.reshape(self.trainPredictOriginal,[1,np.size(self.trainPredictOriginal)])[0])
                                       
                                       
        self.testCor = stats.pearsonr(np.reshape(self.testx,[1,np.size(self.testx)])[0],\
                                       np.reshape(self.testPredictOriginal,[1,np.size(self.testPredictOriginal)])[0])
    
    def plotResult(self):
        plt.figure(figsize = (12,18))
        plt.subplot(5,2,1)
        plt.plot(self.trainx[:,0],self.trainPredict[:,0],'.',markersize=0.1)

        plt.subplot(5,2,2)
        plt.plot(self.testx[:,0],self.testPredict[:,0],'.',markersize=0.1)

        plt.subplot(5,2,3)
        plt.plot(self.trainx[:,1],self.trainPredict[:,1],'.',markersize=0.1)

        plt.subplot(5,2,4)
        plt.plot(self.testx[:,1],self.testPredict[:,1],'.',markersize=0.1)

        plt.subplot(5,2,5)
        plt.plot(self.trainx[:,2],self.trainPredict[:,2],'.',markersize=0.1)

        plt.subplot(5,2,6)
        plt.plot(self.testx[:,2],self.testPredict[:,2],'.',markersize=0.1)

        plt.subplot(5,2,7)
        plt.plot(self.trainx[:,3],self.trainPredict[:,3],'.',markersize=0.1)

        plt.subplot(5,2,8)
        plt.plot(self.testx[:,3],self.testPredict[:,3],'.',markersize=0.1)

        plt.subplot(5,2,9)
        plt.plot(self.trainx[:,4],self.trainPredict[:,4],'.',markersize=0.1)

        plt.subplot(5,2,10)
        plt.plot(self.testx[:,4],self.testPredict[:,4],'.',markersize=0.1)
    
    
    def copy(self):
        p = Particle(self.ID+100, arguNumber = self.arguNumber, epoch_size = self.epoch_size, batch_size = self.batch_size, model = 0, \
                     displacement = self.displacement,loss = 10, pbestDense = 0,pbestLoss = 0, v = 0, trainsize = self.trainsize, testsize = self.testsize,\
                     historyLoss = 10, lossProcess = [],count = 0, vmax = 0)
        p.initialize()
        p.ID = copy.deepcopy(self.ID)
        p.epoch_size = copy.deepcopy(self.epoch_size)
        p.batch_size = copy.deepcopy(self.batch_size)
        p.displacement = copy.deepcopy(self.displacement)
        p.loss = copy.deepcopy(self.loss)
        p.pbestDense = copy.deepcopy(self.pbestDense)
        p.pbestLoss = copy.deepcopy(self.pbestLoss)
        p.v = copy.deepcopy(self.v)
        p.lossProcess = copy.deepcopy(self.lossProcess)
        p.historyLoss = copy.deepcopy(self.historyLoss)
        p.count = copy.deepcopy(self.count)
        p.vmax = copy.deepcopy(self.vmax)
        p.trainPredict = copy.deepcopy(self.trainPredict)
        p.trainLoss = copy.deepcopy(self.trainLoss)
        p.testPredict = copy.deepcopy(self.testPredict)
        p.testLoss = copy.deepcopy(self.testLoss)
        p.model.set_weights(self.displacement) 
        p.scale = copy.deepcopy(self.scale)
        p.random = copy.deepcopy(self.random)
        return p
        
class Swarm:
    def __init__(self, arguNumber = [50,25], random = 0, amount = 10, times = 5, mode = 0, epoch_size = 20, trainsize = 10000, testsize = 10000, \
                 batch_size = 50, scale = 0.1, countmax = 500, particle_list = []):
        tic = time.time()
        self.times = times
        self.count = 0
        self.trainTime = -1
        self.psoTime = -1
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.countmax = countmax
        self.gbestHistory = np.zeros(self.countmax*self.epoch_size)
        self.meanHistory = np.zeros(self.countmax*self.epoch_size)
        self.validNumber = 0
        self.trainy = 0
        self.trainx = 0
        self.testy = 0
        self.testx  = 0
        self.trainPredict = 0
        self.testPredict = 0
        self.testAverageLoss = 0
        self.arguNumber = arguNumber
        self.amount = amount
        self.trainsize = trainsize
        self.testsize = testsize
        self.scale = scale
        if mode == 0:
            self.initialize(amount, times = self.times,random = random)
        elif mode == 1:
            self.group = particle_list
            self.amount = len(self.group)
            self.groupLoss = self.group.copy()
            self.historyLoss = np.ones([len(particle_list),self.countmax*self.epoch_size])
            for i in range(len(self.group)):
                self.group[i] = particle_list[i].copy()
                self.groupLoss[i] = copy.deepcopy(self.group[i].loss)
                self.historyLoss[i,:] = copy.deepcopy(self.group[i].historyLoss)
           
            self.gbestLoss = np.min(self.groupLoss)
            self.gbest = self.group[self.groupLoss.index(min(self.groupLoss))]  
            self.count = self.count + 1
            
            
        self.gbestHistory= self.gbest.historyLoss
        self.meanHistory = copy.deepcopy(np.mean(self.historyLoss,0))
        toc = time.time()
        self.testLoss = self.groupLoss.copy()
        
        self.initTime = toc - tic
        
    def initialize(self,amount, times = 5, random = 0):
        self.group = list([])
        self.groupLoss = list([])
        self.historyLoss = np.ones([amount,self.countmax*self.epoch_size])
        for i in range(amount):
            particle = Particle(i, random = random, scale = self.scale, arguNumber = self.arguNumber, epoch_size= self.epoch_size, \
                                batch_size = self.batch_size, trainsize = self.trainsize, testsize= self.testsize)
            particle.initialize()
            self.group.append(particle)
            self.groupLoss.append(particle.loss)
            self.historyLoss[i,:] = copy.deepcopy(particle.historyLoss)

        self.gbestLoss = copy.deepcopy(np.min(self.groupLoss))
        self.gbest = self.group[self.groupLoss.index(min(self.groupLoss))]
        self.count = self.count + 1
        self.valid()
        
        
    def valid(self):
        self.validNumber = 0
        for i in range(self.amount):
            if self.groupLoss[i] < 0.4:
                self.validNumber = self.validNumber + 1
                
    def train(self):
        tic = time.time()
        for i in range(self.amount):
            self.group[i].train()
            self.historyLoss[i,self.count] = self.group[i].loss
            self.groupLoss[i] = self.group[i].loss
            self.historyLoss[i,:] = self.group[i].historyLoss
            
            
        self.gbestLoss = np.min(self.groupLoss)
        self.gbest = self.group[self.groupLoss.index(min(self.groupLoss))]
        self.gbestHistory[self.count] = self.gbest.loss
        self.meanHistory = np.mean(self.historyLoss,0)
        self.count = self.count + 1
        toc = time.time()
        self.valid()
        self.trainTime = toc-tic
        
    def psoUpdate(self,c1 = 2, c2 = 2,omega = 0.4, vmax = 0.1):
        tic = time.time()
        for i in range(self.amount):
            self.group[i].psoUpdate(self.gbest.displacement,c1 = c1, c2 = c2, omega = omega, vmax = vmax)
            self.historyLoss[i,self.count] = self.group[i].loss
            self.groupLoss[i] = self.group[i].loss
            self.historyLoss[i,:] = self.group[i].historyLoss
        
        self.valid()
        self.gbestLoss = np.min(self.groupLoss)
        self.gbest = self.group[self.groupLoss.index(min(self.groupLoss))]
        self.gbestHistory[self.count] = self.gbest.loss
        self.meanHistory = np.mean(self.historyLoss,0)
        self.count = self.count + 1
        toc = time.time()
        self.psoTime =  toc-tic
            
    def plotLoss(self, mode = 0,begin = 0, end = 0,):
        X0 = np.empty([self.count*self.epoch_size,self.amount])
        for i in range(self.amount):
            X0[0:self.count*self.epoch_size,i] = np.arange(1,self.count*self.epoch_size+1)
        if mode == 0:
            plt.plot(X0, self.historyLoss[:,0:self.count*self.epoch_size].T)
        elif mode == 1:
            plt.plot(X0[begin*self.epoch_size:end*self.epoch_size,:], \
                     self.historyLoss[:,begin*self.epoch_size,end*self.epoch_size].T)
    
    def plotAll(self, mode = 0,begin = 0, end = 0,):
        if mode == 0:
            plt.plot(np.arange(1,self.count*self.epoch_size+1), self.gbestHistory[0:self.count*self.epoch_size]) 
            plt.plot(np.arange(1,self.count*self.epoch_size+1), self.meanHistory[0:self.count*self.epoch_size])
        elif mode == 1:
            plt.plot(np.arange(begin*self.epoch_size,end*self.epoch_size),\
                     self.gbestHistory[begin*self.epoch_size,end*self.epoch_size]) 
            plt.plot(np.arange(begin*self.epoch_size,end*self.epoch_size), \
                     self.meanHistory[begin*self.epoch_size,end*self.epoch_size])     
    
    def test(self):
        self.testLoss = self.groupLoss.copy()
        for i in range(self.amount):
            self.group[i].test()
            self.testLoss[i] = self.group[i].testLoss
            
        self.testAverageLoss = np.mean(self.testLoss)
        self.trainx = np.zeros([self.amount*self.trainsize,5])
        self.testx  = np.zeros([self.amount*self.testsize,5])
        self.trainPredict = np.zeros([self.amount*self.trainsize,5])
        self.testPredict = np.zeros([self.amount*self.testsize,5])
        self.trainCor = []
        self.testCor = []
        
        for i in range(0,self.amount):
            self.trainx[i*self.trainsize:(i+1)*self.trainsize,:] = self.group[i].trainx
            self.testx[i*self.testsize:(i+1)*self.testsize,:] = self.group[i].testx
            self.trainPredict[i*self.trainsize:(i+1)*self.trainsize,:] = self.group[i].trainPredictOriginal
            self.testPredict[i*self.testsize:(i+1)*self.testsize,:] = self.group[i].testPredictOriginal 
            
        for j in range(0,5):                                                  
            self.trainCor.append(stats.pearsonr(self.trainx[:,j],self.trainPredict[:,j]))
            self.testCor.append(stats.pearsonr(self.testx[:,j],self.testPredict[:,j]))

                                                  
    def plotResult(self):
        
        plt.figure(figsize = (12,18))
        x1min = 1
        x1max = 2
        x2min = 2
        x2max = 3
        x3min = 1.5
        x3max = 2.5
        x4min = 4.5
        x4max = 5
        x5min = 4.8
        x5max = 5.5
        
        label=np.linspace(x1min,x1max,num=10)
        ax1 = plt.subplot(5,2,1)
        plt.plot(self.trainx[:,0],self.trainPredict[:,0],'.',markersize=0.5)
        ax1.set_xlim(x1min,x1max)
        ax1.set_ylim(x1min,x1max)
        
        
        ax2 = plt.subplot(5,2,2)
        plt.plot(self.testx[:,0],self.testPredict[:,0],'.',markersize=0.5)
        ax2.set_xlim(x1min,x1max)
        ax2.set_ylim(x1min,x1max)
        
        ax3 = plt.subplot(5,2,3)
        plt.plot(self.trainx[:,1],self.trainPredict[:,1],'.',markersize=0.5)
        ax3.set_xlim(x2min,x2max)
        ax3.set_ylim(x2min,x2max)
        
        ax4 = plt.subplot(5,2,4)
        plt.plot(self.testx[:,1],self.testPredict[:,1],'.',markersize=0.5)
        ax4.set_xlim(x2min,x2max)
        ax4.set_ylim(x2min,x2max)
        
        ax5 = plt.subplot(5,2,5)
        plt.plot(self.trainx[:,2],self.trainPredict[:,2],'.',markersize=0.5)
        ax5.set_xlim(x3min,x3max)
        ax5.set_ylim(x3min,x3max)
        
        ax6 = plt.subplot(5,2,6)
        plt.plot(self.testx[:,2],self.testPredict[:,2],'.',markersize=0.5)
        ax6.set_xlim(x3min,x3max)
        ax6.set_ylim(x3min,x3max)
        
        ax7 = plt.subplot(5,2,7)
        plt.plot(self.trainx[:,3],self.trainPredict[:,3],'.',markersize=0.5)
        ax7.set_xlim(x4min,x4max)
        ax7.set_ylim(x4min,x4max)
        
        ax8 = plt.subplot(5,2,8)
        plt.plot(self.testx[:,3],self.testPredict[:,3],'.',markersize=0.5)
        ax8.set_xlim(x4min,x4max)
        ax8.set_ylim(x4min,x4max)
        
        ax9 = plt.subplot(5,2,9)
        plt.plot(self.trainx[:,4],self.trainPredict[:,4],'.',markersize=0.5)
        ax9.set_xlim(x5min,x5max)
        ax9.set_ylim(x5min,x5max)
        
        ax10 = plt.subplot(5,2,10)
        plt.plot(self.testx[:,4],self.testPredict[:,4],'.',markersize=0.5)        
        ax10.set_xlim(x5min,x5max)
        ax10.set_ylim(x5min,x5max)    
    
        plt.show()
        
    def copy(self):
        s = Swarm(arguNumber = self.arguNumber, amount = self.amount, times = self.times, mode = 0,trainsize = self.trainsize, testsize = self.testsize, \
                  epoch_size = self.epoch_size, batch_size = self.batch_size, countmax = self.countmax, particle_list = [])
        
        s.times = copy.deepcopy(self.times)
        s.count = copy.deepcopy(self.count)
        s.psoTime = copy.deepcopy(self.psoTime)
        s.epoch_size = copy.deepcopy(self.epoch_size)
        s.batch_size = copy.deepcopy(self.batch_size)
        s.gbestHistory = copy.deepcopy(self.gbestHistory)
        s.meanHistory = copy.deepcopy(self.meanHistory)
        s.validNumber = copy.deepcopy(self.validNumber)
        s.historyLoss = copy.deepcopy(self.historyLoss)
        s.gbest = self.gbest.copy()
        s.gbestLoss = copy.deepcopy(self.gbestLoss)
        s.amount = copy.deepcopy(self.amount)
        s.initTime = copy.deepcopy(self.initTime)
        s.trainTime = copy.deepcopy(self.trainTime)
        s.testAverageLoss = copy.deepcopy(self.testAverageLoss)
        for i in range(self.amount):
            s.group[i] = self.group[i].copy()
            s.groupLoss[i] = copy.deepcopy(self.groupLoss[i])
            s.testLoss[i] = copy.deepcopy(self.testLoss[i])
    
        s.trainy = copy.deepcopy(self.trainy)
        s.trainx = copy.deepcopy(self.trainx)
        s.testy = copy.deepcopy(self.testy)
        s.testx  = copy.deepcopy(self.testx )
        s.trainPredict = copy.deepcopy(self.trainPredict)
        s.testPredict = copy.deepcopy(self.testPredict)
        s.scale = copy.deepcopy(self.scale)
        return s
    def save(self,name = 'saveswarm/'):
        self.savefile = {"times":self.times,"count":self.count,"epoch_size":self.epoch_size,"batch_size":self.batch_size,\
                        "validNumber":self.validNumber,"gbestLoss":self.gbestLoss,\
                         "amount":self.amount,"initTime":self.initTime,"trainTime":self.trainTime,# "testAverageLoss":self.testAverageLoss,
                         "scale":self.scale,"psoTime":self.psoTime,\
                         "scale":self.scale}
        np.save('my_file.npy',self.savefile)
        gbestHistory = pd.DataFrame(self.gbestHistory)
        meanHistory = pd.DataFrame(self.meanHistory)
        historyLoss = pd.DataFrame(self.historyLoss)
        trainy = pd.DataFrame(self.trainy)
        trainx = pd.DataFrame(self.trainx)
        trainPredict = pd.DataFrame(self.trainPredict)
        testPredict = pd.DataFrame(self.testPredict)
        
    def load(self,name = '/saveswarm/'):
        load_dict = np.load('my_file.npy',allow_pickle=True)
        loadData = load_dict.item()
        self.times = loadData['times']
        self.count = loadData['count']
        self.epoch_size = loadData['epoch_size']
        self.batch_size = loadData['batch_size']
        self.validNumber = loadData['validNumber']
        self.initTime = loadData['initTime']
        self.trainTime = loadData['trainTime']
        self.scale = loadData['scale']
        self.psoTime = loadData['psoTime']

for j in range(1):
    r = 1
    tic = time.time()
    s1 = Swarm(amount = 20,epoch_size = 10, scale = 0.1, trainsize = 10000, testsize = 10000)
    s1.train()
    s1.train()
    s1.train()
    s1.train()
    s1.train()
    s1.train()
#     s2 = s1.copy()
#     s2.train()
    s1.psoUpdate(omega = 0.01,vmax =0.1)
    toc = time.time()
    print("PSO: meanError is {0}, gbestError is {1}, time cost is {2}, validNumber is {3}".\
          format(s1.meanHistory[s1.count*s1.epoch_size-1],s1.gbest.loss, toc-tic, s1.validNumber))
#     print("Tra: meanError is {0}, gbestError is {1}, time cost is {2}, validNumber is {3}".\
#           format(s2.meanHistory[s2.count*s2.epoch_size-1],s2.gbest.loss, toc-tic, s2.validNumber))
#     tic0 = time.time()
    for i in range(100):
        tic = time.time()
        s1.train()
#         s2.train()
        toc = time.time()
        print('---------------------------------')
        
        print("PSO: meanError is {0}, gbestError is {1}, time cost is {2}, validNumber is {3}".\
              format(s1.meanHistory[s1.count*s1.epoch_size-1], s1.gbest.loss, s1.psoTime, s1.validNumber))
#         print("Tra: meanError is {0}, gbestError is {1}, time cost is {2}, validNumber is {3}".\
#               format(s2.meanHistory[s2.count*s1.epoch_size-1], s2.gbest.loss, s2.trainTime, s2.validNumber))
        if s1.validNumber == s1.amount:
            toc = time.time()
            print('i is {0}, total time cost is {1}, meanError is {2} ,gbestError is {3}, '.\
                  format(i,toc-tic,s1.meanHistory[s1.count*s1.epoch_size-1],s1.gbest.loss))
            if i%5 == 0:
                s1.test()
#                 s2.test()
                print("PSO:trainCOR:{0}".format(s1.trainCor[r]))
                print("PSO:testCOR:{0}".format(s1.testCor[r]))
#                 print("TRA:COR:{0}".format(s2.trainCor[0]))
                print("************************")
            s1.psoUpdate(omega = 0.01,vmax =0.1)

