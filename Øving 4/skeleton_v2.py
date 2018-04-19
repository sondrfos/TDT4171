import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import csv
import time

def logistic_z(z): 

    return 1.0/(1.0+np.exp(-z))

def logistic_wx(w,x): 

    return logistic_z(np.inner(w,x))

def classify(w,x):
    x=np.hstack(([1],x))
    return 0 if (logistic_wx(w,x)<0.5) else 1
#x_train = [number_of_samples,number_of_features] = number_of_samples x \in R^number_of_features
def stochast_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in xrange(niter):
        if(len(index_lst)==0):
            index_lst=random.sample(xrange(num_n), k=num_n)
        xy_index = index_lst.pop()
        x=x_train[xy_index,:]
        y=y_train[xy_index]
        for i in xrange(dim):
            update_grad = ((logistic_wx(w,x)-y)*x[i]*logistic_wx(w,x)*(1-logistic_wx(w,x))) ### something needs to be done here
            w[i] -= learn_rate * update_grad ### something needs to be done here
    return w
def batch_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in xrange(niter):
        for i in xrange(dim):
            update_grad=0.0
            for n in xrange(num_n):
            	x = x_train[n]
            	y = y_train[n]
                update_grad += ((logistic_wx(w,x)-y)*x*logistic_wx(w,x)*(1-logistic_wx(w,x)))# something needs to be done here
            w[i] -= learn_rate * update_grad[i]/num_n
    return w


def train_and_plot(xtrain,ytrain,xtest,ytest,training_method,learn_rate=0.1,niter=10):
    #plt.figure()
    #train data
    data = pd.DataFrame(np.hstack((xtrain,ytrain.reshape(xtrain.shape[0],1))),columns=['x','y','lab'])
    #ax=data.plot(kind='scatter',x='x',y='y',c='lab',cmap=cm.copper,edgecolors='black')

    #train weights
    w=training_method(xtrain,ytrain,learn_rate,niter)
    error=[]
    y_est=[]
    for i in xrange(len(ytest)):
        error.append(np.abs(classify(w,xtest[i])-ytest[i]))
        y_est.append(classify(w,xtest[i]))
    y_est=np.array(y_est)
    data_test = pd.DataFrame(np.hstack((xtest,y_est.reshape(xtest.shape[0],1))),columns=['x','y','lab'])
    #data_test.plot(kind='scatter',x='x',y='y',c='lab',ax=ax,cmap=cm.coolwarm,edgecolors='black')

    print "error=",np.mean(error)
    return np.mean(error)#w

def readData(separable = 1):
	if separable:
		filenameTrain = 'data_big_separable_train.csv'
		filenameTest = 'data_big_separable_test.csv'
	else:
		filenameTrain = 'data_big_nonsep_train.csv'
		filenameTest = 'data_big_nonsep_test.csv'
	with open(filenameTrain) as csvfile:
		readCSV = csv.reader(csvfile, delimiter='\t')
		x_train = []
		y_train = []
		for row in readCSV:
			x_train.append([float(row[0]), float(row[1])])
			y_train.append(float(row[2]))
	with open(filenameTest) as csvfile:
		readCSV = csv.reader(csvfile, delimiter='\t')
		x_test = []
		y_test = []
		for row in readCSV:
			x_test.append([float(row[0]), float(row[1])])
			y_test.append(float(row[2]))
	x_train = np.array(x_train)
	y_train = np.array(y_train)
	x_test = np.array(x_test)
	y_test = np.array(y_test)
	return x_train, y_train, x_test, y_test

xtrain, ytrain, xtest, ytest = readData(1)
execution = []
errors = []
T = [10, 20, 50, 100, 200, 500]
for niter in T:
	start = time.time()
	error = train_and_plot(xtrain, ytrain, xtest, ytest, stochast_train_w, niter = niter)
	print(niter, error)
	end = time.time()
	execution.append(end-start)
	errors.append(error)
plt.subplot(211)
plt.ylabel('Error rate')
plt.plot(T, errors)
plt.subplot(212)
plt.xlabel('Iterations')
plt.ylabel('Execution time')
plt.plot(T, execution)
plt.show()
#print(end-start)
#start = time.time()
#train_and_plot(xtrain, ytrain, xtest, ytest, batch_train_w, niter = 100)
#plt.show()
#end = time.time()
#print(end-start)