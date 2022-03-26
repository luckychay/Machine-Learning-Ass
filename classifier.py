'''
Description: 
Version: 
Author: Xuanying Chen
Date: 2022-03-23 18:03:15
LastEditTime: 2022-03-26 14:33:29
'''

from cmath import exp, pi
from enum import unique
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
import joblib
import argparse
from sklearn.manifold import TSNE


class Classifier:
        
    def train(self,X,y):
        pass
    def test(self,X):
        pass


class Bayes(Classifier):
    def __init__(self) -> None:
        self.class_num = 0
        self.classes = None
        self.M = None  
        self.C = None

    def Guassian(self,m,c,X):
        
        X_m_transpose = np.reshape(X-m,(-1,1))
        c_inv = np.linalg.inv(c)
        res = np.math.exp(-0.5*(X-m)@c_inv@X_m_transpose)      \
                /(2*pi*np.math.sqrt(np.linalg.det(c)))
        return res

    def train(self,X,y):

        classes = np.unique(y)
        self.classes = classes
        self.class_num = len(classes)

        ## get mean vector and covariance matrices of each class
        M = np.zeros(shape=(self.class_num,X.shape[1]))
        C = np.zeros(shape=(self.class_num,X.shape[1],X.shape[1]))
        for i,c in enumerate(classes):
            index = np.where(y==c)[0]
            M[i] = np.mean(X[index],axis=0)

            tmp = np.zeros(shape=(X.shape[1],X.shape[1]))
            for j in range(X[index].shape[0]):
                tmp += np.outer((X[index][j] - M[i]),X[index][j] - M[i])
            tmp /= X[index].shape[0]
            C[i] = tmp

        self.M = M
        self.C = C
        
    def test(self,X):
        test_num = X.shape[0]
        Y = np.zeros(test_num,dtype=int)
        for i in range(test_num):
            scores = np.zeros(self.class_num)
            for j in range(self.class_num):
                scores[j] = self.Guassian(self.M[j],self.C[j],X[i])
            Y[i] = self.classes[np.argmax(scores)]

        return Y


class LDA(Classifier):
    def __init__(self) -> None:

        pass
    def train(self):
        pass
    def test(self,X):
        pass


class Decision_tree(Classifier):
    def __init__(self) -> None:

        pass
    def train(self):
        pass
    def test(self,X):
        pass

def tSNE_visualize(X,y,title="data T-SNE projection"):

    embedded =  TSNE(n_components=2,verbose=1, random_state=123).fit_transform(X)
    df = pd.DataFrame()
    df['y'] = y
    df["comp-1"] = embedded[:,0]
    df["comp-2"] = embedded[:,1]

    num_components = np.unique(y).shape[0]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", num_components),
                data=df).set(title=title)

    plt.show()
    

def loadmat(mat_filename):
    data = scipy.io.loadmat(mat_filename)
    key_name = mat_filename.split('/')[-1].split('.')[0]
    data = np.array(data[key_name])
    return data

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type= str, default='train', help='choose runtime mode')
    args = parser.parse_args()
    if args.mode == 'train':
        
        print('loading data...')
        train_X = loadmat("./Data_Train.mat")
        train_y = loadmat("./Label_Train.mat")
        print('finish loading data.')

        print('begin training...')
        cls = Bayes()
        cls.train(train_X,train_y)
        print('finish training.')
        
        joblib.dump(cls,"{}.pkl".format(type(cls).__name__))
        print('model saved.')

    elif args.mode == 'test':
        
        print('loading data and model...')
        cls = joblib.load("Bayes.pkl")
        test_X = loadmat("./Data_test.mat")
        print('finish loading data and model.')
    
        print('begin testing...')
        test_y = cls.test(test_X)
        test_y += 10
        np.save('test_y',test_y)
        print('predict result:',test_y)

    elif args.mode == "vis":
        train_X = loadmat("./Data_Train.mat")
        train_y = loadmat("./Label_Train.mat")
        test_X = loadmat("./Data_test.mat")
        test_y = np.load("test_y.npy")
        print(test_y)
        X = np.append(train_X,test_X,axis=0)
        y = np.append(np.reshape(train_y,(-1,)),test_y)

        tSNE_visualize(X,y,"test result")


       