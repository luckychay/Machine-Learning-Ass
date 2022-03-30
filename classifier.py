'''
Description: 
Version: 
Author: Xuanying Chen
Date: 2022-03-23 18:03:15
LastEditTime: 2022-03-30 11:17:18
'''

from cmath import exp, pi
from enum import unique
from time import sleep

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
from scipy.linalg import eigh
import joblib
import argparse

from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier


class Bayes():
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


class MLDA():
    def __init__(self) -> None:
        self.class_num = 0
        self.classes = None
        self.left_class = 0
        self.used_classes = None
        self.total_mean_vec = None
        self.classes_mean_vecs = None  
        self._significance = None
        self.St = None
        self.Sw = None
        self.Sb = None
        self.W  = None
        self.W0 = None

    def _cal_mean_vec(self,vectors):
        
        return np.mean(vectors,axis=0)

    def _cal_Sb(self):

        return self.St-self.Sw

    def _cal_Sw(self,scatter_matrices):

        return np.sum(scatter_matrices,axis=0)

    def _cal_Sb_direct(self,y):

        Sb = np.zeros_like(self.Sw)
        for m, c in zip(self.classes_mean_vecs,self.classes):
            n = np.where(y==c)[0].shape[0]
            Sb += n * np.outer(m-self.total_mean_vec,m-self.total_mean_vec) 

        return Sb/y.shape[0]

    def _cal_scatter_matrix(self,vectors,mean):

        ## should not normalize(divid by n) for normal definition

        s = np.zeros(shape=(vectors.shape[1],vectors.shape[1]))
        for vec in vectors:
            s += np.outer(vec-mean,vec-mean)
         
        return s/vectors.shape[0]

    def MLDC_train(self,mean_proj):
        '''Multicalss Marginal Linear Discriminant Classifier(MLDC) train'''

        t = int(0.5 * self.class_num*(self.class_num-1))
        self.W0 = np.zeros(shape=(t,self.class_num-1))
        self.CN = np.zeros_like(self.W0)
        self.CP = np.zeros_like(self.W0)

        k = 0
        for i in range(t-1):
            for j in range(i+1,t):
                self.W0[k] = -0.5 * (mean_proj[i,:] + mean_proj[j,:])
                comp = mean_proj[i,:] - mean_proj[j,:]
                neg_filter = (comp <= 0).astype(int)
                pos_filter = (comp > 0).astype(int)
                self.CN[k,:] = i * neg_filter + j * pos_filter
                self.CP[k,:] = j * neg_filter + i * pos_filter
                k += 1

        self.CP += 1
        self.CN += 1
        print("W0:",self.W0)
        print("CP:",self.CP)
        print("CN:",self.CN)

    def MLDC_test(self,X):
        '''Multicalss Marginal Linear Discriminant Classifier(MLDC) test'''

        t = int(0.5 * self.class_num*(self.class_num-1))
        y = X @ self.W.T
        y = (X @ self.W.T).reshape(y.shape[0],1,y.shape[1])

        decision_mat = np.repeat(y,t,axis=1) + np.repeat(self.W0[np.newaxis,:,:],X.shape[0],axis=0) 

        posi_mask = (decision_mat > 0).astype(int)
        neg_mask  = (decision_mat < 0).astype(int)

        repeat_CP = np.repeat(self.CP[np.newaxis,:,:],X.shape[0],axis=0)
        repeat_CN = np.repeat(self.CN[np.newaxis,:,:],X.shape[0],axis=0)

        res = repeat_CP * posi_mask + repeat_CN * neg_mask

        print(res)

    def dim_reduction(self,X,y):
        '''from n*m dim to c*c-1 dim mapping, where c is the class number of X'''
        '''faced some issue here, should be recorded, mind different definition of Sw and Sb'''
        
        classes = np.unique(y)
        self.classes = classes
        self.class_num = len(classes)

        self.priors_ = np.bincount(y.flatten())[1:] / float(y.shape[0])
        print(self.priors_)
        ## cal mean vec of the whole dataset
        total_mean_vec = self._cal_mean_vec(X)
        self.total_mean_vec = total_mean_vec

        ## cal total scatter matrix
        self.St = self._cal_scatter_matrix(X, total_mean_vec)

        ## cal mean vector and scatter matrix of each class
        M = np.zeros(shape=(self.class_num,X.shape[1]))
        S = np.zeros(shape=(self.class_num,X.shape[1],X.shape[1]))

        self.priors = np.zeros(self.class_num,dtype=float)
        for i,c in enumerate(classes):
            index = np.where(y==c)[0]
            self.priors[i] = index.shape[0]
            M[i] = self._cal_mean_vec(X[index])
            S[i] = self._cal_scatter_matrix(X[index],M[i])

        self.classes_mean_vecs = M
        classes_scatter_mats = S

        ## cal within-class scatter matrix
        self.Sw = self._cal_Sw(classes_scatter_mats)

        ## cal between-class scatter matrix
        self.Sb = self._cal_Sb_direct(y)
        
        ## cal eigen vector for generalized eigen value problem
        eig_num = self.class_num - 1
        eigen_vals,eigen_vecs = eigh(self.Sb,self.St-self.Sb)

        ## for classification, we use all the eigenvector 
        eigen_vecs = eigen_vecs[::-1]
        self.W = np.dot(self.classes_mean_vecs, eigen_vecs).dot(eigen_vecs.T)

        self.W0 = -0.5 * np.diag(np.dot(self.classes_mean_vecs, self.W.T)) + np.log(
            self.priors_
        )

        # ## sort the eigenvalues and eigenvectors in descending order
        # ## and pick the first c-1 ones 
        # eigen_vals = eigen_vals[::-1][:eig_num]
        # eigen_vecs = eigen_vecs.T[::-1][:eig_num]

        # self._significance = np.real(eigen_vals/np.sum(eigen_vals))
        # self.W = eigen_vecs
        # mean_proj = self.classes_mean_vecs @ self.W.T

        # print("mean_proj:\n",mean_proj)   

        # return mean_proj

    def centroid_margin_train(self,mean_proj):

        norm_mean_proj = mean_proj - np.mean(mean_proj,axis=0)
        print("norm_mean_proj:\n",norm_mean_proj)
        eig_num = self.class_num - 1

        ## simply use the avarage of biggest and second biggest
        used_index = np.argmax(mean_proj,axis=0)
        nearests = np.argsort(mean_proj,axis=0)[-2]
        self.W0 = -0.5 *(mean_proj[used_index,range(eig_num)] + mean_proj[nearests,range(eig_num)]) 
            
        self.used_classes = self.classes[used_index]
        self.left_class = np.delete(self.classes,used_index)
        print("used_index:",used_index)
        print("self.left_class",self.left_class,"self.used_class",self.used_classes)

    def centroid_margin_test(self,X):

        projects = self.W@np.transpose(X)+np.reshape(self.W0,(-1,1))
        print("projects.shape:",projects.shape)
        print(projects)
        max_scores = np.max(projects,axis=0)
        left_class_index = np.where(max_scores<0)
        test_Y = self.used_classes[np.argmax(projects,axis=0)]
        test_Y[left_class_index] = self.left_class

        return test_Y

    def significance_base_train(self,mean_proj):

        eig_num = self.class_num - 1

        ## choose eigen value according to significance
        sig = self._significance
        self.W = self.W[np.where(sig>0.1)]

        if self.W.shape[0] < eig_num:

            ## cal W0  
            mean_proj = self.classes_mean_vecs @ self.W.T## project mean vector to hyper plane
            print("mean_proj:\n",mean_proj)
            
            sort_index = np.argsort(mean_proj[:,0])
            used_index = sort_index[[0,-1]]
            print(mean_proj[sort_index])
            self.W0 = -0.5 * (np.array([mean_proj[0,0]+mean_proj[2,0],mean_proj[1,0]+mean_proj[2,0]]))
            # self.W0 = -0.5*(mean_proj[sort_index][1:,0]+mean_proj[sort_index][-1:,0])
        elif self.W.shape[0] == eig_num:
            pass

    def significance_base_test(self,X):

        projects = self.W@np.transpose(X)+np.reshape(self.W0,(-1,1))

        test_Y =np.zeros(X.shape[0])
        for j in range(projects.shape[1]):
            if projects[0,j]>0 and projects[1,j]>0:
                test_Y[j] = 1
            if projects[0,j]<0 and projects[1,j]<0:
                test_Y[j] = 2
            if projects[0,j]<0 and projects[1,j]>0:
                test_Y[j] = 3

        return test_Y

    def train(self,X,y):

        self.dim_reduction(X,y)
        
        # self.significance_base_train(proj)


    def test(self,X):

        y = X @ self.W.T + self.W0

        y_index = np.argmax(y,axis=1)
        y_test = self.classes[y_index]

        return y_test
        

class Decision_tree(DecisionTreeClassifier):
    def __init__(self) -> None:
        
        super().__init__()

    def _cal_Gini_impurity(self,y):
        
        classes = np.unique(y)
        hist = np.zeros_like(classes)
        for i, cls in enumerate(classes):
            hist[i] = y[y==cls].shape[0]

        gini_impur = 0.5*(1-np.sum((hist/y.shape[0])**2))
        
        return gini_impur

    def train(self,X,y):
        
        self.fit(X,y) 

    def test(self,X):
        
        return self.predict(X)

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
    parser.add_argument('--classifier',type = str, default='LDA',help='choose classifier type')
    args = parser.parse_args()
    if args.mode == 'train':
        
        print('loading data...')
        train_X = loadmat("./Data_Train.mat")
        train_y = loadmat("./Label_Train.mat")
        print('finish loading data.')

        print('begin training...')
        if args.classifier == 'LDA':
            cls = MLDA()
        elif args.classifier == 'Bayes':
            cls = Bayes()
        elif args.classifier == 'DT':
            cls = Decision_tree()
        cls.train(train_X,train_y)
        print('finish training.')
        
        joblib.dump(cls,"{}.pkl".format(type(cls).__name__))
        print('model saved.')

    elif args.mode == 'test':

        print('loading data and model...')
        if args.classifier == 'LDA':
            cls = joblib.load("MLDA.pkl")
        elif args.classifier == 'Bayes':
            cls = joblib.load("Bayes.pkl")
        elif args.classifier == 'DT':
            cls = joblib.load("Decision_tree.pkl")
        
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
