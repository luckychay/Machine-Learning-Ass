'''
Description: 
Version: 
Author: Xuanying Chen
Date: 2022-03-23 18:03:15
LastEditTime: 2022-03-24 22:15:31
'''

import numpy as np
import scipy.io
import joblib
import argparse

class Classifier:
    def __init__(self) -> None:
        pass
    def train(self):
        pass
    def test(self):
        pass


class Bayes(Classifier):
    def __init__(self) -> None:
        super().__init__()
        pass
    def train(self):
        pass
    def test(self):
        pass


class LDA(Classifier):
    def __init__(self) -> None:
        super().__init__()
        pass
    def train(self):
        pass
    def test(self):
        pass


class Decision_tree(Classifier):
    def __init__(self) -> None:
        super().__init__()
        pass
    def train(self):
        pass
    def test(self):
        pass


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type= str, default='train', help='choose runtime mode')
    args = parser.parse_args()
    if args.mode == 'train':
        print('begin training...')
        X = scipy.io.loadmat("./Data_Train.mat")
        X = np.array(y["Data_Train"])
        y = scipy.io.loadmat("./Label_Train.mat")
        y = np.array(y["Label_Train"])

        cls = Bayes()

        W = np.zeros(10)
        joblib.dump(W,"data.pkl")
    elif args.mode == 'test':
        print('begin testing...')
        model= joblib.load("data.model")
        print()