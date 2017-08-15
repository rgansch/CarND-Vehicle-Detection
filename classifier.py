# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 19:14:11 2017

@author: gansc
"""
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler    

import config

class Classifier(config.ConfigClass):
    def __init__(self):
        super().__init__()
        
        self._x_scaler = None
        self._clf = None
        
        self._clf_func = {'SVM' : self._svc, 'DT' : self._dt}
        
    def _svc(self, feature_vectors, labels):
        C = float(self._config['SVM']['c'])
        kernel = self._config['SVM']['kernel']
        
        self._clf = SVC(kernel=kernel, C=C)
        self._clf.fit(feature_vectors, labels)
      
    def _dt(self, feature_vectors, labels):
        criterion = self._config['DT']['criterion']
        min_samples_split = int(self._config['DT']['min_samples_split'])
        min_samples_leaf = int(self._config['DT']['min_samples_leaf'])
        
        self._clf = DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf)
        self._clf.fit(feature_vectors, labels)
        
    def train(self, feature_vectors, labels):
        clf_type = self._config['classifier']['type']
        
        self._x_scaler = StandardScaler().fit(feature_vectors)
        feat_vec_scaled = self._x_scaler.transform(feature_vectors)        
        
        clf_func = self._clf_func[clf_type]
        clf_func(feat_vec_scaled, labels)
    
    def accuracy(self, feature_vectors, labels):
        labels_pred = self.predict(feature_vectors)
        accuracy = accuracy_score(labels, labels_pred)
        return accuracy
    
    def predict(self, feature_vectors):
        if feature_vectors.ndim == 1:
            feat_vec_shaped = feature_vectors.reshape(1, -1)
        else:
            feat_vec_shaped = feature_vectors

        feat_vec_scaled = self._x_scaler.transform(feat_vec_shaped)
        labels = self._clf.predict(feat_vec_scaled)
        return labels