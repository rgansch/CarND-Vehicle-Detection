# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:37:57 2017

@author: gansc
"""
import numpy as np
import pickle

from featureextractor import FeatureExtractor
import config

class ObjectFinder(config.ConfigClass):
    def __init__(self, clf=None):
        ''' clf is either a Classifier class instance or a string pointing
            to a pickled one '''
        super().__init__()
        
        if isinstance(clf, str):
            with open(clf , 'rb') as fid:
                clf = pickle.load(fid) 
        self._clf = clf
        
        self._feat_ext = FeatureExtractor()
        
    def _interpolate(self, point_start, point_end, factor): 
        ''' Interpolate a point between start and end '''
        x = int(point_start[0] + factor * (point_end[0] - point_start[0]))
        y = int(point_start[1] + factor * (point_end[1] - point_start[1]))
        return (x, y)
    
    def _get_z_boxes(self):
        ''' Returns the bounding boxes for the z-planes '''
        ll_n = tuple( [int(i) for i in self._config['boundingbox']['lower_left_near'].split(',')] )
        ur_n = tuple( [int(i) for i in self._config['boundingbox']['upper_right_near'].split(',')] )
        ll_f = tuple( [int(i) for i in self._config['boundingbox']['lower_left_far'].split(',')] )
        ur_f = tuple( [int(i) for i in self._config['boundingbox']['upper_right_far'].split(',')] )
        s_z = int(self._config['steps']['steps_z'])
        ws_n = tuple( [int(i) for i in self._config['window']['window_size_near'].split(',')] )
        ws_f = tuple( [int(i) for i in self._config['window']['window_size_far'].split(',')] )
        
        boxes = []
        windows = []
        for i in range(s_z):
            # calculate boxes
            ll = self._interpolate(ll_n, ll_f, i/float(s_z-1))
            ur = self._interpolate(ur_n, ur_f, i/float(s_z-1))
            boxes.append([ll, ur])
            
            # calculate windows
            ws = self._interpolate(ws_n, ws_f, i/float(s_z-1))
            windows.append(ws)
        return boxes, windows
    
    def _get_windows(self):
        ''' Returns all sliding windows and z bounding boxes '''
        s_x = int(self._config['steps']['steps_x'])
        s_y = int(self._config['steps']['steps_y'])

        z_boxes, win_sizes = self._get_z_boxes()
        
        windows = []        
        for box, ws in zip(z_boxes, win_sizes):
            # x and y points for lower left and upper rights
            ll_x = np.linspace(box[0][0], box[1][0] - ws[0], num=s_x, dtype=np.int32)
            ur_x = np.linspace(ws[0] + box[0][0], box[1][0], num=s_x, dtype=np.int32)
            ll_y = np.linspace(box[1][1], box[0][1] - ws[1], num=s_y, dtype=np.int32)
            ur_y = np.linspace(ws[1] + box[1][1], box[0][1], num=s_y, dtype=np.int32)
            
            # Create all combinations of x and y points for lower left and upper right
            ll = np.array(np.meshgrid(ll_x, ll_y)).T.reshape(-1,2)
            ur = np.array(np.meshgrid(ur_x, ur_y)).T.reshape(-1,2)
            
            # Add the x, y combinations as window
            windows.append([])
            for i in range(ll.shape[0]):
                win = [(ll[i][0], ll[i][1]), (ur[i][0], ur[i][1])]
                windows[-1].append(win)
        return windows, z_boxes    
                
    def get_window_matches(self, image):
        ''' Iterate through all sliding windows and return windows with match
            the classifier '''
        window_matches = []
        
        windows, z_boxes = self._get_windows()
        for idx_z in range(len(windows)):
            for win in windows[idx_z]:    
                # window coordinates in image
                img_win = image[win[0][1]:win[1][1], win[0][0]:win[1][0], :]
                # Extract feature vector from window
                features = self._feat_ext.feature_vector(img_win)
                # Check if classifier matches
                match = self._clf.predict(features)
                if match:
                    window_matches.append(win)
        return window_matches