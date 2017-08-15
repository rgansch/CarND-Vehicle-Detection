# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 21:57:30 2017

@author: gansc
"""
import numpy as np
from scipy.ndimage.measurements import label
import cv2
from moviepy.editor import VideoFileClip

import config
from objectfinder import ObjectFinder

class ObjectTracker(config.ConfigClass):
    def __init__(self, clf=None):
        super().__init__()
        self._init_state()
        
        self._obj_find = ObjectFinder(clf)
        
    def _init_state(self):
        ''' Init the internal state '''
        self._heatmap = None
    
    def _fade_heat(self):
        ''' Fades heat from the internal heat map '''
        fadeheat = float(self._config['heatmap']['fadeheat'])
        
        if not self._heatmap is None:
            self._heatmap -= fadeheat
    
    def _add_heat(self, image):
        ''' Adds heat according to objects found in image '''
        maxheat = int(self._config['heatmap']['maxheat'])
        
        windows = self._obj_find.get_window_matches(image)
        
        # initialize heatmap of in init state
        if self._heatmap is None:
            heatmap = np.zeros_like(image[:,:,0], dtype=np.float32)
        else:
            heatmap = self._heatmap
        
        # Add heat for all windows matching the classifier
        for win in windows:
            heatmap[win[0][1]:win[1][1], win[0][0]:win[1][0]] += 1
            heatmap[heatmap > maxheat] = maxheat
        self._heatmap = heatmap
        
        return heatmap
    
    def _labels(self, heatmap):
        ''' Extract labels from heatmap '''
        threshold = int(self._config['heatmap']['threshold'])
        
        heatmap[heatmap <= threshold] = 0
        labels = label(heatmap)
        return labels
    
    def _draw_boundingboxes(self, image, labels):
        ''' Draw bounding boxes around labels on copy of image '''
        img_bb = np.copy(image)
        
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            if bbox[0][0] > 640:
                cv2.rectangle(img_bb, bbox[0], bbox[1], (0,0,1), 6)
        # Return the image
        return img_bb
    
    def _process_frame(self, image):
        ''' Process a video frame '''
        img32 = image.astype(np.float32) / 255
        self._fade_heat()
        heatmap = self._add_heat(img32)
        labels = self._labels(heatmap)
        img_bb = self._draw_boundingboxes(img32, labels)
        return img_bb*255
    
    def detect_and_track_objects(self, video_input, video_output):
        ''' Track objects in video_input and write to video_output '''
        self._init_state()
        clip = VideoFileClip(video_input)
        track_clip = clip.fl_image(self._process_frame)
        track_clip.write_videofile(video_output, audio=False)
        