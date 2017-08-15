# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 19:14:11 2017

@author: gansc
"""
import glob
import random
import numpy as np
import cv2
import matplotlib.image as mpimg

import config

class ImageLoader(config.ConfigClass):
    def __init__(self):
        super().__init__()
        
    def get_random(self, img_set):
        ''' Returns a random image from the img_set '''
        all_image_names = []
        for path in self._config[img_set].values():
            all_image_names.extend(glob.glob(path))
        
        image_name = random.choice(all_image_names)
        image = mpimg.imread(image_name)
        if image_name.endswith('.jpg'):
            image = image.astype(np.float32) / 255
        return image, image_name
    
    def get_count(self, img_set):
        ''' Returns count of totally available images in img_set '''
        all_image_names = []
        for path in self._config[img_set].values():
            all_image_names.extend(glob.glob(path))   
            
        return len(all_image_names)
    
    def get_all(self, img_set):
        ''' Returns all images in the img_set '''
        all_image_names = []
        for path in self._config[img_set].values():
            all_image_names.extend(glob.glob(path))
        
        all_images = []
        for image_name in all_image_names:
            image = mpimg.imread(image_name)
            if image_name.endswith('.jpg'):
                image = image.astype(np.float32) / 255
            all_images.append(image)
            
        return np.array(all_images)
            