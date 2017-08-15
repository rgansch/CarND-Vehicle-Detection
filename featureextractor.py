# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 19:14:11 2017

@author: gansc
"""
import cv2
from skimage.feature import hog
import numpy as np

import config

class FeatureExtractor(config.ConfigClass):
    def __init__(self, full_img=None):
        super().__init__()
        
        # Define the colorspce and mappings of the channels
        self._color_space_mapping = {}
        self._color_space_mapping ['cs_HLS'] = {'cv2_id' : cv2.COLOR_RGB2HLS, 'channels' : ['H', 'L', 'S']}
        self._color_space_mapping ['cs_HSV'] = {'cv2_id' : cv2.COLOR_RGB2HSV, 'channels' : ['H', 'S', 'V']}
        self._color_space_mapping ['cs_LUV'] = {'cv2_id' : cv2.COLOR_RGB2LUV, 'channels' : ['L', 'U', 'V']}
        self._color_space_mapping ['cs_YUV'] = {'cv2_id' : cv2.COLOR_RGB2HLS, 'channels' : ['Y', 'U', 'V']}
        self._color_space_mapping ['cs_YCrCb'] = {'cv2_id' : cv2.COLOR_RGB2YCrCb, 'channels' : ['Y', 'Cr', 'Cb']}
        
    def _hog(self, image, vis=False):
        ''' Extract the Histogram of Gradient feature vector from image.
            Returns visualization image with vis=True '''
        # Fetch relevant config for feature vector
        pixels_per_cell_x = int(self._config['hog']['pixels_per_cell_x'])
        pixels_per_cell_y = int(self._config['hog']['pixels_per_cell_y'])
        pixels_per_cell = (pixels_per_cell_x, pixels_per_cell_y)
        
        cells_per_block_x = int(self._config['hog']['cells_per_block_x'])  
        cells_per_block_y = int(self._config['hog']['cells_per_block_y'])
        cells_per_block = (cells_per_block_x, cells_per_block_y)
        
        orient = int(self._config['hog']['orient'])
        
        # convert to grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)     
        # Calculate HOG and depending on vis include visualization image
        if vis == True:
            features, hog_image = hog(gray_img, orientations=orient, pixels_per_cell=pixels_per_cell,
                                      cells_per_block=cells_per_block, transform_sqrt=False, 
                                      visualise=True, feature_vector=True)
            return features, hog_image
        else:      
            features = hog(gray_img, orientations=orient, pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block, transform_sqrt=False, 
                           visualise=False, feature_vector=True)
            return features
        
    def _spatial_bin(self, image):
        ''' Extract the Spatial Binning feature vector from image '''
        # Fetch relevant config for feature vector
        sizex = int(self._config['spatial']['sizex'])
        sizey = int(self._config['spatial']['sizey'])
        size = (sizex, sizey)
        
        # Resize image and ravel to feature vector
        small_img = cv2.resize(image, size)
        feature_vec = small_img.ravel()
        return feature_vec
    
    def _convert_color_space(self, image, colorspace, channels):
        ''' Converts image to the desired colorspace and returns specified channels '''
        # Map colorspace to cv2 constant
        cv2_id = self._color_space_mapping[colorspace]['cv2_id']
        channel_mapping = self._color_space_mapping[colorspace]['channels']
        # Map channel ids to array index
        channel_id = []
        for ch in channels:
            channel_id.append(channel_mapping.index(ch))
        
        # Convert image and extract channel id
        image_channels = []
        converted_image = cv2.cvtColor(image, cv2_id)
        for ch in channel_id:
            image_channels.append(converted_image[:,:,ch])
            
        return image_channels
    
    def _color_space(self, image):
        ''' Extracts the Colorspace feature vector from image '''
        # Fetch relevant config for feature vector
        colorspaces = self._config['colorspace']['names'].split(',')
        
        # Get feature vector for all colorspaces
        feature_vector = []
        for cs in colorspaces:
            cs_channels = self._config[cs]['channel'].split(',')
            cs_nbins = int(self._config[cs]['nbins'])
            
            converted = self._convert_color_space(image, cs, cs_channels)
            for channel in converted:
                # Calculate histogram and append to feature vector
                hist = np.histogram(channel, bins=cs_nbins, range=(0,1))
                feature_vector.extend(hist[0])
        
        return np.array(feature_vector)
    
    def feature_vector(self, image, basex=None, basey=None):
        ''' Extracts the concatenated feature vector from image '''
        # Convert to 64x64 image 
        image64 = cv2.resize(image, (64, 64), interpolation = cv2.INTER_CUBIC)
        
        # Extract feature vectors
        feature_spatial = self._spatial_bin(image64)
        feature_hog = self._hog(image64)
        feature_cs = self._color_space(image64)
        
        # Concetante to full feature vector
        full_feature = np.concatenate((feature_spatial, feature_hog, feature_cs))
        return full_feature