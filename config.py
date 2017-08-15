# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 19:14:11 2017

@author: gansc
"""

import configparser
import os

class ConfigClass(object):
    def __init__(self, configfile=None, configdata=None):
        # Load default config from file
        if configfile is None:
            configfile = self.__class__.__name__ + '.ini'
        self.read_config_file(configfile)
        
        # Overwrite with additional configdata
        if configdata is not None:
            self._set_config(configdata)
        
    def _as_dict(self, cfg_parser):
        ''' Convert a ConfigParser object into a dictionary '''
        the_dict = {}
        for section in cfg_parser.sections():
            the_dict[section] = {}
            for key, val in cfg_parser.items(section):
                the_dict[section][key] = val
        return the_dict
    
    def read_config_file(self, configfile):
        ''' Read a config file into self._config '''
        if not os.path.exists(configfile):
            raise Exception('Config file not found: %s' % configfile)
        cfg_parser = configparser.ConfigParser()
        cfg_parser.read(configfile)
        self._config = self._as_dict(cfg_parser)
    
    def set_config(self, configdata):
        ''' Set multiple config items from configdata dict '''
        for section in configdata.keys():
            for key,val in configdata[section].items():
                self.set_config_item(section, key, val)           
    
    def set_config_item(self, section, key, value):
        ''' Set a config item to value '''
        if not section in self._config.keys():
            raise Exception('Section not defined in .ini: [%s]' % section)
        if not key in self._config[section].keys():
            raise Exception('Item not defined in .ini: [%s] %s' % (section, key))  
        self._config[section][key] = value
    
    def write_config(self, configfile=None):
        ''' Write the current config to configfile '''
        if configfile is None:
            configfile = self.__class__.__name__ + '.ini'
        
        cfg_parser= configparser.ConfigParser()
        cfg_parser.read_dict(self._config)
        
        with open(configfile, 'w') as fid:
            cfg_parser.write(fid)
        
    def dump_config(self):
        ''' Dump the current config to stdout '''
        for section in self._config.keys():
            print('[%s]' % section)
            for key, val in self._config[section].items():
                print('%s = %s' % (key, str(val)))
                