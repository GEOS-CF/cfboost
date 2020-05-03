#!/usr/bin/env python
# ****************************************************************************
# configfile.py 
#
# DESCRIPTION: 
# Handle configuration file 
# 
# HISTORY:
# 20200130 - christoph.a.keller at nasa.gov - Initial version
# ****************************************************************************
import os
import yaml
import logging
import numpy as np

DEFAULT_TYPE = 'bias'

def load_config(configfile):
    '''Load the configuration file.'''
    log = logging.getLogger(__name__)
    if not os.path.isfile(configfile):
        log.error('CFBoost configuration file does not exist: {}'.format(configfile),exc_info=True)
        return None
    with open(configfile,'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_locations_and_species(config):
    '''Return a list with all location and species names in the configuration file'''
    locs  = list(config.get('locations').keys())
    specs = list(config.get('species').keys())
    return locs,specs


def get_location_info(config,location_key=None):
    '''Return the location info for the given location key'''
    location_key = location_key if location_key is not None else list(config.get('locations').keys())[0]
    location = config.get('locations').get(location_key)
    name = location.get('name_in_obsfile',location_key)
    lat  = location.get('lat',np.nan)
    lon  = location.get('lon',np.nan)
    region = location.get('region_name',location_key)
    return location_key,name,lat,lon,region


def get_species_info(config,species):
    '''Return species info'''
    species_key = species if species is not None else list(config.get('species').keys())[0]
    specs = config.get('species').get(species_key)
    name_in_obsfile = specs.get('name_in_obsfile',species_key)
    species_mw   = specs.get('MW',np.nan)
    prediction_type = specs.get('prediction_type',DEFAULT_TYPE)
    prediction_unit = 'ugm-3' if 'pm25' in species else 'ppbv'
    transform = specs.get('transform','N/A')
    offset = specs.get('offset',None)
    return species_key,name_in_obsfile,species_mw,prediction_type,prediction_unit,transform,offset
