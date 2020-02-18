#!/usr/bin/env python
# ****************************************************************************
# obs.py 
#
# DESCRIPTION: 
# Handle observation data 
# 
# HISTORY:
# 20200130 - christoph.a.keller at nasa.gov - Initial version
# ****************************************************************************
import os
import pandas
import logging
import datetime as dt
from cfobs.cfobs_load import load as cfobs_load


def obs_load(config,obsfile=None,location=None,read_all=False,**kwargs):
    '''Load the observation file'''
    log = logging.getLogger(__name__)
#---Observation file
    if obsfile is None:
        obsfile = config.get('observations').get('obsfile')
#---Eventually specify locations filter
    if read_all:
        locations = None
    else:
        locs = config.get('locations')
        if locs is not None:
            if location is not None:
                locations = [locs.get(location).get('name_in_obsfile',location)]
            else:
                locations = [locs.get(k).get('name_in_obsfile',k) for k in locs]
        else:
            log.warning('No locations specified in configuration file - will read full file')
            locations = [location]
        obsfile = obsfile.replace('%n',locations[0])
#---Read observations file
    if locations is not None:
        obs_location_key = config.get('observations').get('obs_location_key','original_station_name')
        locations_filter = {obs_location_key: locations}
    obs = cfobs_load(file_template=obsfile,filter=locations_filter,**kwargs)
    if obs.shape[0] == 0:
        log.error('File does not exist: {}'.format(obsfile),exc_info=True)
        return None
    return obs 
