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
import cfobs.cfobs_load as cfobs_load


def obs_load(obsfile,startdate,**kwargs):
    '''Load the observation file.'''
    log = logging.getLogger(__name__)
    if not os.path.isfile(obsfile):
        log.error('File does not exist: {}'.format(obsfile),exc_info=True)
        return None
    startdate = dt.datetime(2018,1,1) if startdate is None else startday
    obs = cfobs_load.load(file_template=obsfile,startday=startdate,**kwargs)
    return obs 
