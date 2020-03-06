#!/usr/bin/env python
# ****************************************************************************
# model.py 
#
# DESCRIPTION: 
# Handle model data 
# 
# HISTORY:
# 20200130 - christoph.a.keller at nasa.gov - Initial version
# ****************************************************************************
import os
import pandas
import logging
import datetime as dt
import pandas as pd

from cfobs.cfobs_load import load as cfobs_load

from .cf2csv import cf2csv
from .tools  import filename_parse


def model_load(config_cf,location='',error_if_missing=True,**kwargs):
    '''Load the model data.'''
    log = logging.getLogger(__name__)
#---Load model data
    ofile_template = config_cf.get('ofile_template')
    if ofile_template is None:
        ofile_template = 'cf_%l.csv'
        log.warning('No template for output file found - will use default {}'.format(ofile_template))
    modfile = filename_parse(ofile_template,loc=location)
    mod = cfobs_load(file_template=modfile,**kwargs)
    if mod.shape[0] == 0:
        if error_if_missing:
            log.error('File does not exist: {}'.format(modfile),exc_info=True)
#---Reduce to variables & apply scale factors
    else:
        mod = _check_vars(mod,config_cf)
    return mod


def model_read(config_cf,config_loc,startday,locations=None,**kwargs):
    '''Read model data from netCDF source'''
    log = logging.getLogger(__name__)
#---Locations to be read.
    if locations is not None:
        locs = dict()
        for l in locations:
            locs[l] = config_loc.get(l)
    else:
        locs = config_loc.copy()
    dat = cf2csv(config_cf,config_loc,startday,**kwargs)
    if dat is not None:
        for i in dat:
            if dat[i] is not None: 
                dat[i] = _check_vars(dat[i],config_cf)
    return dat


def _check_vars(mod,config_cf):
    '''Make sure all variables in config_cf are available, and applies specified scale factors to each variable'''
    log = logging.getLogger(__name__)
    mod_out = pd.DataFrame()
    # pass 'standard' fields
    for var in ['ISO8601','location','lat','lon']:
        if var not in mod.keys():
            log.error('Variable not found in CF file: {}'.format(var),exc_info=True)
            return None
        mod_out[var] = mod[var].values
    # pass model variables
    collections = config_cf.get('collections')
    for col in collections.keys():
        skip = collections.get(col).get('skip_for_ml',False)
        if skip:
            log.debug('Skip collection {}'.format(col))
            continue
        vars = collections.get(col).get('vars')
        for var in vars:
            skip = vars.get(var).get('skip_for_ml',False)
            if skip:
                log.debug('Skip variable {}'.format(var))
                continue
            ivar = vars.get(var).get('name_in_file',var) 
            scal = vars.get(var).get('scal',1.0)
            if ivar not in mod.keys():
                log.error('Variable not found in CF file: {}'.format(ivar),exc_info=True)
                return None
            log.debug('Read variable {} into field {}, use scale factor {}'.format(ivar,var,scal))
            mod_out[var] = mod[ivar].values*scal
    del mod
    return mod_out

