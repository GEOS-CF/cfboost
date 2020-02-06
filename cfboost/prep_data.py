#!/usr/bin/env python
# ****************************************************************************
# prep_data.py 
#
# HISTORY:
# 20200203 - christoph.a.keller at nasa.gov - Initial version
# ****************************************************************************
import os
import pandas
import logging
import datetime as dt
import numpy as np
import cfobs.units as cfobs_units
from sklearn.model_selection import train_test_split

from .config import get_location_info
from .config import get_species_info


# PARAMETER
ROUNDING_PRECISION = 4
DEFAULT_TYPE = 'tend'


def prepare_training_data(mod,obs,config,location,species=None,check_latlon=False, 
                          drop=['location','lat','lon'],**kwargs):
    '''Prepare data for ML training.'''
    log = logging.getLogger(__name__)
#---location settings
    location_name, location_lat, location_lon = get_location_info(config,location)
#---(target) species settings
    species_name, species_mw, prediction_type, prediction_unit = get_species_info(config,species)
#---verbose
    log.info('Prepare data for ML training - settings:')
    log.info('--> Location: {} ({:} degN, {:} degE)'.format(location_name,location_lat,location_lon))
    log.info('--> Species: {}; MW: {}; Prediction unit: {}; Prediction type: {}'.format(species_name,species_mw,prediction_unit,prediction_type))
#---observation data
    obs_reduced = obs.loc[(obs['obstype']==species_name) & (obs['value']>=0.0) & (obs['location']==location_name)].copy()
    if check_latlon:
        laterr,lonerr = _latlon_check(obs_reduced,location_lat,location_lon)
        if laterr or lonerr:
            log.error('At least one latitude or longitude mismatch in observation data: {:}/{:}'.format(np.round(location_lat,ROUNDING_PRECISION),np.round(location_lon,ROUNDING_PRECISION)),exc_info=True)
            return None,None,None,None
#---model data
    if 'ISO8601' in drop:
        drop.remove('ISO8601')
    mod_reduced = prepare_prediction_data(mod,config,location=None,
                   location_name=location_name,location_lat=location_lat,
                   location_lon=lon,check_latlon=check_latlon,drop=drop)
#---reduce to overlapping dates
    dates = list(set(mod_reduced['ISO8601']).intersection(obs_reduced['ISO8601']))
    if len(dates)==0:
        log.error('No overlap found between obs & mod!',exc_info=True)
        return None,None,None,None
    obs_reduced = obs_reduced.loc[obs_reduced['ISO8601'].isin(dates)].copy()
    obs_reduced = obs_reduced.sort_values(by='ISO8601')
    mod_reduced = mod_reduced.loc[mod_reduced['ISO8601'].isin(dates)].copy()
    mod_reduced = mod_reduced.sort_values(by='ISO8601')
#---convert units if needed
    if prediction_unit == 'ppbv':
        obs_reduced = _convert2ppbv(obs_reduced,mod_reduced,species_mw)
#---group by dates
    obs_reduced = obs_reduced.groupby('ISO8601').sum().reset_index()
#---extract prediction values, convert to tendency if needed
    if prediction_type == 'tend':
        obs_reduced['value'] = obs_reduced['value'].values - mod_reduced[species_name].values
#---split data
    obs_reduced = obs_reduced.pop('value')
    _ = mod_reduced.pop('ISO8601')
    Xtrain,Xvalid,Ytrain,Yvalid = train_test_split( mod_reduced, obs_reduced, **kwargs )
    return Xtrain,Xvalid,Ytrain,Yvalid


def prepare_prediction_data(mod,config,location=None,location_name=None,location_lat=None,
                            location_lon=None,check_latlon=False,drop=['location','lat','lon']):
    '''Prepare model data for model prediction'''
    log = logging.getLogger(__name__)
#---location settings
    if location_name is None:
        location_name, location_lat, location_lot = get_location_info(config,location)
#---prepare model data
    mod_reduced = mod.loc[mod['location']==location_name].copy()
    if check_latlon:
        laterr,lonerr = _latlon_check(mod_reduced,location_lat,location_lon)
        if laterr or lonerr:
            log.error('At least one latitude or longitude mismatch in model data: {:}/{:}'.format(np.round(location_lat,ROUNDING_PRECISION),np.round(location_lon,ROUNDING_PRECISION)),exc_info=True)
            return None
    mod_reduced = mod_reduced.groupby('ISO8601').sum().reset_index()
    mod_reduced['Hour'] = [i.hour for i in mod_reduced['ISO8601']]
#---eventually drop values
    _ = [mod_reduced.pop(var) for var in drop if var in mod_reduced.keys()]
    return mod_reduced 


def _convert2ppbv(obs,mod,mw,temp_name='t10m',ps_name='ps'):
    '''Convert observations to ppbv'''
    log = logging.getLogger(__name__)
#---ugm-3 to ppbv:
    idx = obs.index[obs['unit']=='ugm-3']
    if len(idx)>0:
        if temp_name not in mod:
            log.error('Temperature not found in model: {}'.format(temp_name),exc_info=True)
            return None
        if ps_name not in mod:
            log.error('Surface pressure not found in model: {}'.format(ps_name),exc_info=True)
            return None
        # Calculate conversion factor for selected dates
        dates = obs.loc[idx,'ISO8601']
        imod  = mod.loc[mod['ISO8601'].isin(dates),[temp_name,ps_name]]
        imod  = imod.sort_values(by='ISO8601')
        conv = cfobs_units.get_conv_ugm3_to_ppbv(imod,temperature_name=temp_name,pressure_name=ps_name,mw=mw)
        obs.loc[idx,'value'] = obs.loc[idx,'value'].values*conv
        log.debug('Converted ugm-3 to ppbv for {:} values'.format(len(idx)))
#---ppmv to ppbv:
    idx = obs.index[obs['unit']=='ppmv']
    if len(idx)>0:
        obs.loc[idx,'value'] = obs.loc[idx,'value'].values*cfobs_units.PPM2PPB
        log.debug('Converted ppmv to ppbv for {:} values'.format(len(idx)))
    return obs


def _latlon_check(dat,lat,lon):
    '''Check if all latitudes and longitudes in data frame have specified lat&lon values'''
    laterr = any(np.round(dat['lat'].values,ROUNDING_PRECISION)!=np.round(lat,ROUNDING_PRECISION))
    lonerr = any(np.round(dat['lon'].values,ROUNDING_PRECISION)!=np.round(lon,ROUNDING_PRECISION))
    return laterr,lonerr 

