#!/usr/bin/env python
# ****************************************************************************
# prepare_data.py 
#
# HISTORY:
# 20200203 - christoph.a.keller at nasa.gov - Initial version
# ****************************************************************************
import os
import pandas
import logging
import datetime as dt
import numpy as np
from sklearn.model_selection import train_test_split

import cfobs.units as cfobs_units

from .configfile import get_location_info
from .configfile import get_species_info


# PARAMETER
ROUNDING_PRECISION = 4


def prepare_training_data(mod,obs,config,location,species=None,check_latlon=False,mod_drop=['location','lat','lon'],round_minutes=True,**kwargs):
    '''Prepare data for ML training.'''
    log = logging.getLogger(__name__)
#---location settings
    obs_location_key = config.get('observations').get('obs_location_key','original_station_name')
    location_key, location_name_in_obsfile, location_lat, location_lon, region_name = get_location_info(config,location)
#---(target) species settings
    species_key, species_name_in_obsfile, species_mw, prediction_type, prediction_unit = get_species_info(config,species)
#---verbose
    log.info('Prepare data for ML training - settings:')
    log.info('--> Location: {} ({:} degN, {:} degE; name in obsfile: {})'.format(location_key,location_lat,location_lon,location_name_in_obsfile))
    log.info('--> Species: {}; Species name in obsfile: {}; MW: {}; Prediction unit: {}; Prediction type: {}'.format(species_key,species_name_in_obsfile,species_mw,prediction_unit,prediction_type))
#---observation data
    obs_reduced = obs.loc[(obs['obstype']==species_name_in_obsfile) & (obs['value']>=0.0) & (~np.isnan(obs['value'])) & (obs[obs_location_key]==location_name_in_obsfile)].copy()
    if check_latlon:
        laterr,lonerr = _latlon_check(obs_reduced,location_lat,location_lon)
        if laterr or lonerr:
            log.error('At least one latitude or longitude mismatch in observation data: {:}/{:}'.format(np.round(location_lat,ROUNDING_PRECISION),np.round(location_lon,ROUNDING_PRECISION)),exc_info=True)
            return None,None,None,None
    if round_minutes:
        obs_reduced['ISO8601'] = [dt.datetime(i.year,i.month,i.day,i.hour,0,0) for i in obs_reduced['ISO8601']]
    log.debug('Shape of obs_reduced: {}'.format(obs_reduced.shape))
#---model data
    if 'ISO8601' in mod_drop:
        mod_drop.remove('ISO8601')
    if 'temp_for_unit' in mod_drop:
        mod_drop.remove('temp_for_unit')
    if 'press_for_unit' in mod_drop:
        mod_drop.remove('press_for_unit')
    mod_reduced = prepare_prediction_data(mod,config,location=None,location_name=location_key,
                  location_lat=location_lat,location_lon=location_lon,check_latlon=check_latlon,
                  drop=mod_drop,round_minutes=round_minutes)
    log.debug('Shape of mod_reduced: {}'.format(mod_reduced.shape))
#---reduce to overlapping dates
    dates = list(set(mod_reduced['ISO8601']).intersection(obs_reduced['ISO8601']))
    if len(dates)==0:
        log.warning('No overlap found between obs & mod!')
        return None,None,None,None
    obs_reduced = obs_reduced.loc[obs_reduced['ISO8601'].isin(dates)].copy().sort_values(by='ISO8601')
    mod_reduced = mod_reduced.loc[mod_reduced['ISO8601'].isin(dates)].copy().sort_values(by='ISO8601')
#---convert units if needed
    if prediction_unit == 'ppbv':
        obs_reduced = _convert2ppbv(obs_reduced,mod_reduced,species_mw)
    obs_reduced = obs_reduced.groupby('ISO8601').sum().reset_index()
    log.debug('After grouping: {}'.format(np.mean(obs_reduced['value'])))
#---extract prediction values, convert to bias if needed
    if prediction_type == 'bias':
        obs_reduced['value'] = np.array(obs_reduced['value'].values) - np.array(mod_reduced[species_key].values)
        log.debug('After calculating bias: {}'.format(np.mean(obs_reduced['value'])))
#---split data
    predicted_values = np.array(obs_reduced['value'].values)
    if 'ISO8601' in mod_reduced.keys():
        _ = mod_reduced.pop('ISO8601')
    if 'press_for_unit' in mod_reduced.keys():
        _ = mod_reduced.pop('press_for_unit')
    if 'temp_for_unit' in mod_reduced.keys():
        _ = mod_reduced.pop('temp_for_unit')
    Xtrain,Xvalid,Ytrain,Yvalid = train_test_split( mod_reduced, predicted_values, **kwargs )
    return Xtrain, Xvalid, Ytrain, Yvalid


def prepare_prediction_data(mod,config,location=None,location_name=None,location_lat=None,
                            location_lon=None,check_latlon=False,drop=['location','lat','lon','press_for_unit','temp_for_unit'],
                            round_minutes=True):
    '''Prepare model data for model prediction'''
    log = logging.getLogger(__name__)
#---location settings
    if location_name is None:
        location_name, location_name_in_obsfile, location_lat, location_lot, region_name = get_location_info(config,location)
#---prepare model data
    mod_reduced = mod.loc[mod['location']==location_name].copy()
    if check_latlon:
        laterr,lonerr = _latlon_check(mod_reduced,location_lat,location_lon)
        if laterr or lonerr:
            log.error('At least one latitude or longitude mismatch in model data: {:}/{:}'.format(np.round(location_lat,ROUNDING_PRECISION),np.round(location_lon,ROUNDING_PRECISION)),exc_info=True)
            return None
    if round_minutes:
        mod_reduced['ISO8601'] = [dt.datetime(i.year,i.month,i.day,i.hour,0,0) for i in mod_reduced['ISO8601']]
    mod_reduced = mod_reduced.groupby('ISO8601').sum().reset_index()
    mod_reduced['Hour'] = [i.hour for i in mod_reduced['ISO8601']]
#---eventually drop values
    _ = [mod_reduced.pop(var) for var in drop if var in mod_reduced.keys()]
    return mod_reduced 


def _convert2ppbv(obs,mod,mw,temp_name='temp_for_unit',ps_name='press_for_unit'):
    '''Convert observations to ppbv'''
    log = logging.getLogger(__name__)
    log.debug('Before unit conversion: {}'.format(np.mean(obs['value'].values)))
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
        imod  = mod.loc[mod['ISO8601'].isin(dates),['ISO8601',temp_name,ps_name]].sort_values(by='ISO8601')
        conv = cfobs_units.get_conv_ugm3_to_ppbv(imod,temperature_name=temp_name,pressure_name=ps_name,mw=mw)
        obs.loc[idx,'value'] = np.array(obs.loc[idx,'value'].values*conv)
        log.debug('Converted ugm-3 to ppbv for {:} values'.format(len(idx)))
#---ppmv to ppbv:
    idx = obs.index[obs['unit']=='ppmv']
    if len(idx)>0:
        obs.loc[idx,'value'] = np.array(obs.loc[idx,'value'].values*cfobs_units.PPM2PPB)
        log.debug('Converted ppmv to ppbv for {:} values'.format(len(idx)))
    log.debug('After unit conversion: {}'.format(np.mean(obs['value'].values)))
    return obs


def _latlon_check(dat,lat,lon):
    '''Check if all latitudes and longitudes in data frame have specified lat&lon values'''
    laterr = any(np.round(dat['lat'].values,ROUNDING_PRECISION)!=np.round(lat,ROUNDING_PRECISION))
    lonerr = any(np.round(dat['lon'].values,ROUNDING_PRECISION)!=np.round(lon,ROUNDING_PRECISION))
    return laterr,lonerr 

