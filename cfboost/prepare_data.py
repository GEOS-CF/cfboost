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
import random
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import cfobs.units as cfobs_units

from .configfile import get_location_info
from .configfile import get_species_info


# PARAMETER
ROUNDING_PRECISION = 4


def prepare_training_data(mod,obs,config,location,species=None,check_latlon=False,mod_drop=['location','lat','lon'],round_minutes=True,trendday=None,minval=0.01,outliers_sigma=None,minobs=None,mindate=None,maxdate=None,nsplit=None,isplit=None,**kwargs):
    '''Prepare data for ML training.'''
    log = logging.getLogger(__name__)
#---location settings
    obs_location_key = config.get('observations').get('obs_location_key','original_station_name')
    location_key, location_name_in_obsfile, location_lat, location_lon, region_name = get_location_info(config,location)
#---(target) species settings
    species_key, species_name_in_obsfile, species_mw, prediction_type, prediction_unit, transform, offset = get_species_info(config,species)
#---verbose
    log.info('Prepare data for ML training - settings:')
    log.info('--> Location: {} ({:} degN, {:} degE; name in obsfile: {})'.format(location_key,location_lat,location_lon,location_name_in_obsfile))
    log.info('--> Species: {}; Species name in obsfile: {}; MW: {}; Prediction unit: {}'.format(species_key,species_name_in_obsfile,species_mw,prediction_unit))
    log.info('--> Prediction: {}; Transform: {}; Offset: {:.2}'.format(prediction_type,transform,offset if offset is not None else 0.0))
    if trendday is not None:
        log.info(trendday.strftime('Will use trendday starting at %Y-%m-%d'))
#---observation data
    obs_reduced = obs.loc[(obs['obstype']==species_name_in_obsfile) & (obs['value']>=minval) & (~np.isnan(obs['value'])) & (obs[obs_location_key]==location_name_in_obsfile)].copy()
#---restrict to range of dates if specified
    if mindate is not None:
         obs_reduced = obs_reduced.loc[obs_reduced['ISO8601']>=mindate].copy()
    if maxdate is not None:
         obs_reduced = obs_reduced.loc[obs_reduced['ISO8601']<maxdate].copy()
    nobs = obs_reduced.shape[0]
    if minobs is not None:
        if nobs < minobs:
            log.warning('Not enough observations found: {} vs {}'.format(nobs,minobs))
            return None,None,None,None
    if outliers_sigma is not None and nobs>0:
        avg = obs_reduced['value'].values.mean()
        std = obs_reduced['value'].values.std()
        minval = avg - outliers_sigma*std
        maxval = avg + outliers_sigma*std
        nbefore = obs_reduced.shape[0]
        obs_reduced = obs_reduced.loc[(obs_reduced['value']>=minval)&(obs_reduced['value']<=maxval)].copy()
        nafter  = obs_reduced.shape[0]
        log.info('Removed {} outliers ({:.2f}%)'.format(nbefore-nafter,(nbefore-nafter)/nbefore*100.0))
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
                  drop=mod_drop,round_minutes=round_minutes,trendday=trendday)
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
    #obs_reduced = obs_reduced.groupby('ISO8601').sum().reset_index()
    obs_reduced = obs_reduced.groupby('ISO8601').mean().reset_index()
    log.debug('After grouping: {}'.format(np.mean(obs_reduced['value'])))
#---extract prediction values, convert to bias if needed
    if prediction_type == 'bias':
        mod = np.array(mod_reduced[species_key].values)
        obs = np.array(obs_reduced['value'].values) 
        # log transform: convert concentrations to log before doing bias correction:
        # log(obs)-log(mod) = log(obs/mod).
        if transform == 'log':
            if minval is not None:
                mod[mod<=0.0] = minval
            #mod = np.where(mod>0.0,np.log(mod),0.0)
            #obs = np.where(obs>0.0,np.log(obs),0.0)
            mod[mod>0.0] = np.log(mod[mod>0.0])
            obs[obs>0.0] = np.log(obs[obs>0.0])
        obs_reduced['value'] = obs - mod
        log.debug('After calculating bias: {}'.format(np.mean(obs_reduced['value'])))
#---split data
    if 'press_for_unit' in mod_reduced.keys():
        _ = mod_reduced.pop('press_for_unit')
    if 'temp_for_unit' in mod_reduced.keys():
        _ = mod_reduced.pop('temp_for_unit')
    if 'ISO8601' in mod_reduced.keys():
        _ = mod_reduced.pop('ISO8601')
    predicted_values = np.array(obs_reduced['value'].values)
    # split in set chunks
    if nsplit is not None:
        mod_split = np.array_split(mod_reduced,nsplit)
        obs_split = np.array_split(predicted_values,nsplit)
        ii = isplit if isplit is not None else random.choice(np.arange(nsplit))
        Xvalid = mod_split.pop(ii)
        Yvalid = obs_split.pop(ii)
        Xtrain = pd.concat(mod_split)
        Ytrain = np.concatenate(obs_split)
    # split randomly
    else:
        Xtrain,Xvalid,Ytrain,Yvalid = train_test_split( mod_reduced, predicted_values, **kwargs )
    return Xtrain, Xvalid, Ytrain, Yvalid


def prepare_prediction_data(mod,config,location=None,location_name=None,location_lat=None,
                            location_lon=None,check_latlon=False,drop=['location','lat','lon','press_for_unit','temp_for_unit'],
                            round_minutes=True,trendday=None,group=True):
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
    if group:
        #mod_reduced = mod_reduced.groupby('ISO8601').sum().reset_index()
        mod_reduced = mod_reduced.groupby('ISO8601').mean().reset_index()
    mod_reduced['Hour'] = [i.hour for i in mod_reduced['ISO8601']]
    mod_reduced['Weekday'] = [i.weekday() for i in mod_reduced['ISO8601']]
    mod_reduced['Month'] = [i.month for i in mod_reduced['ISO8601']]
    if trendday is not None:
        mod_reduced['Trendday'] = [(i-trendday).days for i in mod_reduced['ISO8601']]
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
        dates = obs.loc[idx,'ISO8601'].unique()
        imod  = mod.loc[mod['ISO8601'].isin(dates),['ISO8601',temp_name,ps_name]].sort_values(by='ISO8601').copy()
        imod['conv'] = cfobs_units.get_conv_ugm3_to_ppbv(imod,temperature_name=temp_name,pressure_name=ps_name,mw=mw)
        tmp = pd.DataFrame()
        tmp['ISO8601'] = obs.loc[idx,'ISO8601']
        tmp = tmp.merge(imod)
        obs.loc[idx,'value'] = np.array(obs.loc[idx,'value'].values*tmp['conv'].values)
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

