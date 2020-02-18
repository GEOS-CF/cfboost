#!/usr/bin/env python
# ****************************************************************************
# cf2csv.py
# 
# HISTORY:
# 20190424 - christoph.a.keller at nasa.gov - Initial version 
# ****************************************************************************

import logging
import argparse
import sys
import time
import numpy as np
import datetime as dt
from calendar import monthrange
import os
import xarray as xr
import pandas as pd
import yaml
from multiprocessing.pool import ThreadPool
import dask

from .csv_table import write_csv

def cf2csv(config_cf,config_loc,startday,endday=None,duration=None,forecast=False,read_freq='1D',error_if_not_found=True,
           resample=None,batch_write=False,write_data=True,append=False,return_data=False,**kwargs):
    '''
    Opens hourly CF files (one day at a time), reads selected variables 
    at defined locations, and writes these data into individual csv files.
    The files and variables as well as the locations to be read are defined in two
    YAML files (passed through the argument list). 
    Supports reading files locally (e.g., on Discover) or remotely via OpenDAP.

    p.add_argument('-f','--files',type=str,help='yaml file with file information',default='config/cf2csv_files_opendap.yml')
    p.add_argument('-l','--locs',type=str,help='yaml file with location information',default='config/cf2csv_locations.yml')
    p.add_argument('-o','--ofile',type=str,help='output file. Use token %s for station name',default='csv/GEOS-CF.v01.rpl.tavg_1hr_%s.csv')
    p.add_argument('-a','--append',type=int,help='append to existing file (0=no, 1=yes)',default=0)
    p.add_argument('-y1','--year1',type=int,help='start year',default=2018) 
    p.add_argument('-y2','--year2',type=int,help='end year',default=None) 
    p.add_argument('-m1','--month1',type=int,help='start month', default=1) 
    p.add_argument('-m2','--month2',type=int,help='end month', default=None) 
    p.add_argument('-d1','--day1',type=int,help='start day', default=1) 
    p.add_argument('-d2','--day2',type=int,help='end day', default=None)
    p.add_argument('-r','--resample',type=str,help='resample method, will be used with pandas.resample(XXX). For instance, use `D` for daily averages', default=None)
    p.add_argument('-e','--error-if-not-found',type=int,help='raise error and stop script if file not found (1=yes, 0=no)', default=0)
    p.add_argument('-rf','--read-freq',type=str,help='frequency string for sifting through the files. For example, if using `1D` the files will be read day by day',default='1D')
    p.add_argument('-b','--batch-write',type=int,help='if set to 1, will write out everything at the end. Otherwise, data will be written out continuously',default=0)
    '''
    log = logging.getLogger(__name__)
    dask.config.set(pool=ThreadPool(10))
#---Setup
    ofile_template = config_cf.get('ofile_template')
    if ofile_template is None:
        ofile_template = 'cf_%l.csv'
        log.warning('No template for output file found - will use default {}'.format(ofile_template))
    opened_files = []
    # for convenience, prestore list of collections and variables to be read for every collection
    templ_key = 'template_forecast' if forecast else 'template'
    readcols = dict()
    readvars = dict()
    collections = config_cf.get('collections')
    for icol in collections.keys():
        templ = collections.get(icol).get(templ_key)
        if templ is None:
            log.error('Template for collection {} not found, was looking for template key {}'.format(icol,templ_key),exc_info=True)
            return None
        if templ == 'skip' :
            log.debug('Will skip collection {} because template key {} set to value "skip"'.format(icol,templ_key))
            continue
        var_list = []
        vars = collections.get(icol).get('vars')
        for ivar in vars.keys():
            tvar = vars.get(ivar).get('name_in_file',ivar)
            if tvar not in var_list:
                var_list.append(tvar)
        readvars[icol] = var_list
        readcols[icol] = templ
    # also create lists of location names, latitudes and longitudes
    locs=[]; lons=[]; lats=[]
    for iloc in config_loc.keys():
        locs.append(iloc)
        lons.append(config_loc.get(iloc).get('lon'))
        lats.append(config_loc.get(iloc).get('lat'))
    # create (empty) data frame for every location. Will be filled with data below
    if batch_write or return_data:
        df_loc = dict()
        for iloc in config_loc.keys():
            df_loc[iloc] = pd.DataFrame()
    else:
        df_loc = None
#---Select timestamps to be read
    datelist = []
    if duration is not None:
        endday = startday + dt.timedelta(hours=duration)
    if endday is not None:
        datelist = pd.date_range(start=startday,end=endday,freq=read_freq).tolist()
    if len(datelist)<2:
        datelist = [startday,None]
    log.debug('Read dates: {}'.format(datelist))
    # By default we read all hourly files: 
    hrtoken = '*'
#---Read data for each date
    t1 = time.time()
    for i in range(len(datelist)-1):
        idate = datelist[i]
        jdate = datelist[i+1]
        if 'H' in read_freq:
            hrtoken = str(idate.hour).zfill(2)
        log.info('working on '+idate.strftime('%Y-%m-%d %H:%M'))
        #---Load collections for this day 
        dslist = _load_files(readcols,idate,jdate,hrtoken,error_if_not_found)
        #---Read data for every location
        dfs = _sample_files(dslist,readvars,locs,lats,lons,resample)      
        for iloc in locs:
            if batch_write or return_data:
                df_loc[iloc] = df_loc[iloc].append(dfs.get(iloc).copy(),sort=True)
            if write_data and not batch_write:
                write_csv(dfs.get(iloc),ofile_template,opened_files,idate,iloc,append,**kwargs)
        #---Close all files
        for l in dslist:
            if dslist[l] is not None:
                fid = dslist.get(l)
                fid.close()
    # Write out files
    if write_data and batch_write:
        idate = datelist[0] + ( datelist[-1]-datelist[0] ) / 2
        for iloc in locs:
            opened_files = write_csv(df_loc[iloc],ofile_template,opened_files,idate,iloc,append,**kwargs)
    # All done
    t2 = time.time()
    log.info('This took {:.4f} seconds'.format(t2-t1))
    return df_loc 


def _load_files(readcols,idate,jdate=None,hrtoken='*',error_if_not_found=False):
    '''Loads all files into memory for a given date.'''
    log = logging.getLogger(__name__)
    dslist = dict() 
    # if reading forecasts, make sure we use the correct collection.
    for icol in readcols:
        templ = readcols.get(icol)
        templ = idate.strftime(templ.replace('%c',icol))
        log.info('Reading {}'.format(templ))
        if 'opendap.nccs.nasa.gov' in templ:
            try: 
                ds = xr.open_dataset(templ)
                if jdate is not None:
                    ds = ds.sel(time=slice(idate,jdate))
            except:
                if error_if_not_found:
                    log.error('Could not read {}'.format(templ),exc_info=True)
                else:
                    log.warning('Error reading file - will will with NaNs: {}'.format(templ))
                    ds = None
        else:
            try: 
                ds = xr.open_mfdataset(templ) 
            except:
                if error_if_not_found:
                    log.error('Could not read {}'.format(templ),exc_info=True)
                else:
                    log.warning('Error reading file - will will with NaNs: {}'.format(templ))
                    ds = None
        dslist[icol] = ds
    return dslist


def _sample_files(dslist,readvars,locs,lats,lons,resample):
    '''
    Sample the previously opened files (--> dslist) at location ilat, ilon. 
    '''
    log = logging.getLogger(__name__)
    # Create output data frame, set 'meta data' 
    df_empty = pd.DataFrame()
    for icol in dslist:
        if dslist.get(icol) is None:
            continue
        else:
            break
    df_empty['ISO8601'] = dslist.get(icol).time.values
    nval = df_empty.shape[0]
    dflist = dict()
    for l in locs:
        dflist[l] = df_empty.copy() 
    # Loop over all collections 
    for icol in readvars:
        log.info('Reading collection {}...'.format(icol))
        vars = readvars.get(icol)
        idflist = dict()
        for l in locs:
            idflist[l] = pd.DataFrame()
        # Fill with NaN's if collection not found 
        ids = dslist.get(icol)
        if ids is None:
            for l in locs:
                for v in vars:
                    dflist[l][v] = np.zeros((nval,))*np.nan
            continue
        # Error check: make sure all collections have same # of time stamps
        if ids.time.shape[0] != nval:
            log.warning('Warning: not the same number of time stamps - some values will be filled with NaNs: {}'.format(icol))
        # Set time stamp
        for l in locs:
            idflist[l]['ISO8601'] = ids.time.values
        if 'lev' in ids.dims:
            nlev = len(ids.lev)
        else:
            nlev = 0
        # Get variable
        for v in vars:
            for iloc,ilat,ilon in zip(locs,lats,lons):
                if 'lev' in ids[v].dims:
                    idflist[iloc][v] = ids[v].sel(lat=ilat,lon=ilon,lev=nlev-1,method='nearest').values
                else:
                    idflist[iloc][v] = ids[v].sel(lat=ilat,lon=ilon,method='nearest').values
        # Merge into main frame
        for l in locs:
            dflist[l] = pd.merge(dflist[l],idflist[l],on='ISO8601',how='outer')
    # post-processcing
    for iloc,ilat,ilon in zip(locs,lats,lons):
        df = dflist.get(iloc)
        # eventually resample data 
        if resample is not None:
            df.index = df['ISO8601']
            df = df.resample(resample).mean().reset_index()
        # set 'meta data'
        nrow           = df.shape[0]
        df['location'] = [iloc for x in range(nrow)]
        df['lat']      = [ilat for x in range(nrow)]
        df['lon']      = [ilon for x in range(nrow)]
        df['year']     = [i.year for i in df['ISO8601']]
        df['month']    = [i.month for i in df['ISO8601']]
        df['day']      = [i.day for i in df['ISO8601']]
        df['hour']     = [i.hour for i in df['ISO8601']]
        # add this to be safe, not sure it is needed:
        dflist[iloc] = df
    return dflist
