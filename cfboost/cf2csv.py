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
           resample=None,write_data=True,batch_write=False,batch_size=-999,append=False,return_data=False,**kwargs):
    '''
    Open GEOS-CF data, read selected variables at given locations, and write them to a pandas data frame.
    The file and variable information, as well as the location information, must be provided through dictionaries.
    This routine supports reading files locally (e.g., directly from local netCDF files) or remotely via OpenDAP.

    Arguments
    ---------
    config_cf: dict
        model configuration (model output collections and variables to be read).
    config_loc: dict
        location configuration (locations to be sampled).
    startday: dt.datetime
        sampling start day.
    endday: dt.datetime
        sampling end day.
    duration: 
        sampling duration, in hours from startday. If specified, overwrites endday.
    forecast: bool
        read forecast collections?
    read_freq: str
        frequency string for sifting through the data. For example, if using '1D' the files will be processed in batches of one day.
    error_if_not_found: bool
        raise error and stop script if file not found.
    resample: str
        resampling method, to be used with pandas.resample(). For instance, use 'D' to generate daily averages.
    write_data: bool
        write data to csv file?
    batch_write: bool
        if true, will write out the entire data at the end. Otherwise, the data will be written out as the batches are being read.
    batch_size: int 
        if set to positive value, will write batches after that many tiem updates. If negative, will write at the end. 
    append: bool
        append data that is written out to an existing file?
    return_data: bool
        if True, the batch data will be accumulated in a 'master' data frame that is returned at the end. If False, the batch data
        will be tossed after writing it and None is returned. 
    **kwargs: dict
        additional arguments passed to write_csv
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
    locs=[]; lons=[]; lats=[]; lonidxs=dict(); latidxs=dict()
    for iloc in config_loc.keys():
        locs.append(iloc)
        lons.append(config_loc.get(iloc).get('lon'))
        lats.append(config_loc.get(iloc).get('lat'))
    # create (empty) data frame for all data. Will be filled with data below
    alldat = pd.DataFrame()
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
    cnt = 0
    batchdat = pd.DataFrame()
    for i in range(len(datelist)-1):
        if cnt > batch_size and batch_size > 0:
            opened_files = _write_all(batchdat,locs,resample,ofile_template,opened_files,idate,append,**kwargs)
            del(batchdat)
            batchdat = pd.DataFrame()
            cnt = 0
        idate = datelist[i]
        jdate = datelist[i+1]
        if 'H' in read_freq:
            hrtoken = str(idate.hour).zfill(2)
        log.info('working on '+idate.strftime('%Y-%m-%d %H:%M'))
        #---Load collections for this day 
        dslist = _load_files(readcols,idate,jdate,hrtoken,error_if_not_found)
        #---Read data for every location
        outdat,latidxs,lonidxs = _sample_files(dslist,readvars,locs,lats,lons,latidxs,lonidxs)
        #---Full data array
        if return_data:
            alldat = alldat.append(outdat)
        if batch_write:
            batchdat = batchdat.append(outdat)
        # Eventually write to file
        if write_data and not batch_write:
            opened_files = _write_all(outdat,locs,resample,ofile_template,opened_files,idate,append,**kwargs)
        #---Close all files
        for l in dslist:
            if dslist[l] is not None:
                fid = dslist.get(l)
                fid.close()
        # increase counter
        cnt += 1 
    # Write out files
    if write_data and batch_write:
        idate = datelist[0] + ( datelist[-1]-datelist[0] ) / 2
        opened_files = _write_all(batchdat,locs,resample,ofile_template,opened_files,idate,append,**kwargs)
    if return_data:
        df_loc = dict()
        for l in locs:
            df_loc[l] = alldat.loc[alldat['location']==l].copy()
    else:
        df_loc = None
    # All done
    del(alldat)
    del(batchdat)
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
        # Make data 3d (time,lat,lon)
        if ds is not None:
            if 'lev' in ds.coords:
                nlev = len(ds.lev)
                ds = ds.sel(lev=nlev-1,method='nearest')
        dslist[icol] = ds
    return dslist


def _sample_files(dslist,readvars,locs,lats,lons,latidxs,lonidxs):
    '''
    Sample the previously opened files (--> dslist) at given locations. 
    The data is written into a single data frame.
    '''
    log = logging.getLogger(__name__)
    outdat = pd.DataFrame()
    empty_collections = []
    # Loop over all collections 
    for icol in readvars:
        idat = pd.DataFrame()
        ids = dslist.get(icol)
        if ids is None:
            emtpy_collections.append(icol)
            log.warning('No data found for collection {} - will be filled with NaNs'.format(icol))
            continue
        log.info('Reading collection {}...'.format(icol))
        # variables and lat/lon indeces to read
        vars = readvars.get(icol)
        if icol not in latidxs:
            latidx = [np.abs(ids.lat.values-i).argmin() for i in lats]
            lonidx = [np.abs(ids.lon.values-i).argmin() for i in lons]
            latidxs[icol] = latidx 
            lonidxs[icol] = lonidx 
        else:
            latidx = latidxs.get(icol)
            lonidx = lonidxs.get(icol)
        # create tuple of indeces
        ntime = len(ids.time.values) 
        tidx = np.repeat(np.arange(ntime),len(locs)) # [0,0,...,1,1,...,ntime-1,ntime-1]
        lidx = np.tile(np.array(latidx),ntime)       # [latidx0,latidx1,...,latidx1,latidx2,...]
        nidx = np.tile(np.array(lonidx),ntime)       # [lonidx0,lonidx1,...,lonidx1,lonidx2,...]
        idxs = tuple((tidx,lidx,nidx))
        # create data frame for this collection, add meta data 
        idat['ISO8601'] = ids.time.values[tidx]
        idat['location'] = list(np.tile(locs,ntime)) 
        idat['lat'] = list(np.tile(lats,ntime)) 
        idat['lon'] = list(np.tile(lons,ntime)) 
        # read data for each variable
        for v in vars:
            idat[v] = ids[v].values[idxs]
        if outdat.shape[0] == 0:
            outdat = idat
        else:
            outdat = outdat.merge(idat,how='outer')
    # check for emtpy collections
    if outdat.shape[0] > 0:
        for icol in empty_collections: 
            idat = pd.DataFrame()
            vars = readvars.get(icol)
            idat['ISO8601'] = np.repeat(outdat['ISO8601'].values[0],len(locs))
            idat['location'] = locs
            idat['lat'] = lats
            idat['lon'] = lons
            for v in vars:
                idat[v] = [np.nan for i in range(len(loncs))]
            outdat = outdat.merge(idat,how='outer')
    return outdat,latidxs,lonidxs


def _write_all(alldat,locs,resample,ofile_template,opened_files,idate,append,**kwargs):
    '''
    Wrapper routine to write data to csv file (by location)
    '''
    for l in locs:
        idat = alldat.loc[alldat['location']==l].copy()
        if idat.shape[0]>0:
            if resample is not None:
                idat.index = idat['ISO8601']
                idat = idat.resample(resample).mean().reset_index()
            idat['year'] = [i.year for i in idat['ISO8601']]
            idat['month'] = [i.month for i in idat['ISO8601']]
            idat['day'] = [i.day for i in idat['ISO8601']]
            idat['hour'] = [i.hour for i in idat['ISO8601']]
            opened_files = write_csv(idat,ofile_template,opened_files,idate,l,append,**kwargs)
    return opened_files

