#!/usr/bin/env python
# ****************************************************************************
# plot.py 
#
# DESCRIPTION: 
# Plotting routines 
# 
# HISTORY:
# 20200130 - christoph.a.keller at nasa.gov - Initial version
# ****************************************************************************
import os
import pandas
import logging
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

import cfobs.units as cfobs_units


def plot_pred_vs_obs(modtable,obs,**kwargs):
    '''Plot predictions vs observations.'''
    log = logging.getLogger(__name__)
    ilocs = list(modtable.Location.unique()) 
    for iloc in ilocs:
        _plot_ts(modtable,obs,iloc,**kwargs)
    return


def _plot_ts(modtable,obs,iloc,ofile_template='mod_vs_obs_%l.png',sample_freq='1D',loccol='original_station_name'):
    '''Plot predictions vs observations.'''
    log = logging.getLogger(__name__)
    idat = modtable.loc[modtable['Location']==iloc]
    if idat.shape[0]==0:
        log.warning('Location {} not found in table - skip'.format(iloc))
        return
    colnames = modtable.keys()
    nrow = 3
    ncol = 1
    fig = plt.figure(figsize=(6*ncol,2*nrow))
    cnt = 0
    for irow,ispec in enumerate(['o3','no2','pm25_gcc']):
        obsspec = ispec if 'pm25' not in ispec else 'pm25'
        iobs = obs.loc[(obs[loccol]==iloc)&(obs.obstype==obsspec)].copy()
        if iobs.shape[0]==0:
            log.info('no {} observations found for {}'.format(ispec,iloc))
            continue
        origvar = [i for i in modtable.keys() if ispec+'_orig_' in i][0]
        mlvar   = [i for i in modtable.keys() if ispec+'_ML_' in i][0]
        munit = origvar.split('[')[1]
        ldat = idat.merge(iobs[['ISO8601','unit','value']],on='ISO8601',how='outer')
        ldat = _check_units(ldat,ispec,munit)
        ldat = ldat.groupby('ISO8601').mean().resample(sample_freq).mean().reset_index()
        ax = fig.add_subplot(nrow,ncol,irow+1)
        _plot_single_ts(ax,ldat,ispec,munit,origvar,mlvar)
        cnt += 1
    # clean up
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(iloc)
    ofile = ofile_template.replace('%l',iloc)
    fig.savefig(ofile)
    plt.close()
    log.info('Figure written to {}'.format(ofile))
    return


def _plot_single_ts(ax,ldat,ispec,munit,origvar,mlvar,obsvar='value',title=None):
    xlocs = [dt.datetime(2018,1,1),dt.datetime(2019,1,1),dt.datetime(2020,1,1)]
    xlabs = [i.strftime('%Y') for i in xlocs].copy()
    xticks = []
    idate = dt.datetime(2018,1,1)
    while idate < dt.datetime(2020,1,1):
        xticks.append(idate)
        idate = idate + dt.timedelta(days=35)
        idate = dt.datetime(idate.year,idate.month,1)
    ax.plot(ldat.ISO8601,ldat[origvar],color='dimgray',linewidth=2,label='Model',alpha=0.75)
    ax.plot(ldat.ISO8601,ldat[mlvar],color='black',linewidth=2,alpha=0.75,label='Model + ML')
    ax.plot(ldat.ISO8601,ldat['value'],color='red',linewidth=2,alpha=0.75,label='Observation')
    ax.legend(loc='upper left',ncol=3,frameon=False,bbox_to_anchor=(0.0,1.00))
    ax.set_xlim([min(xlocs),max(xlocs)])
    ax.set_xticks(xticks,minor=True)
    ax.set_xticks(xlocs)
    ax.set_xticklabels(xlabs)
    unitl = '[$\mu$gm$^{-3}$]' if 'ugm-3' in munit else '[ppbv]'
    if ispec=='no2':
        ylabel = 'NO$_{2}$ '+unitl
        defmax = 40.0
    if ispec=='o3':
        ylabel = 'O$_{3}$ '+unitl
        defmax = 80.0
    if 'pm25' in ispec:
        ylabel = 'PM$_{2.5}$ '+unitl
        defmax = 100.0
    origmax = np.nanmax(ldat[origvar].values)
    origmax = defmax if np.isnan(origmax) else origmax
    mlmax = np.nanmax(ldat[mlvar].values)
    mlmax = defmax if np.isnan(mlmax) else mlmax
    obsmax = np.nanmax(ldat['value'].values)
    obsmax = defmax if np.isnan(obsmax) else obsmax
    maxval = np.max([origmax,mlmax,obsmax])
    ylim = [0,maxval]
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title,fontweight='bold')
    return ax


def _check_units(ldat,ispec,munit):
    log = logging.getLogger(__name__)
    mw = 1.0
    if ispec=='o3':
        mw=48.0
    if ispec=='no2':
        mw=46.0
    # convert all ppmv to ppbv
    ldat.loc[ldat.unit.isin(['ppmv','ppm']),'value'] = ldat.loc[ldat.unit.isin(['ppmv','ppm']),'value'].values*1000.0    
    ldat.loc[ldat.unit.isin(['ppmv','ppm']),'unit'] = 'ppbv'
    if 'ugm-3' in munit:
        tmp = ldat.loc[ldat.unit.isin(['ppb','ppbv'])]
    if 'ppbv' in munit:
        tmp = ldat.loc[ldat.unit.isin(['ugm-3'])]
    if tmp.shape[0] > 0:
        ldat['conv'] = cfobs_units.get_conv_ugm3_to_ppbv(ldat,temperature_name='Temperature_[K]',pressure_name='SurfacePressure_[hPa]',pressure_scal=100.0,mw=mw)
    if 'ugm-3' in munit and tmp.shape[0]>0:
        ldat.loc[ldat.unit.isin(['ppbv','ppb']),'value'] = ldat.loc[ldat.unit.isin(['ppbv','ppb']),'value'].values/ldat.loc[ldat.unit.isin(['ppbv','ppb']),'conv'].values
    if 'ppbv' in munit and tmp.shape[0]>0:
        ldat.loc[ldat.unit.isin(['ugm-3']),'value'] = ldat.loc[ldat.unit.isin(['ugm-3']),'value'].values*ldat.loc[ldat.unit.isin(['ugm-3']),'conv'].values
    return ldat
