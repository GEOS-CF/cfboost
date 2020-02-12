#!/usr/bin/env python
# ****************************************************************************
# cfboost.py 
# 
# DESCRIPTION:
# Definition of the CFBoost object that handles bias correction of GEOS-CF
# output using surface observations and XGBoost. 
# 
# HISTORY:
# 20200130 - christoph.a.keller at nasa.gov - Initial version 
# ****************************************************************************
import os
import logging
import pickle
import datetime as dt
import pandas as pd
import numpy as np

from .configfile   import load_config
from .configfile   import get_species_info
from .configfile   import get_location_info
from .configfile   import get_locations_and_species
from .obs          import obs_load
from .model        import model_load
from .model        import model_read
from .prepare_data import prepare_training_data
from .prepare_data import prepare_prediction_data
from .xgb_base     import BoosterObj
from .tools        import filename_parse
from .csv_table    import write_csv


class CFBoost(object):
    '''
    Bias correction of GEOS-CF (surface) output using surface observations and
    the XGBoost algorithm.
    '''
    def __init__(self, configfile):
        self._config     = None
        self._cfconfig   = None
        self._configfile = 'N/A'
        self._obs        = None
        self._mod        = dict()
        self._bstobj     = dict()
        if configfile is not None:
            self.load_config(configfile)
            self._configfile = configfile


    def load_config(self, configfile):
        '''Load overall configuration file and the CF configuration file.'''
        self._config = load_config(configfile=configfile)
        # read CF configuration
        cf = self._read_config('model')
        cfconfigfile = self._read_config('config',config=cf)
        if not os.path.isfile(cfconfigfile):
            cfconfigfile = '/'.join(configfile.split('/')[0:-1])+'/'+cfconfigfile
        assert(os.path.isfile(cfconfigfile)),'File not found: {}'.format(cfconfigfile)
        self._cfconfig = load_config(configfile=cfconfigfile)
        return


    def train(self,location=None,species=None,read_obs_by_location=False,read_bst_if_exists=True,**kwargs):
        '''Train model for locations and species in the configuration file'''
        locs,specs = get_locations_and_species(self._config)
        locations = [location] if location is not None else locs
        species   = [species]  if species  is not None else specs
        if not read_obs_by_location:
            self.obs_load()
        for iloc in locations:
            if read_obs_by_location:
                self.obs_load( location=iloc )
            self.model_load( location=iloc )
            for ispec in species:
                rc = self.prepare_training_data( location=iloc, species=ispec, **kwargs )
                if rc != 0:
                    continue
                self.bst_add( read_if_exists=read_bst_if_exists, species=ispec, location=iloc )
                self.bst_train_and_validate( species=ispec, location=iloc )
                self.bst_save( species=ispec, location=iloc )
        return


    def predict(self,startday=None,location=None,species=None,read_netcdf=True,var_PS='ps',var_T='t10m',var_TPREC='tprec',**kwargs):
        '''Make a prediction for locations and species in the configuration file and save if to csv file.'''
        log = logging.getLogger(__name__)
        # settings
        xgbconfig = self._read_config('xgboost_config')
        ofile_template = self._read_config('prediction_file',config=xgbconfig,default='cf_output_%Y%m%d_%l.csv')
        locs,specs = get_locations_and_species(self._config)
        locations = [location] if location is not None else locs
        species   = [species]  if species  is not None else specs
        # read model data at all locations
        if read_netcdf:
            startday = self.model_read( startday=startday, write_data=False, return_data=True, **kwargs )
        opened_files = []
        for iloc in locations:
            if not read_netcdf:
                self.model_load( location=iloc )
            # prepare model data at this location
            lockey,locname,lat,lon = get_location_info(self._config,iloc)
            self.prepare_prediction_data( location=lockey, drop=['location','lat','lon'] )
            idat = self._init_prediction_table(lockey,lat,lon,var_PS,var_T,var_TPREC)
            # make prediction for each species, collect and write to file
            for ispec in species:
                speckey,specname,mw,type,unit = get_species_info(self._config,ispec)
                prior,pred = self.bst_predict( species=speckey, location=lockey )
                if prior is None or pred is None:
                    log.warning('Booster does not exist - skip prediction for species {} at location {}'.format(speckey,lockey))
                    prior = self._mod[lockey][speckey].values
                    pred  = prior * np.nan
                # collect data in table
                idat[speckey+'_orig_['+unit+']'] = prior 
                idat[speckey+'_ML_['+unit+']'  ] = pred
            # write table
            opened_files = write_csv(df=idat,ofile_template=ofile_template,opened_files=opened_files,idate=startday,iloc=lockey,append=False,float_format='%.4f')
        return


    def obs_load(self,**kwargs):
        '''Load observation data'''
        self._obs = obs_load(config=self._config,**kwargs)
        return


    def model_load(self,location='',**kwargs):
        '''Load (pre-saved) model data'''
        log = logging.getLogger(__name__)
        if self._mod is None:
            self._mod = dict()
        if location in self._mod:
            log.warning('Model data for this station already exists - will be overwritten: {}'.format(location))
        self._mod[location] = model_load(self._cfconfig,location,**kwargs)
        return 


    def model_read(self, startday, **kwargs):
        '''Read model data from original source (netCDF file)'''
        log = logging.getLogger(__name__)
        config_loc = self._read_config('locations') 
        if startday is None:
            iday = dt.datetime.today() - dt.timedelta(days=2)
            startday = dt.datetime(iday.year,iday.month,iday.day)
        self._mod = model_read(config_cf=self._cfconfig,config_loc=config_loc,startday=startday,**kwargs)
        return startday


    def prepare_training_data(self,location,**kwargs):
        '''Prepare data for xgboost training'''
        self._Xtrain,self._Xvalid,self._Ytrain,self._Yvalid = prepare_training_data(self._mod[location],self._obs,self._config,location,**kwargs)
        rc = 0 if self._Xtrain is not None else -1
        return rc


    def prepare_prediction_data(self,location,**kwargs):
        '''Prepare data for xgboost prediction'''
        self._Xpred = prepare_prediction_data(self._mod[location],self._config,location,**kwargs)
        return


    def bst_add(self,bstfile=None,read_if_exists=True,species=None,location=None):
        '''Add booster object'''
        log = logging.getLogger(__name__)
        if bstfile is None:
            bstfile = self._get_bstfile( location=location, species=species )
        if os.path.exists(bstfile) and read_if_exists:
            self.bst_load(bstfile=bstfile)
        else:
            lockey,locname,lat,lon  = get_location_info(self._config,location)
            speckey,specname,mw,type,unit = get_species_info(self._config,species)
            xgbconfig = self._read_config('xgboost_config')
            validation_figures = self._read_config('validation_figures',config=xgbconfig)
            bst = BoosterObj(bstfile=bstfile,cfconfig=self._cfconfig,spec=speckey,
                   mw=mw,type=type,unit=unit,location=lockey,lat=lat,lon=lon,validation_figures=validation_figures)
            self._bst_add(bst,bstfile)
        return


    def bst_train_and_validate(self,species=None,location=None):
        '''Train the model'''
        bst = self.bst_get(species,location)
        bst.train(self._Xtrain,self._Ytrain)
        bst.validate(self._Xvalid,self._Yvalid)
        return


    def bst_predict(self,species=None,location=None):
        '''Make prediction'''
        bst = self.bst_get(species,location)
        if bst is not None:
            prior,Ypred = bst.predict(self._Xpred)
        else:
            prior=None; Ypred=None
        return prior,Ypred


    def bst_save(self,species=None,location=None,bstfile=None):
        '''Save the booster object to disk'''
        log = logging.getLogger(__name__)
        bst = self.bst_get(species,location)
        if bst is not None:
            bstfile = bst._bstfile if bstfile is None else bstfile
            pickle.dump(bst, open(bst._bstfile, "wb"))
        log.info('Booster object written to {}'.format(bstfile))
        return


    def bst_load(self,bstfile=None,species=None,location=None,not_found_ok=False):
        '''Load booster object'''
        log = logging.getLogger(__name__)
        if bstfile is None:
            bstfile = self._get_bstfile( location=location, species=species )
        if not os.path.exists(bstfile):
            if not_found_ok:
                log.warning('File not found: {}'.format(bstfile))
            else:
                log.error('File not found: {}'.format(bstfile))
        else:
            bst = pickle.load(open(bstfile,"rb"))
            self._bst_add(bst,bstfile)
        return


    def _init_prediction_table(self,loc,lat,lon,var_PS,var_T,var_TPREC):
        '''Initialize table with model data & predictions'''
        log = logging.getLogger(__name__)
        iso8601 = self._Xpred['ISO8601']
        nr = len(iso8601)
        idat = pd.DataFrame()
        idat['ISO8601']  = iso8601
        idat['Location'] = [loc for x in range(nr)] 
        idat['Lat_[degN]'] = [lat for x in range(nr)] 
        idat['Lon_[degE]'] = [lon for x in range(nr)]
        idat = self._add_var(idat,loc,'SurfacePressure_[hPa]',var_PS,0.01)
        idat = self._add_var(idat,loc,'Temperature_[K]',var_T)
        idat = self._add_var(idat,loc,'Precipitation_[mm]',var_TPREC)
        return idat


    def _add_var(self,df,location,var,varname,scal=1.0):
        '''Add variable from data set to data frame'''
        log = logging.getLogger(__name__)
        dat = self._mod[location] if location in self._mod else []
        if varname in dat:
            df[var] = dat[varname].values * scal
        else:
            df[var] = np.zeros((df.shape[0],))*np.nan
            log.warning('Variable {} not found in original data set, variable {} is set to NaN'.format(varname,var)) 
        return df


    def _bst_add(self,bst,bstfile):
        '''Attach booster object'''
        log = logging.getLogger(__name__)
        if bstfile in self._bstobj:
            log.warning('booster object already exists in memory, will be overwritten: {}'.format(bstfile))
        self._bstobj[bstfile] = bst
        return


    def bst_get(self,species=None,location=None,bstfile=None,load_if_not_found=True):
        '''Return the booster object for the given species and location'''
        log = logging.getLogger(__name__)
        if bstfile is None:
            bstfile = self._get_bstfile(location=location,species=species)
        if bstfile not in self._bstobj and load_if_not_found:
            self.bst_load(bstfile=bstfile,not_found_ok=True)
        if bstfile not in self._bstobj:
            log.warning('booster object not found - return empty object: {}'.format(bstfile))
            bst = None
        else:
            bst = self._bstobj[bstfile]
        return bst


    def _get_bstfile(self, location, species):
        '''Return the booster object filename'''
        lockey,locname,lat,lon  = get_location_info(self._config,location)
        speckey,specname,mw,type,unit = get_species_info(self._config,species)
        xgbconfig = self._read_config('xgboost_config')
        bstfile = self._read_config('bstfile',config=xgbconfig,default='bst_%l_%s_%t.pkl')
        bstfile = filename_parse(bstfile,lockey,speckey,type)
        return bstfile


    def _read_config(self, key, config=None, default=None):
        '''Helper routine to read configuration file'''
        config = self._config if config is None else config
        val = config.get(key,default)
        assert(val is not None),'keyword '+key+' not found in configuration file: '+self._configfile
        return val
