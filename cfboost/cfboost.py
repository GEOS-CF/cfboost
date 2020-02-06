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

from .config    import load_config
from .config    import get_species_info
from .config    import get_location_info
from .config    import get_locations_and_species
from .obs       import obs_load
from .model     import model_load
from .model     import model_read
from .prep_data import prepare_training_data
from .prep_data import prepare_prediction_data
from .xgb_base  import BoosterObj
from .tools     import filename_parse


class CFBoost(object):
    '''
    Bias correction of GEOS-CF (surface) output using surface observations and
    the XGBoost algorithm.
    '''
    def __init__(self, configfile):
        self._config     = None
        self._cfconfig   = None
        self._configfile = 'N/A'
        self._bstlist    = []
        self._obs        = None
        self._mod        = dict()
        if configfile is not None:
            self.load_config(configfile)
            self._configfile = configfile


    def load_config(self, configfile):
        '''Load overall configuration file and the CF configuration file.'''
        self._config = load_config(configfile=configfile)
        # read CF configuration
        cf = self.__read_config('model')
        cfconfigfile = self.__read_config('config',config=cf)
        if not os.path.isfile(cfconfigfile):
            cfconfigfile = '/'.join(configfile.split('/')[0:-1])+'/'+cfconfigfile
        assert(os.path.isfile(cfconfigfile)),'File not found: {}'.format(cfconfigfile)
        self._cfconfig = load_config(configfile=cfconfigfile)
        return


    def train_all(self,read_bst_if_exists=True,**kwargs):
        '''Train models for all locations and species in the configuration file'''
        locs,specs = get_locations_and_species(self._config)
        for iloc in locs:
            self.obs_load( location=iloc )
            self.model_load( location=iloc )
            for ispec in specs:
                self.prepare_training_data( location=iloc, species=ispec, **kwargs )
                self.bst_add( read_if_exists=read_bst_if_exists, species=ispec, location=iloc )
                self.bst_train_and_validate( species=ispec, location=iloc )
                self.bst_save( species=ispec, location=iloc )
        return


    def predict_all(self,read_netcdf=False,var_PS='ps',var_T='t10m',var_TPREC='tprec',**kwargs):
        '''Make a prediction for all locations and species in the configuration file'''
        log = logging.getLogger(__name__)
        locs,specs = get_locations_and_species(self._config)
        ofile_template = self.__read_config('predfile',default='cf_output_%Y%m%d_%l.csv')
        # read model data at all locations
        if read_netcdf:
            locations = self.__read_config('locations')
            self.model_read( **kwargs )
        for iloc in locs:
            if not read_netcdf:
                self.model_load( location=iloc )
            # prepare model data at this location
            location,lat,lon = get_location_info(self._config,iloc)
            self.prepare_prediction_data( location=iloc, drop=['location','lat','lon'] )
            iso8601 = self._Xpred.pop('ISO8601')
            nr = len(iso8601)
            # make prediction for each species, collect and write to file
            idat = pd.DataFrame()
            idat['ISO8601']  = iso8601
            idat['Location'] = [location for x in range(nr)] 
            idat['Lat_[degN]'] = [lat for x in range(nr)] 
            idat['Lon_[degE]'] = [lon for x in range(nr)] 
            idat['SurfacePressure_[hPa]'] = self._mod[var_PS].values*0.01
            idat['Temperature_[K]'] = self._mod[var_T].values
            idat['Precipitation_[mm]'] = self._mod[var_TPREC].values
            for ispec in specs:
                spec,mw,type,unit = get_species_info(self._config,ispec)
                bstfile = self.__get_bstfile( location=iloc, species=ispec )
                if os.path.isfile(bstfile):
                    self.bst_load( bstfile )
                    prior,pred = self.bst_predict( species=ispec, location=iloc )
                else:
                    log.warning('Booster does not exist - skip prediction: {}'.format(bstfile))
                    prior = self._mod[spec]
                    pred  = prior * np.nan
                # collect data in table
                idat[spec+'_orig_['+unit+']'] = prior 
                idat[spec+'_ML_['+unit+']']   = pred
            # write table
            ofile = filename_parse(ofile_template,location,spec,type)
            startday = idat['ISO8601'].min()
            ofile = startday.strftime(ofile)
            idat.to_csv(ofile,date_format='%Y-%m-%dT%H:%M:%SZ',index=False,float_format='%.4f')
            log.info('Model output written to {}'.format(ofile)) 
        return


    def obs_load(self,startdate=None,location='',**kwargs):
        '''Load observation data'''
        if location == '':
            locs = self.__read_config('locations')
            locations = [locs.get(k).get('name',k) for k in locs.keys()]
        else:
            locations = [location]
        obs = self.__read_config('observations')
        obsfile = self.__read_config('obsfile',config=obs)
        obsfile = filename_parse(obsfile,loc=location)
        self._obs = obs_load(obsfile,startdate,location_filter=locations,**kwargs)
        return


    def model_load(self,location='',**kwargs):
        '''Load (pre-saved) model data'''
        log = logging.getLogger(__name__)
        if location in self._mod:
            log.warning('Model data for this station already exists - will be overwritten: {}'.format(location))
        self._mod[location] = model_load(self._cfconfig,location,**kwargs)
        return 


    def model_read(self, startday, **kwargs):
        '''Read model data from original source (netCDF file)'''
        log = logging.getLogger(__name__)
        config_loc = self.__read_config('locations') 
        self._mod = model_read(config_cf=self._cfconfig,config_loc=config_loc,startday=startday,**kwargs)
        return


    def prepare_training_data(self,location,**kwargs):
        '''Prepare data for xgboost training'''
        self._Xtrain,self._Ytrain,self._Xvalid,self._Yvalid = prepare_training_data(self._mod[location],self._obs,self._config,location,**kwargs)
        return


    def prepare_prediction_data(self,location,**kwargs):
        '''Prepare data for xgboost prediction'''
        self._Xpred = prepare_prediction_data(self._mod[location],self._config,location**kwargs)
        return


    def bst_add(self,bstfile=None,read_if_exists=True,species=None,location=None):
        '''Add booster object'''
        if bstfile is None:
            bstfile = self.__get_bstfile( location=location, species=species )
        if os.path.exists(bstfile) and read_if_exists:
            self.bst_load(bstfile)
        else:
            location,lat,lon  = get_location_info(self._config,location)
            spec,mw,type,unit = get_species_info(self._config,species)
            bst = BoosterObj(bstfile=bstfile,cfconfig=self._cfconfig,spec=spec,
                             mw=mw,type=type,unit=unit,location=location,lat=lat,lon=lon)
            self._bstlist.append(bst)
        return


    def bst_load(self,bstfile):
        '''Load booster object'''
        assert(os.path.exists(bstfile)), 'File not found: {}'.format(bstfile)
        bst = pickle.load(open(bstfile,"r+"))
        self._bstlist.append(bst)
        return


    def bst_train_and_validate(self,species=None,location=None):
        '''Train the model'''
        bst = self.__bst_get(species,location)
        bst.train(self._Xtrain,self._Ytrain)
        bst.validate(self._Xvalid,self._Yvalid)
        return


    def bst_predict(self,species=None,location=None):
        '''Make prediction'''
        bst = self.__bst_get(species,location)
        prior,Ypred = bst.predict(self._Xpred)
        return prior,Ypred


    def bst_save(self,species=None,location=None):
        '''Save the booster object to disk'''
        log = logging.getLogger(__name__)
        bst = self.__bst_get(species,location)
        if bst is not None:
            pickle.dump(bst, open(bst._bstfile, "wb"))
        log.info('Booster object written to {}'.format(bst._bstfile))
        return


    def __bst_get(self,species,location):
        '''Return the booster object for the given species and location'''
        bst = None
        spec,mw,type,unit = get_species_info(self._config,species)
        location,lat,lon = get_location_info(self._config,location)
        for ibst in self._bstlist:
            if ibst._spec==spec and ibst._location==location:
                bst = ibst
                break                 
        return bst


    def __get_bstfile(self, location, species):
        '''Return the booster object filename'''
        location,lat,lon  = get_location_info(self._config,location)
        spec,mw,type,unit = get_species_info(self._config,species)
        bstfile = self.__read_config('bstfile',default='bst_%l_%s_%t.pkl')
        bstfile = filename_parse(bstfile,location,spec,type)
        return bstfile


    def __read_config(self, key, config=None, default=None):
        '''Helper routine to read configuration file'''
        config = self._config if config is None else config
        val = config.get(key,default)
        assert(val is not None),'keyword '+key+' not found in configuration file: '+self._configfile
        return val
