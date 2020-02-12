#!/usr/bin/env python
# ****************************************************************************
# Build a 
# 
# USAGE: 
# python example.py 
#
# HISTORY:
# 20191220 - christoph.a.keller at nasa.gov - Initial version 
# ****************************************************************************

# requirements
import argparse
import sys
import numpy as np
import datetime as dt
import os
import pandas as pd
import logging

# import CFtools
sys.path.insert(1,'../')
sys.path.insert(1,'../../cfobs')

import cfboost.cfboost as cfboost

def main(args):
    # set up logger
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

    # create CFBoost object
    cfbst = cfboost.CFBoost('config/rio.yaml')

#---CF data preprocessing for training
    # read data from source (for training), write to csv file 
    if args.readcf==1:
        log.info('Read CF data needed for training (and write to csv file)')
        start = dt.datetime(2018,1,1)
        end = dt.datetime(2018,1,3)
        _ = cfbst.model_read( startday=start, endday=end, error_if_not_found=True, hdr_start=['ISO8601','year','month','day','hour','location','lat','lon'] )

#---XGBoost training
    if args.train==1:
        # run a specific location / species:
        if args.local==1:
            iloc = 'rio_bangu'
            ispec = 'o3'
            cfbst.obs_load()
            cfbst.model_load( location=iloc )
            cfbst.prepare_training_data( location=iloc, species=ispec )
            cfbst.bst_add( read_if_exists=False, species=ispec, location=iloc )
            cfbst.bst_train_and_validate( species=ispec, location=iloc )
            cfbst.bst_save( species=ispec, location=iloc )
            cfbst.bst_load( species=ispec, location=iloc )

        # train all species & locations
        else:
            cfbst.train()

#---XGBoost prediction for current 5-day forecast
    if args.prediction==1:
        iloc = 'rio_bangu' if args.local==1 else None
        ispec = 'o3' if args.local==1 else None
        cfbst.predict( startday=None, location=iloc, species=ispec, read_netcdf=True, forecast=True, duration=120, read_freq='1D' )

    # All done
    return


def parse_args():
    p = argparse.ArgumentParser(description='Undef certain variables')
    p.add_argument('-r','--readcf',type=int,help='read CF data needed for training and save to local csv file',default=1)
    p.add_argument('-t','--train',type=int,help='train XGBoost bias corrector',default=1)
    p.add_argument('-p','--prediction',type=int,help='pull current CF forecast and make corresponding bias correction using XGBoost',default=1)
    p.add_argument('-l','--local',type=int,help='train / predict for one location/species only',default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
