#!/usr/bin/env python
# ****************************************************************************
# Do a comparison between GEOS-CF and OpenAQ for Dec 1, 2019. 
#
# USAGE: 
# python example_openaq.py 
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

log = logging.getLogger()
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

cfbst = cfboost.CFBoost('config/rio.yaml')

# read data from source, write to csv file 
start = dt.datetime(2018,1,1)
end = dt.datetime(2018,1,3)
cfbst.model_read( startday=start,endday=end,error_if_not_found=True)

# train data 
cfbst.train_all( ... )

