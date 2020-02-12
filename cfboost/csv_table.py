#!/usr/bin/env python
# ****************************************************************************
# csv_table.py
# 
# HISTORY:
# 20200210 - christoph.a.keller at nasa.gov - Initial version 
# ****************************************************************************

import logging
import argparse
import sys
import numpy as np
import datetime as dt
import os
import pandas as pd


def write_csv(df,ofile_template,opened_files,idate,iloc,append=False,hdr_start=None,**kwargs):
    '''Write the dataframe 'df' to a csv file.'''
    log = logging.getLogger(__name__)
    # File to write to
    ofile = ofile_template.replace('%l',iloc)
    ofile = idate.strftime(ofile)
    # Does file exist?
    hasfile = os.path.isfile(ofile)
    # Determine if we need to append to existing file or if a new one shall be created. Don't write header if append to existing file. 
    if not hasfile:
        wm    = 'w+'
        hdr   = True
    # File does exist:
    else:
        # Has this file been previously written in this call? In this case we always need to append it. Same is true if append option is enabled
        if ofile in opened_files or append:
            wm  = 'a'
            hdr = False
        # If this is the first time this file is written and append option is disabled: 
        else:
            wm  = 'w+'
            hdr = True
    # If appending, make sure order is correct. This will also make sure that all variable names match
    if wm == 'a':
        file_hdr = pd.read_csv(ofile,nrows=1)
        df = df[file_hdr.keys()]
    else:
        # reorder to put date and location first
        if hdr_start is not None:
            old_hdr = list(df.keys())
            for i in hdr_start:
                old_hdr.remove(i)
            for i in old_hdr:
                hdr_start.append(i)
            df = df[hdr_start]
    # Write to file
    df.to_csv(ofile,mode=wm,date_format='%Y-%m-%dT%H:%M:%SZ',index=False,header=hdr,na_rep='NaN',**kwargs)
    log.info('Data for location {} written to {}'.format(iloc,ofile))
    if ofile not in opened_files:
        opened_files.append(ofile)
    return opened_files
