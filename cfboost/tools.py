#!/usr/bin/env python
# ****************************************************************************
# tools.py 
# 
# HISTORY:
# 20200204 - christoph.a.keller at nasa.gov - Initial version
# ****************************************************************************
import os
import logging


def filename_parse(string,loc='unknown',spec='unknown',type='unknown',instance=None):
    '''Replace tokens with actual values'''
    string = string.replace('%l',loc).replace('%s',spec).replace('%t',type)
    if '%n' in string:
        string = string.replace('%n','{:03}'.format(instance)) if instance is not None else string.replace('%n','')
    return string
