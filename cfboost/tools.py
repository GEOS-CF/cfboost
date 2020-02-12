#!/usr/bin/env python
# ****************************************************************************
# tools.py 
# 
# HISTORY:
# 20200204 - christoph.a.keller at nasa.gov - Initial version
# ****************************************************************************
import os
import logging


def filename_parse(string,loc='unknown',spec='unknown',type='unknown'):
    '''Replace tokens with actual values'''
    string = string.replace('%l',loc).replace('%s',spec).replace('%t',type)
    return string
