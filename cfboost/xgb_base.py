#!/usr/bin/env python
# ****************************************************************************
# xgb_base.py 
# 
# DESCRIPTION:
# Class for booster object 
# 
# HISTORY:
# 20200203 - christoph.a.keller at nasa.gov - Initial version 
# ****************************************************************************
import os
import numpy as np
import pickle 
import xgboost as xgb
import logging
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy import stats 

from .tools import filename_parse


class BoosterObj(object):
    '''
    Booster object. 
    '''
    def __init__(self,bstfile,cfconfig,spec,mw,type,unit,location,lat,lon,validation_figures=None):
        self._bstfile = bstfile
        self._cfconfig = cfconfig
        self._spec = spec 
        self._mw = mw 
        self._type = type
        self._unit = unit 
        self._location = location 
        self._lat = lat 
        self._lon = lon 
        self._bst = None
        self._feature_names = None
        self._validation_figures = validation_figures if validation_figures is not None else 'xgb_%k_%l_%s_%t.png'


    def train(self,Xtrain,Ytrain,resume=True,params=None):
        '''Train model'''
        log = logging.getLogger(__name__)
        Xtrain = self._check_features(Xtrain)
        if Xtrain is None:
            return
        train = xgb.DMatrix(Xtrain,np.array(Ytrain))
        if self._bst is not None and not resume:
            self._bst = None
        params = {'booster':'gbtree'} if params is None else params
        log.info('Training XGBoost model...')
        log.debug('Use the folloing XGBoost parameter: {}'.format(params))
        self._bst = xgb.train(params,train,xgb_model=self._bst)
        return


    def predict(self,Xpred):
        '''Make prediction'''
        log = logging.getLogger(__name__)
        log.info('Make prediction for species {} at location {}'.format(self._spec,self._location))
        Xpred = self._check_features(Xpred)
        if Xpred is None:
            return None,None
        pred = xgb.DMatrix(Xpred)
        prediction = self._bst.predict(pred)
        prior = np.array(Xpred[self._spec])
        if self._type == 'bias':
            prediction = prior + prediction
        return prior,prediction


    def validate(self,Xvalid,Yvalid):
        '''Validate model'''
        log = logging.getLogger(__name__)       
        prior,prediction,truth = self.prediction_and_truth(Xvalid,Yvalid)
        self.make_scatter_plot(prior,prediction,truth)
        self.make_features_plot()
        return


    def prediction_and_truth(self,X,Y):
        '''Return predicted and true value'''
        prior,prediction = self.predict(X)
        truth = np.array(Y)
        if self._type=='bias':
            truth = prior + truth
        return prior,prediction,truth

 
    def make_scatter_plot(self,orig,predict,truth,title="Scatter plot %s at %l",minval=None,maxval=None):
        '''Make scatter plot and save figure'''
        log = logging.getLogger(__name__)       
        nfig = 3 if self._type == 'bias' else 2
        fig, axs = plt.subplots(1,nfig,figsize=(nfig*5,5))
        ii = 0
        if self._type == "bias": 
            ititle = 'Bias'
            x = predict-orig
            y = truth-orig
            m1 = min((min(x),min(y)))
            m2 = max((max(x),max(y)))
            maxval = max((abs(m1),abs(m2)))
            minval = -maxval
            xlab   = "Predicted bias ["+self._unit+"]"
            ylab   = "True bias ["+self._unit+"]"
            axs[ii] = plot_scatter(axs[ii],x,y,minval,maxval,xlab,ylab,ititle)
            ii += 1
        m1 = min((min(orig),min(truth),min(predict)))
        m2 = max((max(orig),max(truth),max(predict)))
        mxval = max((abs(m1),abs(m2)))
        mnval = 0.0
        minval = mnval if minval is None else minval
        maxval = mxval if maxval is None else maxval
        ititle = 'Original (GEOS-CF)'
        xlab   = "Predicted concentration ["+self._unit+"]"
        ylab   = "True concentration ["+self._unit+"]"
        axs[ii] = plot_scatter(axs[ii],orig,truth,minval,maxval,xlab,ylab,ititle)
        ii += 1
        ititle = 'Adjusted (XGBoost)'
        axs[ii] = plot_scatter(axs[ii],predict,truth,minval,maxval,xlab,ylab,ititle)
        title = filename_parse(title,self._location,self._spec,self._type) 
        fig.suptitle(title,y=0.98)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        ofile = filename_parse(self._validation_figures,self._location,self._spec,self._type) 
        ofile = ofile.replace('*','scatter')
        fig.savefig(ofile)
        log.info('Scatter plot written to {}'.format(ofile))
        plt.close()
        return


    def make_features_plot(self,title="Feature importances (%s)"):
        '''Make features plot and save figure'''
        log = logging.getLogger(__name__)       
        fig = plt.figure(figsize=(12,4))
        gain   = self._bst.get_score(importance_type='gain')
        cover  = self._bst.get_score(importance_type='cover')
        weight = self._bst.get_score(importance_type='weight')
        ax1    = fig.add_subplot(131)
        ax1    = plot_features(ax1,gain,title='Gain')
        ax2    = fig.add_subplot(132)
        ax2    = plot_features(ax2,cover,title='Cover')
        ax3    = fig.add_subplot(133)
        ax3    = plot_features(ax3,weight,title='Weight')
        title  = filename_parse(title,self._location,self._spec,self._type) 
        fig.suptitle(title,y=0.98)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        ofile = filename_parse(self._validation_figures,self._location,self._spec,self._type) 
        ofile = ofile.replace('*','features')
        fig.savefig(ofile)
        log.info('Scatter plot written to {}'.format(ofile))
        plt.close()
        return


    def _check_features(self,X):
        '''Make sure that columns in input array X are correctly ordered and all features are available'''
        log = logging.getLogger(__name__)
        features = X.columns.tolist()
        if self._feature_names is None:
            self._feature_names = features
            Xnew = X.copy()
        else:
            # check that all needed features are available
            for v in self._feature_names:
                if v not in features:
                    log.error('Feature "{}" not in input array'.format(v),exc_info=True)
                    return None
            # make sure order is correct
            Xnew = X[self._feature_names].copy()
        del X
        return Xnew


def plot_scatter(ax,x,y,minval,maxval,xlab,ylab,title):
    # statistics
    r,p = stats.pearsonr(x,y)
    nrmse = np.sqrt(mean_squared_error(x,y))/np.std(x)
    mb   = np.sum(y-x)/np.sum(x)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    ax.hexbin(x,y,cmap=plt.cm.gist_earth_r,bins='log')
    ax.set_xlim(minval,maxval)
    ax.set_ylim(minval,maxval)
    ax.plot((0.95*minval,1.05*maxval),(0.95*minval,1.05*maxval),color='grey',linestyle='dashed')
    # regression line
    ax.plot((0.95*minval,1.05*maxval),(intercept+(0.95*minval*slope),intercept+(1.05*maxval*slope)),color='blue',linestyle='dashed')
    ax.set_xlabel(xlab)
    if ylab != '-':
        ax.set_ylabel(ylab)
    istr = 'N = {:,}'.format(y.shape[0])
    _ = ax.text(0.05,0.95,istr,transform=ax.transAxes)
    istr = '{0:.2f}'.format(r**2)
    istr = 'R$^{2}$ = '+istr
    _ = ax.text(0.05,0.90,istr,transform=ax.transAxes)
    istr = 'NRMSE [%] = {0:.2f}'.format(nrmse*100)
    _ = ax.text(0.05,0.85,istr,transform=ax.transAxes)
    #istr = 'NMB [%] = {0:.2f}'.format(nmb*100)
    #ax.text(0.05,0.80,istr,transform=ax.transAxes)
    _ = ax.set_title(title)
    return ax


def plot_features(ax,x,title,max_features=20):
    sorted_x = sorted(x.items(), key=lambda kv: kv[1], reverse=True)
    num_features = min((max_features,len(sorted_x)))
    val = np.array([sorted_x[i][1] for i in range(num_features)])
    val = val / np.float(val.sum()) * 100.0
    lab = [sorted_x[i][0] for i in range(num_features)]
    pos = np.arange(num_features)
    ax.barh(pos,val[0:num_features][::-1],align="center",color=plt.cm.jet(1.*pos/float(num_features)))
    ax.set_yticks(pos)
    ax.set_yticklabels(lab[0:num_features][::-1])
    ax.set_ylim([-1,num_features])
    ax.set_xlabel('Feature importance [%]')
    ax.set_title(title)
    return ax
