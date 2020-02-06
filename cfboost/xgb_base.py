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
    def __init__(self,bstfile,cfconfig,spec,mw,type,unit,location,lat,lon):
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


    def train(self,Xtrain,Ytrain,resume=True):
        '''Train model'''
        log = logging.getLogger(__name__)       
        train = xgb.DMatrix(Xtrain,Ytrain)
        param = {'booster' :'gbtree'}        
        if self._bst is not None and not resume:
            self._bst = None
        log.info('Training XGBoost model...')
        self._bst = xgb.train(param,train,xgb_model=self._bst)
        return


    def predict(self,Xpred):
        '''Make prediction'''
        pred = xgb.DMatrix(Xpred)
        prediction = self._bst.predict(pred)
        prior = np.array(Xpred[self._spec])
        if self._type == 'tend':
            prediction = prior + prediction
        return prior,prediction


    def validate(self,Xvalid,Yvalid):
        '''Validate model'''
        log = logging.getLogger(__name__)       
        o_valid,p_valid,t_valid = self.prediction_and_truth(Xvalid,Yvalid)
        self.make_scatter_plot(o_valid,p_valid,t_valid)
        self.make_features_plot()
        return


    def prediction_and_truth(self,X,Y):
        '''Return predicted and true value'''
        prior,prediction = self.predict(X)
        truth = np.array(Y)
        if self._type=='tend':
            truth = prior + truth
        return prior,prediction,truth

 
    def make_scatter_plot(self,orig,predict,truth,title="Scatter plot %s at %l",minval=None,maxval=None):
        '''Make scatter plot and save figure'''
        log = logging.getLogger(__name__)       
        nfig = 3 if self._type == 'tend' else 2
        fig, axs = plt.subplots(1,nfig)
        ii = 0
        if self._type == "tend": 
            title = 'Tendencies'
            x = predict-orig
            y = truth-orig
            m1 = min((min(x),min(y)))
            m2 = max((max(x),max(y)))
            maxval = max((abs(m1),abs(m2)))
            minval = -maxval
            xlab   = "Predicted tendency ["+self._unit+"]"
            ylab   = "True tendency ["+self._unit+"]"
            axs[ii] = plot_scatter(axs[ii],x,y,minval,maxval,xlab,ylab,title)
            ii += 1
        m1 = min((min(orig),min(truth),min(predict)))
        m2 = max((max(orig),max(truth),max(predict)))
        mxval = max((abs(m1),abs(m2)))
        mnval = -maxval
        minval = mnval if minval is None else minval
        maxval = mxval if maxval is None else maxval
        title = 'Original model (GEOS-CF)'
        xlab   = "Predicted concentration ["+self._unit+"]"
        ylab   = "True concentration ["+self._unit+"]"
        axs[ii] = plot_scatter(axs[ii],orig,truth,minval,maxval,xlab,ylab,title)
        ii += 1
        title = 'Adjusted model (XGBoost)'
        axs[ii] = plot_scatter(axs[ii],predict,truth,minval,maxval,xlab,ylab,title)
        title = filename_parse(title,self._location,self._spec,self._type) 
        fig.suptitle(title,y=0.98)
        #plt.tight_layout()
        ofile = filename_parse(ofile,self._location,self._spec,self._type) 
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
        plt.tight_layout()
        ofile = filename_parse(ofile,self._location,self._spec,self._type) 
        ofile = ofile.replace('*','features')
        fig.savefig(ofile)
        log.info('Scatter plot written to {}'.format(ofile))
        plt.close()
        return


def plot_scatter(ax,x,y,minval,maxval,xlab,ylab,title):
    # statistics
    r,p = stats.pearsonr(x,y)
    nrmse = sqrt(mean_squared_error(x,y))/np.std(x)
    nmb   = np.sum(y-x)/np.sum(x)
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
    ax.text(0.05,0.95,istr,transform=ax.transAxes)
    istr = 'R$^{2}$={0:.2f}'.format(r**2)
    ax.text(0.05,0.90,istr,transform=ax.transAxes)
    istr = 'NRMSE [%] = {0:.2f}'.format(nrmse*100)
    ax.text(0.05,0.85,istr,transform=ax.transAxes)
    #istr = 'NMB [%] = {0:.2f}'.format(nmb*100)
    #ax.text(0.05,0.80,istr,transform=ax.transAxes)
    ax.set_title(title)
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
