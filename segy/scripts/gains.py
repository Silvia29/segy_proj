#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Based on seismic unix trace scaling code

tpow          Multiply data by t^tpow
epow          Multiply data by e^(epow*t) or e^(epow*t^etpow)
agc           Apply Automatic Gain Control (AGC)
trap          Zero out outliers
clip          Hard clip values to value clip
pclip         Hard clip positive values to a value pclip
nclip         Hard clip negative values to a value nclip
mean_balance  Subtract the mean
normalize     Normalize such that 0 <= data <= 1
standardize   Standardize data such that mean=0 and std. dev. = 1
median_smooth Apply median smoothing along trace
"""

import numpy as np
import sys

def db(data,scl=True):
    db_data = 20.0*np.log10(abs(data))
    if scl:
        db_data -= db_data.max()
    return db_data

def tpow(data,dt,tpow=0,tmin=0, maxt=2):
    """
    Multiply data by t^tpow

    Args:
	data     the data
	dt       sampling rate in seconds
	tpow=0   multiply data by t^tpow
	tmin     first time on record in seconds
    maxt     keep gain fixed beyond this time (in seconds)
     """
    # Number of time samples
    nt      = data.shape[0]

    # Array with scaling factors
    t_array = tmin+np.squeeze((dt*np.array([list(range(nt))])))
    if tpow  < 0:
        t_array[0] = t_array[1]
    tpowfac = t_array**tpow

    # Keep gain factors constant beyond t=maxt
    itmax   = int(maxt/dt)
    itmax   = min(data.shape[0]-1,itmax)
    tpowfac[itmax:-1] = tpowfac[itmax]

    data    = np.apply_along_axis(lambda m: np.multiply(m, tpowfac), axis=0, arr=data)
    return data

def epow(data,dt,epow=0,etpow=1,tmin=0):
    """
    Multiply data by e^(epow*t) or e^(epow*t^etpow)

    Args:
    data     the data
    epow=0   coefficient of t in exponent
    etpow=1  exponent of t in exponent
    tmin     first time on record
    dt       sampling rate in seconds
    """
    nt       = data.shape[0]
    t_array  = tmin+np.squeeze((dt*np.array([list(range(nt))])))
    etpowfac = t_array**etpow
    data     = np.apply_along_axis(lambda m: np.multiply(m, np.exp(epow*etpowfac)), axis=0, arr=data)
    return data

def agc(data,dt,wagc=1.0):
    """
    Apply Automatic Gain Control (AGC)

    Args:
    data    the data
    dt      sampling interval in seconds
    wagc    AGC window size in seconds (default = 0.5 seconds)
    """
    nt        = data.shape[0]    # number of time samples
    iwagc     = int(wagc/dt/2)   # half window size in samples
    data_orig = np.copy(data)    # copy of input data
    d         = data_orig        # copy of input data
    nwin      = iwagc            # nwin is #samples in RMS computation
    sum       = 0

    # compute initial window for first datum
    sum = np.apply_along_axis(lambda m: np.sum(m[0:iwagc]**2.0), axis=0, arr=d)
    with np.errstate(divide='ignore', invalid='ignore'):
        d[0,:] = np.true_divide(data_orig[0,:],np.sqrt(sum/nwin))
        d[0,d[0,:] == np.inf] = 0
        d[0,:] = np.nan_to_num(d[0,:])

    # The value tmp gets subtracted each time the moving window moves 1 sample
    # forward
    tmp = data_orig[0]**2.0

    # ramping on
    # Add a squared sample and increase window length nwin each iteration
    for t in range(0,iwagc+1):
        sum = sum + data_orig[t+iwagc,:]**2.0
        nwin += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            d[t,:] = np.true_divide(data_orig[t,:],np.sqrt(sum/nwin))
            d[t,d[t,:] == np.inf] = 0
            d[t,:] = np.nan_to_num(d[t,:])

	# middle range -- full rms window
    # Add and subtract a squared sample
    for t in range(iwagc+1,nt-iwagc):
        sum = sum + data_orig[t+iwagc,:]**2.0 - tmp
        tmp =       data_orig[t-iwagc,:]**2.0
        with np.errstate(divide='ignore', invalid='ignore'):
            d[t,:] = np.true_divide(data_orig[t,:],np.sqrt(sum/nwin))
            d[t,d[t,:] == np.inf] = 0
            d[t,:] = np.nan_to_num(d[t,:])

    # ramping off
    # Subtract a squared sample and decrease window length nwin each iteration
    for t in range(nt-iwagc,nt):
        sum = sum - tmp
        tmp = data_orig[t-iwagc,:]**2.0
        nwin -= 1
        with np.errstate(divide='ignore', invalid='ignore'):
            d[t,:] = np.true_divide(data_orig[t,:],np.sqrt(sum/nwin))
            d[t,d[t,:] == np.inf] = 0
            d[t,:] = np.nan_to_num(d[t,:])

    return d

def trap(data,trap):
    """
    Zero out outliers

    Args:
    data     the data
    trap     zero if magnitude > trap
    """
    data[data > trap] = 0
    return data

def clip(data,clip):
    """
    Hard clip values to value clip

    Args:
    data     the data
    clip     clip if abs(magnitude) > clip
    """
    data[data >  clip] =  clip
    data[data < -clip] = -clip
    return data

def pclip(data,pclip):
    """
    Hard clip positive values to a value pclip

    Args:
    data     the data
    pclip    zero if magnitude > pclip
    """
    data[data > pclip] = pclip
    return data

def perc(data,percentile):
    """
    Hard clip values to a percentile

    Args:
    data       the data
    percentile percentile to clip to
    """
    clip_value = np.percentile(data,percentile)
    data = clip(data,clip_value)
    return data

def nclip(data,nclip):
    """
    Hard clip negative values to a value nclip

    Args:
    data     the data
    nclip    zero if magnitude < nclip
    """
    data[data < nclip] = nclip
    return data

def mean_balance(data):
    """
    Subtract the mean
    """
    data = data - data.mean()
    return data

def median_balance(data):
    """
    Subtract the mean
    """
    data = data - np.median(data)
    return data

def norm(data,p=2):
    """
    Divide vector by np.linalg(norm(data,p)
    """
    data = data / np.linalg.norm(data,p)

    return data

def normalize(data):
    """
    Normalize such that 0 <= data <= 1
    """
    data_range = data.max() - data.min()
    #if data_range == 0.:
    #    sys.exit("data.max() - data.min() == 0. !")
    if stddev != 0.:
        data = (data - data.min()) / data_range

    return data

def standardize(data):
    """
    Standardize data such that mean=0 and std. dev. = 1
    """
    stddev = data.std()
    #if stddev == 0.:
    #    sys.exit("data.std() == 0. !")
    if stddev != 0.:
        data = (data - data.mean()) / (data.std())

    return data

def median_smooth(data,smo=20):
    """
    Apply median smoothing along trace

    Args:
    data    the data
    smo     smoothing length in samples
    """
    nt        = data.shape[0]    # number of time samples
    data_orig = np.copy(data)    # copy of input data

    # ramping on
    for t in range(0,smo+1):
        data[t] = np.median(data_orig[0:t+smo])

	# middle range -- full median window
    for t in range(smo+1,nt-smo):
        data[t] = np.median(data_orig[t-smo:t+smo])

    # ramping off
    # Subtract a squared sample and decrease window length nwin each iteration
    for t in range(nt-smo,nt):
        data[t] = np.median(data_orig[t-smo:nt])

    return data

