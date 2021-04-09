#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Paul Zwartjes
"""

## Importing the necessary libraries
## Importing the necessary libraries
import os
import random
import sys
import socket
machine=socket.gethostname()

os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import tensorflow        as tf

from tensorflow.python.client      import device_lib
sys.path.append('/home/zwartjpm/Python/modules')

#import seisplot
import gains
from readsegyio                import ReadSegyio
from processing_tools          import phase_vel_spectrum

print([device.name for device in device_lib.list_local_devices()])

def d_to_fv(d,dt,side,f_abs=0):
    """
        Convert shot gather to phase velocty vs. frequency panel
        Args:
            d       shot gather (time=col, traces=row)
            dt      time sampling interval
            side    gather with negative of positive offsets
            f_abs   plot magnitude (0) or phase
        Return:
            fv      phase velocity vs. frequency panel (v=col, f=row). Tensorflow format
    """
    # Create phase velocity vs. frequency spectrum
    fv,v,freq = phase_vel_spectrum(d,DX,dt,FMAX,VMIN,VMAX,side=1,f_abs=f_abs)
    # The rest of the function is about editing the fv panel
    fv        = np.flipud(np.swapaxes(fv,0,1).T)  # Swap axis so row=f, col=V
    
    # Editing of amplitudes in panel
    # Zero frequencies below 3.5Hz
    mask       = np.repeat(freq>3.5, fv.shape[0]).reshape((fv.shape[1],fv.shape[0])).T
    
    if f_abs == 0:
        # Zero amplitudes at extreme end of scale and convert to nan
        a = fv>=0 ; b = fv<=-60
        fv[a]    = np.nan
        fv[b]    = np.nan
    
        # Convert nan's to numbers
        fv       = np.nan_to_num(fv)
        # Standardize amplitudes (zero mean, unit std.dev.)
        fv       = gains.standardize(fv)
        # Panel has absolute amplitudes, negative extremese are noise. Threshhold at -2
        a        = fv<=-2
        fv[a]    = 0
        # Re-standardize
        fv        = gains.standardize(fv)

    # Change 'dead zone' in spectrum into zeros, Ideally let this follow the velocity 
    # slopes in the fk domain. Simply mute is easier to implement
    fv        = mask*np.nan_to_num(fv)

    # Add dimension for tensorflow
    fv        = fv.reshape([fv.shape[0],fv.shape[1],1])
    # Convert to tensorflow Tensor dtype
    fv        = tf.convert_to_tensor(fv)          

    return fv

def check_files(df,col):
    for file in df[col]:
        if not os.path.isfile(file):
            print ("File {} does not exist".format(file))
        else:
            if os.path.getsize(file) == 0:
                print("File {} is empty: ".format(file))

#========================================================================
#
#  Start here
#
#========================================================================
zmax   = 400          # Maximum depth of model  
dz     = 2            # Sampling interval in model
IZMAX  = np.int(zmax/dz)
depth  = dz + np.arange(0,IZMAX,1) * dz


# Global variables (also used in some functions define above)
DX       = 15         # Seismic spatial sampling interval
FMAX     = 40         # max. frequency for phase velocity vs. frequency panel
VMIN     = 100        # min velocity for phase velocity vs. frequency panel
VMAX     = 3000       # max velocity for phase velocity vs. frequency panel
data_model  = pd.read_csv('NearSurface_files.csv', index_col=0)

# =======================================================================
# Check if all files listed in the csv file actually exist 
# (otherwise the DataLoader crashes)
# =======================================================================
for file_type in data_model.columns:
    print("Checking if all {} files exist".format(file_type))
    check_files(data_model,file_type)

sample_rows = 4
sel = random.sample(range(len(data_model)),sample_rows)
subset = data_model.iloc[sel] ; txt = 'Synthetics_based_on_5_upholes'
fig, m_axs = plt.subplots(sample_rows,4,figsize=(13,6*sample_rows))
fig.subplots_adjust(wspace=0.45)
i=0

for (ax1,ax2,ax3,ax4), (_,files) in zip(m_axs, subset.iterrows()):
    print("file = ",files['shot'])
    input_data  =  ReadSegyio(segyfile=files['shot'],
                              keep_hdrs=[],drop_hdrs=[],
                              gather_id="FieldRecord",verbose=2)
    dt = input_data.sample_rate/1000
    # Extract numpy array with trace values and transpose so 
    # col=time and row=trace
    d = input_data.data["gather"][0].T
    # Apply simple processing: tpow gain to balance amplitudes upto time=maxt
    d = gains.tpow(d,dt,tpow=0.25,tmin=0,maxt=2)
    # 95th percentile gain to clip large outliers
    d = gains.perc(d,95)
    tmax = (input_data.n_samples-1)*dt
    d = gains.standardize(d)
    
    ax1.imshow(d,cmap='Greys',aspect='auto',extent=[0,DX*input_data.n_traces,
                                                    1000*dt*input_data.n_samples,0])
    if i==0: ax1.set_title('data')
    if i==len(subset)-1: ax1.set_xlabel('Offset (m)')
    ax1.set_ylabel('Time (ms)')

    # Convert shot gather to phase velocity vs. frequency panel
    fv_abs = np.squeeze(d_to_fv(d,dt=dt,side=1,f_abs=0).numpy())
    fvimg1 = ax2.imshow(fv_abs,cmap='jet',aspect='auto',extent=[0,FMAX,VMIN,VMAX],
                        vmin=-3,vmax=3)
    if i==0: ax2.set_title('phase velocity spectrum')
    if i==len(subset)-1: ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Velocity (m/s)')
    fig.colorbar(fvimg1, ax=ax2)

    # Convert shot gather to phase velocity vs. frequency panel
    fv_ph   = np.squeeze(d_to_fv(d,dt=dt,side=1,f_abs=1).numpy())
    fvimg2 = ax3.imshow(fv_ph,cmap='jet',aspect='auto',extent=[0,FMAX,VMIN,VMAX],
                        vmin=fv_ph.min(),vmax=fv_ph.max())
    if i==0: ax3.set_title('phase velocity spectrum')
    if i==len(subset)-1: ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Velocity (m/s)')
    fig.colorbar(fvimg2, ax=ax3)
    
    input_model = ReadSegyio(segyfile=files['model'],
                              keep_hdrs=[],drop_hdrs=[],
                              gather_id="FieldRecord",verbose=2)

    m = input_model.data["gather"][0].T
    vp1d = m[:IZMAX,0]    # 0=Vp, 1=Vs, 2=rho
    vs1d = m[:IZMAX,1]    # 0=Vp, 1=Vs, 2=rho

    ax4.plot(vs1d,depth)
    ax4.plot(vp1d,depth)
    ax4.set_xlim([150,3000])
    if i==0: ax4.set_title('model')
    ax4.invert_yaxis()
    if i==len(subset)-1: ax4.set_xlabel('V (m/s)')
    ax4.set_ylabel('Depth (m)')
    ax4.set_xticks(np.arange(0,3000,500))
    i+=1

