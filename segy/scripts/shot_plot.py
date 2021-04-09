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
zmax   = 4000          # Maximum depth of model  
dz     = 5            # Sampling interval in model
IZMAX  = np.int(zmax/dz)
depth  = dz + np.arange(0,IZMAX,1) * dz

# Global variables (also used in some functions define above)
DX       = 5         # Seismic spatial sampling interval
DV       = 10
V0       = 1500
data_model  = pd.read_csv('Stkvel_files.csv', index_col=0)

# =======================================================================
# Check if all files listed in the csv file actually exist 
# (otherwise the DataLoader crashes)
# =======================================================================
for file_type in data_model.columns:
    print("Checking if all {} files exist".format(file_type))
    check_files(data_model,file_type)

sample_rows = 3
sel = random.sample(range(len(data_model)),sample_rows)
subset = data_model.iloc[sel]

fig, m_axs = plt.subplots(sample_rows,3,figsize=(13,6*sample_rows))
fig.subplots_adjust(wspace=0.45)
i=0
for (ax1,ax2,ax3), (_,files) in zip(m_axs, subset.iterrows()):
    print("file = ",files['shot'])

    shot_data  =  ReadSegyio(segyfile=files['shot'],keep_hdrs=[],drop_hdrs=[],
                              gather_id="FieldRecord",verbose=2)
    sembl_data  =  ReadSegyio(segyfile=files['sembl'],keep_hdrs=[],drop_hdrs=[],
                              gather_id="FieldRecord",verbose=2)
    v = ReadSegyio(segyfile=files['model'],keep_hdrs=[],drop_hdrs=[],
                      gather_id="FieldRecord",verbose=2)

    dt = shot_data.sample_rate/1000
    # Extract numpy array with trace values and transpose so 
    # col=time and row=trace
    d = shot_data.data["gather"][0].T
    # Apply simple processing: tpow gain to balance amplitudes upto time=maxt
    d = gains.tpow(d,dt,tpow=0.25,tmin=0,maxt=2)
    # 95th percentile gain to clip large outliers
    d = gains.perc(d,95)
    d = gains.standardize(d)

    # Extract numpy array with trace values and transpose so 
    # col=time and row=trace
    s = sembl_data.data["gather"][0].T

    m      = v.data["gather"][0].T
    vrms_t = m[:IZMAX,0]    # 0=Vrms(t), 1=Vp_int(z), 2=Vs_int(z), 3=rho_int(z)
    vint_z = m[:IZMAX,1]
    tmax   = (v.sample_rate/1000) * (len(vrms_t))

    # Plot semblance
    ax1.imshow(d,cmap='Greys',aspect='auto',extent=[0,DX*shot_data.n_traces,
                                                    1000*dt*shot_data.n_samples,0])
    if i==0: ax1.set_title('data')
    if i==len(subset)-1: ax1.set_xlabel('Offset (m)')
    ax1.set_ylabel('Time (ms)')

    # There was a factor 8 decimation in the time axis but the sample rate 
    # was not updated
    ax2.imshow(s,cmap='Greys',aspect='auto',extent=[V0,V0+DV*sembl_data.n_traces,
                                                    0.032*(sembl_data.n_samples-1),0])
    if i==0: ax2.set_title('Semblance')
    if i==len(subset)-1: ax2.set_xlabel('Velocity (m/s')
    ax2.set_ylabel('Time (ms)')
    
    # Plot Vrms(t))
    ax2.plot(vrms_t,np.linspace(0,tmax,len(vrms_t)),'r')
    ax2.set_xlim([V0,V0+DV*sembl_data.n_traces])
    ax2.set_ylim([0.032*(sembl_data.n_samples-1),0])
    if i==0: ax2.set_title('Semblance + Vrms(t)')
    if i==len(subset)-1: ax2.set_xlabel('Vrms (m/s)')
    ax2.set_ylabel('Depth (m)')
    ax2.set_xticks(np.arange(V0,V0+DV*sembl_data.n_traces,500))
    
    ax3.plot(vint_z,depth)
    ax3.invert_yaxis()
    ax3.set_xlim([V0,V0+DV*sembl_data.n_traces])
    ax3.set_ylim([zmax,0])
    if i==0: ax3.set_title('Interval velocity')
    if i==len(subset)-1: ax3.set_xlabel('V (m/s)')

    i+=1

