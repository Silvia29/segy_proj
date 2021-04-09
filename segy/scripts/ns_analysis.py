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
import time
import pickle
import socket
import shutil
machine=socket.gethostname()

os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import colorcet          as cc
import seaborn           as sns
import tensorflow        as tf
from tqdm import tqdm

from matplotlib import cm
from matplotlib.colors import ListedColormap

from sklearn.model_selection  import train_test_split

from tensorflow.keras              import Model
from tensorflow.keras              import Input

from tensorflow.keras.layers       import Activation
from tensorflow.keras.layers       import Add
from tensorflow.keras.layers       import BatchNormalization
from tensorflow.keras.layers       import Conv2D
from tensorflow.keras.layers       import Dense
from tensorflow.keras.layers       import Dropout
from tensorflow.keras.layers       import GlobalMaxPooling2D
from tensorflow.keras.layers       import MaxPool2D
from tensorflow.python.client      import device_lib

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard, TerminateOnNaN

if machine == 'dgrc-wassenaar':
    sys.path.append('/home/zwartjpm/Python/modules')
elif machine == 'nomad':
    sys.path.append('/data/Python/Aramco/modules')

#import seisplot
import gains
from readsegyio                import ReadSegyio
from processing_tools          import phase_vel_spectrum
from processing_tools          import butter_bandpass_filter

print([device.name for device in device_lib.list_local_devices()])

plt.style.use('seaborn-white')
sns.set_style("white")

def BatchActivate(input_layer):
    """
        Applies a batch normalization and activation
        Args:
            input_layer:    Input layer | tensor
        Return:
            x:              Batch normalized and activated layer
    """
    x = BatchNormalization()(input_layer)
    x = Activation('relu')(x)
    return x

def Convblock(input_layer, filters, size, strides=(1, 1), padding='same', activation=True, name='Conv'):
    """
        Performes a 2D convolution operation on the input_layer and potentially
        also a batch normalization and actiation
        Args:
            input_layer:    Input layer | tensor
            filter:         The number of filters applied to that layer | int
            size:           The size of the filter | (int, int)
            strides:        Step size of the filter | (int, int)
            padding:        How to deal with the edges, see keras documentation | string
            activation:     The type of activation function | boolean
        Return:
            x:              output layer | tensor
    """
    x = Conv2D(filters, size, strides=strides, padding=padding, 
               kernel_initializer='he_normal', name=name)(input_layer)
    if activation == True:
        x = BatchActivate(x)
    return x  
 
def Resblock(input_layer, num_filters=16, batch_activate_output=True, 
             after_pool=False, name='Resblock'):
    """
        Creating a residual block of multiple convolution operations and a skip
        connection.
        Args:
            input_layer:    Input layer | tensor
            num_filters:    The number of filters applied to that layer | int
            batch_activate: Option to apply and batch normalization and
                            activation to the Resblock in the end | boolean
            after_pool:     When the resblock is after a pooling layer, the
                            dimensions are different, a convolution needs to be
                            applied first before the two can be added. | boolean
        Return:
            x:              output layer | tensor
    """
    x = BatchActivate(input_layer)
    x = Convblock(x, num_filters, (3,3), activation=True,  name='{}_Conv1'.format(name))
    x = Convblock(x, num_filters, (3,3), activation=False, name='{}_Conv2'.format(name))
    if after_pool == True:
        input_layer = Convblock(input_layer, num_filters, (1, 1), activation=False, name='{}_Conv1x1'.format(name))
        x = Add()([x, input_layer])
    else:
        x = Add()([x, input_layer])
    if batch_activate_output == True:
        x = BatchActivate(x)
    return x
       
def build_model(in_layer, filters, n_out):

    x = Resblock(in_layer, filters,batch_activate_output=False, name='Resblock1')
    x = Resblock(x,        filters, name='Resblock2')
    x = MaxPool2D(name='Pool1')(x)
    x = Dropout(0.1)(x)

    x = Resblock(x,filters*2, name='Resblock3', after_pool=True)
    x = MaxPool2D(name='Pool2')(x)
    x = Dropout(0.1)(x)

    x = Resblock(x,filters*4, name='Resblock4', after_pool=True)
    x = MaxPool2D(name='Pool3')(x)
    x = Dropout(0.1)(x)

    x = Resblock(x, filters*2, name='Resblock5', after_pool=True)
    x = Resblock(x, filters,   name='Resblock6', after_pool=True)

    x_vp = Conv2D(512,(1,1),strides=(1,1), kernel_initializer='he_normal')(x)
    x_vp = Dense(256,activation='relu',name='Dense_vp',kernel_initializer='he_normal')(x_vp)
    x_vp = GlobalMaxPooling2D(name='Pool_vp')(x_vp)
    output_vp = Dense(n_out,activation='linear',name='vp_output',kernel_initializer='he_normal')(x_vp)

    x_vs = Conv2D(512,(1,1),strides=(1,1), kernel_initializer='he_normal')(x)
    x_vs = Dense(256,activation='relu',name='Dense_vs',kernel_initializer='he_normal')(x_vs)
    x_vs = GlobalMaxPooling2D(name='Pool_vs')(x_vs)
    output_vs = Dense(n_out,activation='linear',name='vs_output',kernel_initializer='he_normal')(x_vs)

    return output_vp, output_vs

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

def get_seismic_and_vpvs(data_file,model_file):
    """
    Used in training data loader to create input and two outputs
    Reads data and model SEGY from path strings and returns Tensorflow objects

    Model: SEGY file consisting of Vp, Vs and density. This functions extracts 
    only vp and vs and converts to float32 tensors
    
    Data: SEGY file. This function extract trace values and applies 
    pre-processing to balance amplitudes, low-cut filter. Then the shot gather
    is transformed from t-x to v-f (freq vs phase velocity)
    
    The phase velocity vs. frequency panel is resized to IMSZ[0] x IMSZ[1]
    Data augmentation is commented out. Perhaps data augmentation is best applied
    in the time domain

    Relies on global parameter IMSZ and boolean AUGMENT!

    Parameters:
    data_file           Full path to SEGY format seismic data
    model_filename      Full path to SEGY format model

    Output
    fv (input_1)        phase velocity (col) vs. frequency (row) panel (tf.float32)
    vp (vp_output)      1D array of velocity values (tf.float32)
    vs (vs_output)      1D array of velocity values (tf.float32)
    """

    # Process filenames
    # Convert to numpy and then convert bytestreams to ASCII
    data_file  = data_file.numpy().decode('ASCII')
    model_file = model_file.numpy().decode('ASCII')

    # For debugging
    #print("data_file: {}, model_file: {}".format(data_file, model_file))

    # Read SEGY model file with SEGYIO
    model1d = ReadSegyio(segyfile=model_file,keep_hdrs=[],drop_hdrs=[],
                              gather_id="FieldRecord",verbose=0)

    # Extract numpy array with trace values
    m = model1d.data["gather"][0].T

    # Extract Vs information
    vp = m[:IZMAX,0]
    vs = m[:IZMAX,1]

    # Convert to tensorflow Tensor dtype
    vp = tf.convert_to_tensor(vp)
    vs = tf.convert_to_tensor(vs)

    # Read SEGY data file with SEGYIO
    seismic  = ReadSegyio(segyfile=data_file,keep_hdrs=[],drop_hdrs=[],
                          gather_id="FieldRecord",verbose=0)
    dt = seismic.sample_rate/1000
    # Extract numpy array with trace values and transpose so 
    # col=time and row=trace
    d = seismic.data["gather"][0].T
    # Apply simple processing: tpow gain to balance amplitudes upto time=maxt
    d = gains.tpow(d,dt,tpow=0.25,tmin=0,maxt=2)
    # Apply a high-pass filter
    #d = np.apply_along_axis(lambda m: butter_highpass_filter(m,order=6, 
    #                                                         lowcut=6,fs=1/dt), 
    #                              axis=0, arr=d)
    # 95th percentile gain to clip large outliers
    d = gains.perc(d,95)

    # Convert shot gather to phase velocity vs. frequency panel
    fv = d_to_fv(d,dt=dt,side=1)

    #if AUGMENT:
    #    # Randomly change amplitudes (bulk changes)
    # Careful: random_brightness changes the mean
    #    fv = tf.image.random_brightness(fv, 1,       seed=42)
    #    fv = tf.image.random_contrast(  fv, 0.75, 2, seed=42)

    # Resize image, avoid tf.image.resize
    # Bi-linear interpolation is fine, it preserve the amplitude ranges fairly
    # well if we turn off the antialias filter. More advanced schemes lessen
    # the dynamic range
    fv = tf.image.resize(fv, [IMSZ[0],IMSZ[1]], method='bilinear', antialias=False)

    # Force to specified dtypes to reduce memory requirements. Note that
    # tf.image.resize always outputs float32. These have to be consistent
    # with the dtypes specified in the Tout option in the tf.py_function during
    # loading with the parallel mapping function
    fv = tf.cast(fv,dtype=tf.float32)
    vp = tf.cast(vp,dtype=tf.float32)
    vs = tf.cast(vs,dtype=tf.float32)

    # Return dictionary so when can specify named input/output in the 
    # Tensorflow model
    return ({"input_1":fv},{"vp_output":vp,"vs_output":vs})

def get_seismic(data_file):
    """
    Used in data loader with seismic only (in case no velocity model is 
    available. Reads data SEGY file from path strings and returns Tensorflow 
    object

    Data: SEGY file. This function extract trace values and applies 
    pre-processing to balance amplitudes, low-cut filter. Then the shot gather
    is transformed from t-x to v-f (freq vs phase velocity)
    
    The phase velocity vs. frequency panel is resized to IMSZ[0] x IMSZ[1]

    Relies on global parameter IMSZ !

    Parameters:
    data_file           Full path to SEGY format seismic data

    Output
    fv                  phase velocity (col) vs. frequency (row) panel (tf.float32)
    """

    # Process filename
    # Convert to numpy and then convert bytestreams to ASCII
    data_file  = data_file.numpy().decode('ASCII')

    # Read SEGY data file with SEGYIO
    seismic  = ReadSegyio(segyfile=data_file,keep_hdrs=[],drop_hdrs=[],
                          gather_id="FieldRecord",verbose=0)
    dt = seismic.sample_rate/1000
    # Extract numpy array with trace values and transpose so 
    # col=time and row=trace
    d = seismic.data["gather"][0].T
    # Apply simple processing: tpow gain to balance amplitudes upto time=maxt
    d = gains.tpow(d,seismic.sample_rate/1000,tpow=0.25,tmin=0,maxt=2)
    # Apply a high-pass filter
    #d = np.apply_along_axis(lambda m: butter_highpass_filter(m,order=6, 
    #                                                         lowcut=6,fs=1/dt), 
    #                              axis=0, arr=d)
    # 95th percentile gain to clip large outliers
    d = gains.perc(d,95)

    # Convert shot gather to phase velocity vs. frequency panel
    fv = d_to_fv(d,dt=dt,side=OFFSIDE)

    # Resize image, avoid tf.image.resize
    # Bi-linear interpolation is fine, it preserve the amplitude ranges fairly
    # well if we turn off the antialias filter. More advanced schemes lessen
    # the dynamic range
    fv = tf.image.resize(fv, [IMSZ[0],IMSZ[1]], method='bilinear', antialias=False)

    # Force to specified dtypes to reduce memory requirements. Note that
    # tf.image.resize always outputs float32. These have to be consistent
    # with the dtypes specified in the Tout option in the tf.py_function during
    # loading with the parallel mapping function
    fv  = tf.cast(fv, dtype=tf.float32)

    return fv

def prepare_for_training(ds, cache=True, batch_size=32, 
                         shuffle_buffer_size=1024, shuffle=True, repeat=True):
    """
    Settings for Tensorflow Dataset

    Parameters:
    ds         = Tensorflow Dataset object
    cache      = cache file name. Manually remove file if anything changes in 
    #            the preprocessing
    batch_size = batch size during training
    shuffle_buffer_size = Number of elements in shuffle buffer
    shuffle    = True for train/validation, False for test data
    repeat     = True (repeat forever) or False (repeat just once)

    Returns
    ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset)
    """
    # This is a small dataset,    only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            print(cache)
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    # cache will produce exactly the same elements during each iteration
    # through the dataset. If you wish to randomize the iteration order,
    # make sure to call shuffle after calling cache
    # The order is random but the same in each iteration if
    # reshuffle_each_iteration=False
    if shuffle:
      ds = ds.shuffle(buffer_size=shuffle_buffer_size,
                      reshuffle_each_iteration=True)

    # Repeat forever (for training/validation data) or just once (for test data)
    if repeat:
      ds = ds.repeat()
    else:
      ds = ds.repeat(1)

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training. Tune the Dataset prefetch value automatically at runtime
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #ds = ds.prefetch(buffer_size=1)

    return ds

#=============================================================================
# Create Tensorflow Slice Dataset from filenames (seismic only)
#=============================================================================
def create_Dataset(data_files, cache_file,batch_size,
                   shuffle=True, repeat=True):
    """
    Define Tensorflow dataset containing path+filename of the SEGY files
    This is for creating a cache file for a dataset with seismic only 
    # (field data)

    This creates a Tensorflow Dataset object containing the filenames (+path)
    of the SEGY files. Assumes one file per shot gather

    Parameters:
    files = list of filenames
    cache_file = name of the cache file. This allows for faster data acces
                 during training. Effecively makes of copy of the pre-processed
                 and batched data on disk
    shuffle = shuffle batches True (training/validation) or False (testing)
    repeat  = True (repeat iterator forever) or False (repeat once)

    Returns one Tensorflow objects
    ds = Tensorflow object
    """

    # Create Tensorflow Dataset using the list of filenames
    ds = tf.data.Dataset.from_tensor_slices((data_files))

    #=============================================================================
    # Define Tensorflow ParallelMapDataset object that reads seismic data from
    # disk as needed rather than from memory. Done with py_function, since
    # pure Python does not work on tensorflow objects (which is slower to
    # execute but we do it once and cache the results). Tune the number of
    # parallel calls automatically at runtime
    #=============================================================================
    ds = ds.map(lambda x: tf.py_function(func=get_seismic,
                                         inp=[x],
                                         Tout=(tf.float32)),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # =============================================================================
    # Define PrefetchDataset object. Cache file is created in directory 
    # "./cache/". Manually remove the cache file if you make any changes to the 
    # preprocessing or batch_size. Do not shuffle and repeat the test_ds 
    # object. Its sorting needs to match that of the test_split DataFrame 
    # since both are used for QC plots after training
    # =============================================================================
    ds  = prepare_for_training(ds,cache=cache_file,batch_size=batch_size,
                               shuffle=shuffle, repeat=repeat)
    return ds

#=============================================================================
# Function from github.com/tensorflow/tensorflow/issues/27679
# Enables handling nested structures like dictionaries
#=============================================================================
def new_py_function(func, inp, Tout, name=None):
  def wrapped_func(*flat_inp):
    reconstructed_inp = tf.nest.pack_sequence_as(inp, flat_inp,
                                                 expand_composites=True)
    out = func(*reconstructed_inp)
    return tf.nest.flatten(out, expand_composites=True)
  
  flat_Tout = tf.nest.flatten(Tout, expand_composites=True)
  flat_out = tf.py_function(func=wrapped_func, 
                            inp=tf.nest.flatten(inp, expand_composites=True),
                            Tout=[_tensor_spec_to_dtype(v) for v in flat_Tout],
                            name=name)
  
  spec_out = tf.nest.map_structure(_dtype_to_tensor_spec, Tout,
                                   expand_composites=True)
  
  out = tf.nest.pack_sequence_as(spec_out, flat_out, expand_composites=True)
  
  return out

def _dtype_to_tensor_spec(v):
  return tf.TensorSpec(None, v) if isinstance(v, tf.dtypes.DType) else v

def _tensor_spec_to_dtype(v):
  return v.dtype if isinstance(v, tf.TensorSpec) else v

#=============================================================================
# Create Tensorflow Slice Dataset from filenames (seismic + Vp + Vs)
#=============================================================================
def create_Dataset3(data_files,model_files, cache_file,batch_size,
                   shuffle=True, repeat=True):
    """
    Define Tensorflow dataset containing path+filename of the SEG files

    This creates a Tensorflow Dataset object containing the filenames (+path)
    of the SEGY files. Assumes one file per shot gather

    Parameters:
    files = list of filenames
    cache_file = name o f the cache file. This allows for faster data acces
                 during training. Effecively makes of copy of the pre-processed
                 and batched data on disk
    shuffle = shuffle batches True (training/validation) or False (testing)
    repeat  = True (repeat iterator forever) or False (repeat once)

    Returns two Tensorflow objects
    ds = Tensorflow object
    """

    # Create Tensorflow Dataset using the list of filenames
    ds = tf.data.Dataset.from_tensor_slices((data_files, model_files))

    #=============================================================================
    # Define Tensorflow ParallelMapDataset object that reads seismic data from
    # disk as needed rather than from memory. Done with py_function, since
    # pure Python does not work on tensorflow objects (which is slower to
    # execute but we do it once and cache the results). Tune the number of
    # parallel calls automatically at runtime
    #=============================================================================
    ds = ds.map(lambda x,y: new_py_function(
                              get_seismic_and_vpvs,
                              inp=[x,y],
                              Tout=({"input_1"  : tf.float32},
                                    {"vp_output": tf.float32,
                                     "vs_output": tf.float32})
                              ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
 
   # =============================================================================
    # Define PrefetchDataset object. Cache file is created in directory "./cache/"
    # Remove the cache file if you make any changes to the preprocessing or
    # batch_size. Do not shuffle and repeat the test_ds object. Its sorting needs
    # to match that of the test_split DataFrame sinceboth are used for QC plots
    # after training
    # =============================================================================
    ds  = prepare_for_training(ds,cache=cache_file,batch_size=batch_size,
                               shuffle=shuffle, repeat=repeat)
    return ds

# =============================================================================
# Simple function to actually loop through the Tensorflow Dataset once to
# create the cache file. Also times execution
# =============================================================================
def create_cachefile(ds, batch_size=32, steps=500):
    """
    Loop through Tensorflow Dataset once to create the cache file.

    Parameters:
    ds         = Tensorflow Dataset object
    steps      = Number of batches to loop over. Set equal to number of batches
                 in 1 epoch (do not forget this if ds.repeat() is set)
    """
    t_start = time.time()
    it = iter(ds)
    for i in range(steps):
      batch = next(it)
      if i%10 == 0:
        print(i,end='')
      elif i%2 == 0:
        print('.',end='')
    print()
    t_end = time.time()

    duration = t_end - t_start
    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} Images/s".format(batch_size*steps/duration))

def check_files(df,col):
    for file in df[col]:
        if not os.path.isfile(file):
            print ("File {} does not exist".format(file))
        else:
            if os.path.getsize(file) == 0:
                print("File {} is empty: ".format(file))

# Some fancy colormaps
jet = cm.get_cmap('jet',256)
new_jet = jet(np.linspace(0,1,256))
new_jet[0,:] = 0
new_jet_cmp = ListedColormap(new_jet)

# Modified rainbow colormap
rainbow = cc.cm.get('rainbow',256)
new_rainbow = rainbow(np.linspace(0,1,256))
new_rainbow[0,:] = 0
new_rainbow_cmp = ListedColormap(new_rainbow)

#========================================================================
#
#  Start here for training
#
#========================================================================
do_plot = False   # Generate QC plots

# local variables
batch_size = 32
vmin   = 100          # For plotting
vmax   = 3000         # For plotting
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

# Global variables (also used in some functions define above)
DX       = 5         # Seismic spatial sampling interval
IMSZ     = (128,128)  # Dimension of phase velocity panel

# South Layla model
num      = 4500   # Number of shot gathers to use for training 
                  # (new cache file needed if you change this)
# Synthetic for EBN data - Netherlands. Low velocity regime
#data_model  = pd.read_csv('Synth_model_data_3.csv', index_col=0)
# Synthetic for Aramco data - Saudi Arabia. High velocity regime
data_model  = pd.read_csv('NearSurface_files.csv', index_col=0)


# Reduce number of shot gathers
data_model = data_model.loc[:num]
IZMAX  = np.int(zmax/dz)
depth  = dz + np.arange(0,IZMAX,1) * dz

# =======================================================================
# Check if all files listed in the csv file actually exist 
# (otherwise the DataLoader crashes)
# =======================================================================
for file_type in data_model.columns:
    print("Checking if all {} files exist".format(file_type))
    check_files(data_model,file_type)

# Select fraction of input data. Then round to multiple of batch_size
print("train_test_split")
train_split, test_split = train_test_split(data_model, test_size=0.05,
                                           random_state=0, shuffle=True)
train_split, valid_split = train_test_split(train_split, test_size=0.2,
                                           random_state=0, shuffle=True)

# Keep only a multiple of batch_size elements
train_split = train_split[:batch_size*(len(train_split)//batch_size)]
valid_split = valid_split[:batch_size*(len(valid_split)//batch_size)]
test_split  = test_split[:batch_size*(len(test_split)//batch_size)]

ntrain = len(train_split)
nvalid = len(valid_split)
ntest  = len(test_split)

print("Number of training   samples/batches = {}/{}".format(ntrain, ntrain//batch_size))
print("Number of validation samples/batches = {}/{}".format(nvalid, nvalid//batch_size))
print("Number of testing    samples/batches = {}/{}".format(ntest,  ntest //batch_size))

#===========================================================================
#
#                 Tensorflow Dataset definition
#
#=============================================================================
cache_ext = ("synth1d_v5_imsz_"+str(IMSZ[0])+"_"+
             str(IMSZ[1])+"_bs"+str(batch_size)+"_num"+str(num)+".tfcache")
train_ds  = create_Dataset3(train_split["shot"].tolist(),
                           train_split["model"].tolist(),
                           cache_file="./cache/train_"+cache_ext,
                           batch_size=batch_size)
valid_ds   = create_Dataset3(valid_split["shot"].tolist(),
                           valid_split["model"].tolist(),
                           cache_file="./cache/valid_"+cache_ext,
                           batch_size=batch_size)
test_ds  = create_Dataset3(test_split["shot"].tolist(),
                           test_split["model"].tolist(),
                           cache_file="./cache/test_"+cache_ext,
                           batch_size=batch_size, shuffle=False, repeat=False)

# QC data in csv file - check whether segy to numpy conversion crashes
#for i,file in enumerate(test_split["model"].tolist()):
#    print(i,file)
#    input_data  =  ReadSegyio(segyfile=file,keep_hdrs=[],drop_hdrs=[],
#                              gather_id="FieldRecord",verbose=0)

# Iterator once throught the Dataset to create the cache file
AUGMENT=False # Don't know how to pass this parameter other than as a global
create_cachefile(test_ds,  batch_size=batch_size, steps = ntest  // batch_size)
AUGMENT=True # Don't know how to pass this parameter other than as a global
create_cachefile(valid_ds, batch_size=batch_size, steps = nvalid // batch_size)
create_cachefile(train_ds, batch_size=batch_size, steps = ntrain // batch_size)

#===========================================================================
#
#                 End Data Generator
#
#===========================================================================
if do_plot:
    sample_rows = 4
    sel = random.sample(range(len(data_model)),sample_rows)
    subset = data_model.iloc[sel] ; txt = 'Synthetics_based_on_5_upholes'
    fig, m_axs = plt.subplots(sample_rows,3,figsize=(13,6*sample_rows))
    fig.subplots_adjust(wspace=0.45)
    i=0
    #for (ax1,ax2,ax3), (_,c_row) in zip(m_axs, model_1d.sample(sample_rows).iterrows()):
    #for (ax1,ax2,ax3), (_,c_row) in zip(m_axs, model_2b.sample(sample_rows).iterrows()):
    for (ax1,ax2,ax3), (_,c_row) in zip(m_axs, subset.iterrows()):
        input_data  =  ReadSegyio(segyfile=c_row['shot'],
                                  keep_hdrs=[],drop_hdrs=[],
                                  gather_id="FieldRecord",verbose=2)
        input_model = ReadSegyio(segyfile=c_row['model'],
                                  keep_hdrs=[],drop_hdrs=[],
                                  gather_id="FieldRecord",verbose=2)
        dt = input_data.sample_rate/1000
        d = input_data.data["gather"][0].T
        d = gains.tpow(d,dt,tpow=0.25,tmin=0,maxt=2)
        d = gains.perc(d,95)
        tmax = (input_data.n_samples-1)*dt
        d = gains.standardize(d)
        
        fv = np.squeeze(d_to_fv(d,dt=dt,side=1).numpy())
        m = input_model.data["gather"][0].T
        vs1d = m[:IZMAX,1]    # 0=Vp, 1=Vs, 2=rho
        vp1d = m[:IZMAX,0]    # 0=Vp, 1=Vs, 2=rho
    
        ax1.imshow(d,cmap='Greys',aspect='auto',extent=[0,DX*input_data.n_traces,
                                                        1000*dt*input_data.n_samples,0])
        if i==0: ax1.set_title('data')
        if i==len(subset)-1: ax1.set_xlabel('Offset (m)')
        ax1.set_ylabel('Time (ms)')
    
        fv_img=ax2.imshow(fv,cmap='jet',aspect='auto',extent=[0,FMAX,VMIN,VMAX],vmin=-3,vmax=3)
        if i==0: ax2.set_title('phase velocity spectrum')
        if i==len(subset)-1: ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Velocity (m/s)')
        fig.colorbar(fv_img, ax=ax2)
    
        ax3.plot(vs1d,depth)
        ax3.plot(vp1d,depth)
        ax3.set_xlim([0,3000])
        if i==0: ax3.set_title('model')
        ax3.invert_yaxis()
        if i==len(subset)-1: ax3.set_xlabel('V (m/s)')
        ax3.set_ylabel('Depth (m)')
        ax3.set_xticks(np.arange(0,3000,500))
        i+=1
    plt.suptitle('Selected shots for '+txt)
    plt.savefig('Synth_shots_v5_'+txt+'.png')

    
    #for file in data_model["shot"]:
    #    print(file)
    data_file=train_split["shot"][0]
    seismic  = ReadSegyio(segyfile=data_file,
                          keep_hdrs=[],drop_hdrs=[],
                          gather_id="FieldRecord",verbose=0)
    d0 = seismic.data["gather"][0].T
    d1 = gains.tpow(d0,seismic.sample_rate/1000,tpow=0.25,tmin=0,maxt=2)
    d2 = gains.perc(d1,98)
    #d3 = np.apply_along_axis(lambda m: butter_highpass_filter(m,order=6, 
    #                                                         lowcut=6,fs=1/dt), 
    #                              axis=0, arr=d2)
    
    fig, axs = plt.subplots(1,3,figsize=(10,10))
    img = axs[0].imshow(d0,cmap='Greys',aspect='auto',vmin=-0.1,vmax=0.1)
    axs[0].set_title('raw') ; fig.colorbar(img, ax=axs[0])
    img = axs[1].imshow(d1,cmap='Greys',aspect='auto',vmin=-0.1,vmax=0.1)
    axs[1].set_title('clipped') ; fig.colorbar(img, ax=axs[1])
    img = axs[2].imshow(d2,cmap='Greys',aspect='auto',vmin=-0.1,vmax=0.1)
    axs[2].set_title('standardized') ; fig.colorbar(img, ax=axs[2])
    plt.suptitle('Pre-processing of synthetic shots')
    plt.savefig('Shot_prepro.png')
    plt.show()
    
    fv,v,freq = phase_vel_spectrum(d2,DX,dt,FMAX,VMIN,VMAX,side=1)
    fv0       = np.flipud(np.swapaxes(fv,  0,1).T)  # Swap axis so x=f, y=V
    # Zero frequencies below 3.5Hz
    mask       = np.repeat(freq>3.5, fv.shape[0]).reshape((fv.shape[1],fv.shape[0])).T
    fv1       = fv0.copy()
    # Zero amplitudes at extreme end of scale
    a = fv0>=0 ; b = fv0<=-60   
    fv1[a]    = np.nan
    fv1[b]    = np.nan
    fv1       = np.nan_to_num(fv1)
    fv2       = gains.standardize(fv1)
    a         = fv2<=-2
    fv2[a]    = 0
    fv2       = gains.standardize(fv2)
    fv3       = mask*np.nan_to_num(fv2)
    fv3       = fv3.reshape([fv3.shape[0],fv3.shape[1],1])
    fv3       = tf.convert_to_tensor(fv3)          # Convert to tensorflow Tensor dtype
    fv3       = tf.image.resize(fv3, [IMSZ[0],IMSZ[1]], method='lanczos5', antialias=False)
    
    fig, axs = plt.subplots(2,2,figsize=(10,10))
    img = axs[0,0].imshow(fv0,cmap='jet',aspect='auto',
                          vmin=-100,vmax=0,extent=[0,FMAX,VMIN,VMAX])
    axs[0,0].set_title('raw') ; fig.colorbar(img, ax=axs[0,0])
    img = axs[0,1].imshow(fv1,cmap='jet',aspect='auto',
                          vmin=-100,vmax=0,extent=[0,FMAX,VMIN,VMAX])
    axs[0,1].set_title('clipped') ; fig.colorbar(img, ax=axs[0,1])
    img = axs[1,0].imshow(fv2,cmap='jet',aspect='auto',
                          vmin=-3,vmax=3,extent=[0,FMAX,VMIN,VMAX])
    axs[1,0].set_title('standardized') ; fig.colorbar(img, ax=axs[1,0])
    img = axs[1,1].imshow(np.squeeze(fv3.numpy()),cmap='jet',aspect='auto',
                          vmin=-3,vmax=3,extent=[0,FMAX,VMIN,VMAX])
    axs[1,1].set_title('resized') ; fig.colorbar(img, ax=axs[1,1])
    plt.suptitle('Pre-processing of synthetic phase velocity spectra')
    plt.savefig('SouthLayla_synthetic_phasevel_prepro_m4b.png')
    plt.show()
    
    fig, axs = plt.subplots(2,2,figsize=(10,10))
    axs[0,0].hist(fv0.flatten(), bins=100) ; axs[0,0].set_title('raw')
    axs[0,1].hist(fv1.flatten(), bins=100) ; axs[0,1].set_title('clipped')
    axs[1,0].hist(fv2.flatten(), bins=100) ; axs[1,0].set_title('standardized')
    axs[1,1].hist(np.squeeze(fv3.numpy()).flatten(), bins=100) ; axs[1,1].set_title('resized')
    plt.suptitle('Histogram of synthetic phase velocity amplitudes')
    plt.savefig('SouthLayla_synthetic_phasevel_histograms_m4b.png')
    
    sample_rows = 2
    sel      = random.sample(range(batch_size),sample_rows)
    d,m = next(iter(train_ds))
    #d,m = next(iter(valid_ds))
    #d,m = next(iter(test_ds))
    d   = d['input_1']
    vp  = m['vp_output']
    vs  = m['vs_output']
    
    plt.figure()
    plt.plot(vs.numpy().T,color='red')
    plt.plot(vp.numpy().T,color='black')
    plt.title('Vp (black) and Vs (red)')
    
    data_sel = d.numpy()[sel,:,:,:]
    vs_sel   = vs.numpy()[sel,:]
    vp_sel   = vp.numpy()[sel,:]
    fig, m_axs = plt.subplots(sample_rows,3,figsize=(10,6*sample_rows))
    for (ax1,ax2,ax3), data_img, vp1d, vs1d in zip(m_axs,data_sel,vp_sel,vs_sel):
        fv_img=ax1.imshow(data_img[:,:,0],cmap='jet',aspect='auto',
                          extent=[0,FMAX,VMIN,VMAX],vmin=-3,vmax=3)
        fig.colorbar(fv_img, ax=ax1)
        # f-v spectrum
        ax1.set_title('Data')
        ax1.set_xlabel('Velocity') ; ax1.set_ylabel('Frequency')
        # Vs model
        len(depth)
        ax2.plot(vs1d,depth)
        ax2.plot(vp1d,depth)
        #ax2.set_xlim([vmin,1500])
        ax2.set_title('Model')
        ax2.invert_yaxis()
        ax2.set_xlabel('Vs (m/s)') ; ax2.set_ylabel('Depth (m)')
        # Vp/Vs ratio
        ax3.plot(vp1d/vs1d,depth,'red')
        #ax3.set_xlim([1.6,2.2])
        ax3.set_title('Vp/Vs ratio')
        ax3.invert_yaxis()
        ax3.set_xlabel('Vp/Vs') ; ax2.set_ylabel('Depth (m)')
    plt.savefig('Synth_shots_model4b.png')
            
    d_it = iter(train_ds) ; nd = ntrain
    #d_it = iter(test_ds) ; nd = ntest
    t_start = time.time()
    for i in tqdm(range(nd//batch_size)):
        _,m = next(d_it)
        vp  = m['vp_output']
        vs  = m['vs_output']
        if i==0:
            vp_model = vp
            vs_model = vs
        else:
            vp_model = np.concatenate((vp_model,vp))
            vs_model = np.concatenate((vs_model,vs))
    t_end = time.time()
    duration = t_end - t_start
    print("{} batches: {} s".format(nd//batch_size, duration))
    
    plt.figure()
    plt.hist(vp_model.flatten(), 
             bins=range(int(vp_model.min()), int(vp_model.max()) + 10, 10), 
             color='blue',alpha=0.5, label='Vp')
    plt.hist(vs_model.flatten(), 
             bins=range(int(vs_model.min()), int(vs_model.max()) + 10, 10), 
             color='red' ,alpha=0.5, label='Vs')
    plt.title("Vp and Vs histograms")
    plt.legend(loc='upper right')
    plt.savefig("SEAM_model4b_Vp-Vs_histograms.png")
    #plt.figure()
    
#=============================================================================
# Clear Tensorflow session to clear memory and reset Tensorflow layer number counters
#=============================================================================
tf.keras.backend.clear_session()
metric = 'mae'  # 'mse' or 'mae'
vpw    = 1 ; vsw    = 1 #Relative weights for Vp and Vs velocity profile
loss_fn = {'vp_output' : metric,'vs_output' : metric}
input_layer  = Input(shape=(IMSZ[0], IMSZ[1], 1),batch_size=batch_size)

model_name = ('model_v5_'+metric+str(vpw)+'vs'+str(vsw)+'_'+ str(IMSZ[0]) + 
              "_" + str(IMSZ[1]) + "_num"+ str(num))
vp_output, vs_output = build_model(input_layer,filters=64, n_out=IZMAX)

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model = Model(inputs={"input_1":input_layer}, 
              outputs={"vp_output":vp_output,"vs_output":vs_output})
model.compile(loss=loss_fn, loss_weights=[vpw, vsw],optimizer=opt,metrics=[metric])
model.summary()
    
# =============================================================================
# Some callbacks for training
# =============================================================================
model_name += '_rdc'
lr_schedule      = ReduceLROnPlateau(monitor='val_loss', factor=0.25,
                                 patience=12,min_lr=1e-6, verbose=1,
                                 cooldown=5)
early_stopping   = EarlyStopping(monitor='val_loss', patience=20, verbose=1,
                                 restore_best_weights=True)
model_checkpoint = ModelCheckpoint("./tf_models/{}.h5".format(model_name),
                                   monitor='val_loss', save_best_only=True,
                                   verbose=1, save_weights_only=False)

logdir="./logs_tb_{}".format(model_name)
tensorboard      = TensorBoard(logdir,histogram_freq=0, write_grads=False,
                               write_graph=True, write_images=False)
terminate_nan    = TerminateOnNaN()
callbacks = [early_stopping, model_checkpoint, lr_schedule, tensorboard,
             terminate_nan]
    
tf.keras.utils.plot_model(model,
                          to_file="{}.png".format(model_name),
                          show_shapes=False, show_layer_names=True,
                          rankdir='TB', expand_nested=False, dpi=96)

#====================================================================
# Train model
#====================================================================
if os.path.isdir(logdir):
    shutil.rmtree(logdir)
t1=time.time()
history = model.fit(train_ds.as_numpy_iterator(),
                    steps_per_epoch  = ntrain//batch_size,
                    validation_data  = valid_ds.as_numpy_iterator(),
                    validation_steps = nvalid//batch_size,
                    initial_epoch    = 0,
                    epochs           = 750,
                    callbacks        = callbacks,
                    verbose          = 1)
    
for key in history.history:
    print(key)

t2=time.time()
print("Training took {} seconds".format(t2-t1))

model.save("tf_models/{}_final.h5".format(model_name))

# Output history information to disk, in case I want to look at it again
with open(model_name+'_hist.pkl','wb') as output:
    pickle.dump(history.history, output, pickle.HIGHEST_PROTOCOL)
    
do_plot=True
if do_plot:
    history = pickle.load( open(model_name+'_hist.pkl',"rb"))
    train_loss = 'loss'
    val_loss   = 'val_loss'
    train_acc_vp = 'vp_output_'+metric     ; train_acc_vs = 'vs_output_'+metric
    val_acc_vp   = 'val_vp_output_'+metric ; val_acc_vs   = 'val_vs_output_'+metric

    fig, m_axs = plt.subplots(2,2,figsize=(10,10))
    plt.subplot(221)
    plt.title('MSE Loss total')
    plt.semilogy(history[train_loss] , color='blue'  , label='loss')
    plt.semilogy(history[val_loss],   color='orange', label='val_loss')
    #plt.ylim(1000,100000)
    plt.grid(True, which="both", ls="-")
    plt.legend(loc='upper right')
    # plot accuracy
    plt.subplot(222)
    plt.title('MAE loss Vp')
    plt.plot(history[train_acc_vp], color='blue'  , label='train_acc_vp')
    plt.plot(history[val_acc_vp], color='orange', label='val_vp_output_mae')
    #plt.ylim(10,100)
    plt.grid(True, which="both", ls="-")
    plt.legend(loc='upper right')
    plt.subplot(224)
    plt.title('MAE loss Vs')
    plt.plot(history[train_acc_vs], color='blue'  , label='vs_output_mae')
    plt.plot(history[val_acc_vs], color='orange', label='val_vs_output_mae')
    #plt.ylim(10,100)
    plt.grid(True, which="both", ls="-")
    plt.legend(loc='upper right')
    plt.subplot(223)
    plt.title('Learning rate')
    plt.plot(history["lr"])
    plt.show()
    plt.savefig("history_"+model_name+".png")
                
model.load_weights("./tf_models/{}.h5".format(model_name))

loss_fn = {'vp_output' : metric,'vs_output' : metric}
model.compile(loss=loss_fn, optimizer='adam', metrics=[metric])
score = model.evaluate(test_ds.as_numpy_iterator(), steps=ntest//batch_size, verbose=1)
print("Model {} has loss: {:6.4f} ".format(model_name,score[0]))
print("Vp output loss {:6.4f} and Vp output mse : {:6.4f}".format(score[1],score[3]))    
print("Vs output loss {:6.4f} and Vs output mse: {:6.4f}".format(score[2],score[4]))    

#============================================================================
#
#                     Evaluate model
#
#============================================================================
d,m      = next(iter(test_ds))
preds    = model.predict(d,batch_size=batch_size,verbose=1)
d        = d['input_1']
vp_true  = m['vp_output']
vs_true  = m['vs_output']
vp_pred  = preds['vp_output']
vs_pred  = preds['vs_output']
vpmin    = vp_pred.min() ; vsmin = vs_pred.min()
vpmax    = vp_pred.max() ; vsmax = vs_pred.max()

sample_size  = 4
sel          = random.sample(range(batch_size),sample_size)
data_sel     = d.numpy()[sel,:,:,:]
vs_model_sel = vs_true.numpy()[sel,:]
vs_pred_sel  = vs_pred[sel,:]
vp_model_sel = vp_true.numpy()[sel,:]
vp_pred_sel  = vp_pred[sel,:]
    
i=0
fig, m_axs = plt.subplots(sample_size,3,figsize=(10,6*sample_size))
for (ax1,ax2,ax3), data_img, vse, vpe, vst, vpt in zip(m_axs,data_sel,vs_pred_sel,vp_pred_sel,vs_model_sel,vp_model_sel):
    ax1.imshow(data_img[:,:,0],cmap='jet',aspect='auto',extent=[0,FMAX,VMIN,VMAX],vmin=-3,vmax=3)
    if i==0: ax1.set_title('data')
    ax2.plot(vst,depth,'blue')
    ax2.plot(vse,depth,'orange')
    ax2.plot(vpt,depth,'black')
    ax2.plot(vpe,depth,'red')
    ax3.plot(vpt/vst,depth,'blue')
    ax3.plot(vpe/vse,depth,'red')
    if i==0: ax2.set_title('model (blue/black) vs. prediction (orange/red)')
    if i==sample_size-1: ax1.set_xlabel('Offset (m)')
    if i==sample_size-1: ax2.set_xlabel('Velocity (m/s)')
    ax2.set_ylabel('Depth (m)')
    ax2.yaxis.set_label_position('right')
    ax2.invert_yaxis()
    ax2.yaxis.tick_right()
    ax2.set_xlim([0.9*vsmin,1.1*vpmax])
    ax3.set_ylabel('Depth (m)')
    ax3.yaxis.set_label_position('right')
    ax3.invert_yaxis()
    ax3.yaxis.tick_right()
    ax3.set_xlim([1.6,2.2])        
    i+=1
    plt.savefig("result_"+model_name+".png")
    
i=0
sample_size  = 4
sel          = random.sample(range(batch_size),sample_size)
data_sel     = d.numpy()[sel,:,:,:]
vs_model_sel = vs_true.numpy()[sel,:]
vs_pred_sel  = vs_pred[sel,:]
vp_model_sel = vp_true.numpy()[sel,:]
vp_pred_sel  = vp_pred[sel,:]

fig, axs = plt.subplots(1,sample_size,figsize=(10,6*sample_size))
fig.suptitle('model (black) vs. prediction (red)')
for ax,data_img, vse,vpe,vst,vpt in zip(axs,data_sel,vs_pred_sel,vp_pred_sel,vs_model_sel,vp_model_sel):
    ax.plot(vst,depth,'blue')
    ax.plot(vse,depth,'orange')
    ax.plot(vpt,depth,'black')
    ax.plot(vpe,depth,'red')
    ax.invert_yaxis()
    ax2.set_xlim([0.9*vsmin,1.1*vpmax])
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')
    if i==0:
        ax.set_ylabel('Depth (m)')
    else:
        ax.axes.yaxis.set_visible(False)
    i+=1

