#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:59:58 2019

@author: zwartjpm
"""
import pandas as pd
import os
from tqdm import tqdm
import sys
import socket
machine=socket.gethostname()

sys.path.append('/home/zwartjpm/Python/modules')
data_dir  = '/home/zwartjpm/data/seismic/StkVelProject/data_cmp/'
model_dir = '/home/zwartjpm/data/seismic/StkVelProject/model_cmp/'

data_prefix       = 'vz_model_'
data_suffix_sembl = '_sembl.sgy'
data_suffix_velan = '_velan.sgy'
model_prefix      = 'model_'
model_suffix      = '_velan.sgy'

#==========================================================================
# xdir, positive offsets
#==========================================================================
#nd_files=len([file for  file in os.listdir(data_dir) if file.endswith('.sgy')])
#nm_files=len([file for file in os.listdir(model_dir) if file.endswith('.sgy')])
nd_sembl_files=0
for file in os.listdir(data_dir):
    if file.endswith(data_suffix_sembl):
        if file.startswith(data_prefix):
            nd_sembl_files+=1

nd_velan_files=0
for file in os.listdir(data_dir):
    if file.endswith(data_suffix_velan):
        if file.startswith(data_prefix):
            nd_velan_files+=1

nm_files=0
for file in os.listdir(model_dir):
    if file.endswith('.sgy'):
        if file.startswith(model_prefix):
            nm_files+=1

#nfiles = max(nm_files, nd_files) + 1
nfiles = 10000
df_velan = pd.DataFrame(index=range(nfiles),columns=["num","shot","sembl","model"])

for file in os.listdir(data_dir):
    if file.endswith(data_suffix_sembl):
        if file.startswith(data_prefix):
            #print(file)
            num = int(file.split('_')[-2])
            #print(num,file)
            df_velan.iloc[num]["num"] = num
            df_velan.iloc[num]["sembl"] = data_dir+file

for file in os.listdir(data_dir):
    if file.endswith(data_suffix_velan):
        if file.startswith(data_prefix):
            #print(file)
            num = int(file.split('_')[-2])
            #print(num,file)
            df_velan.iloc[num]["num"] = num
            df_velan.iloc[num]["shot"] = data_dir+file

for file in os.listdir(model_dir):
    if file.endswith('.sgy'):
        if file.startswith(model_prefix):
            #print(file)
            num = int(file.split('_')[1])
            #print(num,file)
            df_velan.iloc[num]["num"] = num
            df_velan.iloc[num]["model"] = model_dir+file

df_velan.sort_values(by=['num'],inplace=True)
df_velan.dropna(axis=0, how='any', inplace=True)
df_velan.reset_index(inplace=True,drop=True)
df_velan.to_csv('Stkvel_files.csv', index=False)

print("Number of data/model files = ",len(df_velan))
