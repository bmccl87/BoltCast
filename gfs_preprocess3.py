from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import os
import shutil
import pickle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob

#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=18
plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
#################################################################

def norm(input_data):

    input_avg = np.mean(np.mean(np.mean(input_data,axis=2),axis=1),axis=0).values
    input_max = np.max(np.max(np.max(input_data,axis=2),axis=1),axis=0).values
    input_min = np.min(np.min(np.min(input_data,axis=2),axis=1),axis=0).values

    norm_num = input_data.values-input_avg*np.ones(input_data.values.shape)
    norm_den = np.ones(input_data.values.shape)*input_max - np.ones(input_data.values.shape)*input_min
    norm_input = np.divide(norm_num,norm_den)
    da = xr.DataArray(data=norm_input,
                      dims=["time","lat","lon"],
                      coords=dict(
                                lon=(['lon'],input_data['lon'].data),
                                lat=(['lat'],input_data['lat'].data),
                                time=(['time'],input_data['time'].data)),
                                attrs=dict(description="Normalized GFS Data: "+input_data.name))   
    return da


"""

This file normalizes the input layers between 0 and 1. 

"""

f_hour = ['f000','f024','f048','f072','f096','f120','f144','f168','f192']
rot_files = ['r0.pkl','r1.pkl','r2.pkl']

input_labels = ['graupel_250',
         'graupel_500',
         'graupel_700',
         'ice_250',
         'ice_500',
         'ice_700',
         'rain_250',
         'rain_500',
         'rain_700',
         'snow_250',
         'snow_500',
         'snow_700',
         'w_500',
         'w_700',
         'reflectivity',
         'cape',
         'lifted_index']


for i,hr in enumerate(f_hour):
    for rot,file in enumerate(rot_files):
        if i>=0 and rot>=0:
            print(hr,rot)

            all_data = pickle.load(open('/scratch/bmac87/rotations/%s/%s'%(hr,file),'rb'))#dictionary 
            print(type(all_data))

            X_train = all_data['train']['x']#dataset
            y_train = all_data['train']['y']#dataset

            X_val = all_data['val']['x']#dataset
            y_val = all_data['val']['y']#dataset

            X_test = all_data['test']['x']#dataset
            y_test = all_data['test']['y']#dataset

            # for l, label in enumerate(input_labels):

            #     print(l,label,rot,hr)

            #     one_trng_data = X_train[label]#dataArray
            #     trng_norm = norm(one_trng_data)#dataArray
            #     dict1 = {'x_norm':trng_norm}

            #     one_val_data = X_val[label]
            #     val_norm = norm(one_val_data)
            #     dict2 = {'x_norm':val_norm}

            #     one_test_data = X_test[label]
            #     test_norm = norm(one_test_data)
            #     dict3 = {'x_norm':test_norm}


            

            
