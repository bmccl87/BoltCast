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

"""

This file concatenates all of the gfs data together to build them into folds.  

"""

f_hour = 'f192'

x_dir = '/scratch/bmac87/x/'

files = glob.glob(x_dir+f_hour+'_*.nc')

inputs = ['graupel_250',
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


for i,file in enumerate(files):
    if i==0:
        all_ds = xr.open_dataset(file,engine='netcdf4')
    else:
        all_ds = xr.concat([all_ds,
                            xr.open_dataset(file,engine='netcdf4')],
                            dim='time',
                            data_vars='all')

all_ds = all_ds.sortby('time')
print(all_ds)
pickle.dump(all_ds,open('/scratch/bmac87/'+f_hour+'_all.p','wb'))


x_norm_dir = '/scratch/bmac87/x/normalized/'


# #create scaling object 
# scaler = StandardScaler()
# #fit scaler to training data
# scaler.fit(X_train)

# #transform feature data into scaled space 
# X_train = scaler.transform(X_train)
# X_validate = scaler.transform(X_validate)
# X_test = scaler.transform(X_test)

# #double check that mean is 0 and std is 1. 
# np.mean(X_train,axis=0),np.std(X_train,axis=0)