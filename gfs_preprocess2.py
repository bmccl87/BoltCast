# from sklearn.preprocessing import StandardScaler

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

This file builds the folds for model training, testing, and validation.  

"""

f_hour = 'f000'

x_dir = '/scratch/bmac87/'
y_dir = '/scratch/bmac87/y/'

yrs = ['2019','2020','2021','2022']

for i,yr in enumerate(yrs):

    if i==0:
        y = xr.open_dataset(y_dir+'CONUS_Binary_Classification_'+yr+'.nc',engine='netcdf4')
    else:
        y=xr.concat([y,
                    xr.open_dataset(y_dir+'CONUS_Binary_Classification_'+yr+'.nc',engine='netcdf4')],
                    data_vars='all',
                    dim='time'
        )

y = y.sortby('time')

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

output_label = 'binary_ltg'



###############################################################

split_2020 = [slice('1/1/2020 00:00:00','5/31/2020 00:00:00'),
            slice('7/1/2020 00:00:00','12/31/2020 00:00:00')]

split_2021 = [slice('1/1/2021 00:00:00','5/31/2021 00:00:00'),
            slice('7/1/2021 00:00:00','12/31/2021 00:00:00')]

split_2022 = [slice('1/1/2022 00:00:00','5/31/2022 00:00:00'),
            slice('7/1/2022 00:00:00','9/30/2022 00:00:00')]

yr2019 = slice('7/1/2019 00:00:00','12/31/2019 00:00:00')
yr2020 = slice('1/1/2020 00:00:00','12/31/2020 00:00:00')
yr2021 = slice('1/1/2021 00:00:00','12/31/2021 00:00:00')
yr2022 = slice('1/1/2022 00:00:00','9/30/2022 00:00:00')

june2020 = slice('6/1/2020 00:00:00','6/30/2020 00:00:00')
june2021 = slice('6/1/2021 00:00:00','6/30/2021 00:00:00')
june2022 = slice('6/1/2022 00:00:00','6/30/2022 00:00:00')

x = pickle.load(open(x_dir+f_hour+'_all.p','rb'))

yr2019_x = x.sel(time=yr2019)
yr2019_y = y.sel(time=yr2019)

print('2020')
yr2020_x = x.sel(time=yr2020)
yr2020_y = y.sel(time=yr2020)

print('2021')
yr2021_x = x.sel(time=yr2021)
yr2021_y = y.sel(time=yr2021)

print('2022')
yr2022_x = x.sel(time=yr2022)
yr2022_y = y.sel(time=yr2022)


yr2022_x_split = xr.concat([yr2022_x.sel(time=split_2022[0]),
                            yr2022_x.sel(time=split_2022[1])],
                            data_vars='all',
                            dim='time')

yr2022_y_split = xr.concat([yr2022_y.sel(time=split_2022[0]),
                            yr2022_y.sel(time=split_2022[1])],
                            data_vars='all',
                            dim='time')

yr2021_x_split = xr.concat([yr2021_x.sel(time=split_2021[0]),
                            yr2021_x.sel(time=split_2021[1])],
                            data_vars='all',
                            dim='time')

yr2021_y_split = xr.concat([yr2021_y.sel(time=split_2021[0]),
                            yr2021_y.sel(time=split_2021[1])],
                            data_vars='all',
                            dim='time')

yr2020_x_split = xr.concat([yr2020_x.sel(time=split_2020[0]),
                            yr2020_x.sel(time=split_2020[1])],
                            data_vars='all',
                            dim='time')

yr2020_y_split = xr.concat([yr2020_y.sel(time=split_2020[0]),
                            yr2020_y.sel(time=split_2020[1])],
                            data_vars='all',
                            dim='time')                    


#build the validation and test sets
june2020_x = x.sel(time=june2020)
june2020_y = y.sel(time=june2020)

june2021_x = x.sel(time=june2021)
june2021_y = y.sel(time=june2021)

june2022_x = x.sel(time=june2022)
june2022_y = y.sel(time=june2022)

r0_val_x = june2020_x
r0_val_y = june2020_y

r0_test_x = june2021_x
r0_test_y = june2021_y

r1_val_x = june2021_x
r1_val_y = june2021_y

r1_test_x = june2022_x
r1_test_y = june2022_y

r2_val_x = june2022_x
r2_val_y = june2022_y

r2_test_x = june2020_x
r2_test_y = june2020_y

r0_train_x = xr.concat([yr2019_x,yr2020_x_split,yr2021_x_split,yr2022_x],
                        data_vars='all',
                        dim='time')

r0_train_y = xr.concat([yr2019_y,yr2020_y_split,yr2021_y_split,yr2022_y],
                        data_vars='all',
                        dim='time')

print('rotation zero shapes')
print(r0_train_x.dims)
print(r0_train_y.dims)
print()

r0_train = {'x':r0_train_x,'y':r0_train_y}
r0_val = {'x':r0_val_x,'y':r0_val_y}
r0_test = {'x':r0_test_x,'y':r0_test_y}
r0 = {'train':r0_train, 'val':r0_val,'test':r0_test}


r1_train_x = xr.concat([yr2019_x,yr2020_x,yr2021_x_split,yr2022_x_split],
                        data_vars='all',
                        dim='time')

r1_train_y = xr.concat([yr2019_y,yr2020_y,yr2021_y_split,yr2022_y_split],
                        data_vars='all',
                        dim='time')

print('r1 shapes')
print(r1_train_x.dims)
print(r1_train_y.dims)
print()

r1_train = {'x':r1_train_x,'y':r1_train_y}
r1_val = {'x':r1_val_x,'y':r1_val_y}
r1_test = {'x':r1_test_x,'y':r1_test_y}
r1 = {'train':r1_train,'val':r1_val,'test':r1_test}

r2_train_x = xr.concat([yr2019_x,yr2020_x_split,yr2021_x,yr2022_x_split],
                        data_vars='all',
                        dim='time')

r2_train_y = xr.concat([yr2019_y,yr2020_y_split,yr2021_y,yr2022_y_split],
                        data_vars='all',
                        dim='time')

print('r2 shapes')
print(r2_train_x.dims)
print(r2_train_y.dims)

r2_train = {'x':r2_train_x,'y':r2_train_y}
r2_val = {'x':r2_val_x,'y':r2_val_y}
r2_test = {'x':r2_test_x,'y':r2_test_y}
r2 = {'train':r2_train,'val':r2_val,'test':r2_test}


pickle.dump(r0,open('/scratch/bmac87/rotations/'+f_hour+'/r0.pkl','wb'))
pickle.dump(r1,open('/scratch/bmac87/rotations/'+f_hour+'/r1.pkl','wb'))
pickle.dump(r2,open('/scratch/bmac87/rotations/'+f_hour+'/r2.pkl','wb'))
