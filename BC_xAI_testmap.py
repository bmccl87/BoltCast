import pickle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.rcParams['axes.facecolor'] = [0.9,0.9,0.9] #makes a grey background to the axis face
matplotlib.rcParams['axes.labelsize'] = 14 #fontsize in pts
matplotlib.rcParams['axes.titlesize'] = 14 
matplotlib.rcParams['xtick.labelsize'] = 12 
matplotlib.rcParams['ytick.labelsize'] = 12 
matplotlib.rcParams['legend.fontsize'] = 12 
matplotlib.rcParams['legend.facecolor'] = 'w' 
matplotlib.rcParams['savefig.transparent'] = False

rot = 1
conv_size=4

map_res_f000 = pickle.load(open('map_res_rot_'+str(rot)+'_conv_size_'+str(conv_size)+'_f000.pkl','rb'))
map_res_f096 = pickle.load(open('map_res_rot_'+str(rot)+'_conv_size_'+str(conv_size)+'_f096.pkl','rb'))
map_res_f192 = pickle.load(open('map_res_rot_'+str(rot)+'_conv_size_'+str(conv_size)+'_f192.pkl','rb'))

labels = map_res_f000['labels']

f000_out = map_res_f000['model_output']
f096_out = map_res_f096['model_output']
f192_out = map_res_f192['model_output']

valid_times = map_res_f000['time'].values
lat = map_res_f000['lat']
lon = map_res_f000['lon']

nrows=2
ncols=2

for i in range(30):

    if i>=0:
        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,
                                subplot_kw={'projection': ccrs.PlateCarree()},
                                figsize=(10,8))

        axs[0,0].set_title('Model Output: f000')
        cb = axs[0,0].pcolormesh(lon,lat,np.squeeze(f000_out[i,:,:]))
        axs[0,0].add_feature(cfeature.COASTLINE)
        axs[0,0].add_feature(cfeature.STATES)
        plt.colorbar(cb,ax=axs[0,0])

        axs[0,1].set_title('True Labels')
        cb=axs[0,1].pcolormesh(lon,lat,labels[i,:,:])
        axs[0,1].add_feature(cfeature.STATES)
        axs[0,1].add_feature(cfeature.COASTLINE)
        plt.colorbar(cb,ax=axs[0,1])

        
        axs[1,0].set_title('Model Output: f096')
        cb=axs[1,0].pcolormesh(lon,lat,np.squeeze(f096_out[i,:,:]))
        axs[1,0].add_feature(cfeature.COASTLINE)
        axs[1,0].add_feature(cfeature.STATES)
        plt.colorbar(cb,ax=axs[1,0])

        axs[1,1].set_title('Model Output: f192')
        cb=axs[1,1].pcolormesh(lon,lat,np.squeeze(f192_out[i,:,:]))
        axs[1,1].add_feature(cfeature.COASTLINE)
        axs[1,1].add_feature(cfeature.STATES)
        plt.colorbar(cb,ax=axs[1,1])

        ts = pd.to_datetime(valid_times[i])
        plt.suptitle('Valid Time: ' + np.datetime_as_string(valid_times[i],unit='D')+' 00:00Z',fontsize=18)

        plt.savefig('./images/pred_true_map_rot_'+str(rot)+'_conv_size_'+str(conv_size)+'_'+np.datetime_as_string(valid_times[i],unit='D')+'.png')
        plt.savefig('./pdfs/pred_true_map_rot_'+str(rot)+'_conv_size_'+str(conv_size)+'_'+np.datetime_as_string(valid_times[i],unit='D')+'.pdf')
        plt.close()