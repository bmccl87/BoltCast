import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os 
import shutil 
import glob
import numpy as np
import matplotlib.pyplot as plt
import pygrib
import pickle
import datetime
from util import boxbin 

#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=  18
plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
#################################################################

#set the satellite and year
sat = 'G16'
yr = '2022'

#build the array of julian days. add 366 for leap year
j_days = []
for i in range(1,366):
    j_days.append(f"{i:03}")
if yr=='2020':
    j_days.append('366')

#set the storage directory
glm_dir = '/ourdisk/hpc/ai2es/bmac87/GLM/%s/flashes/%s/'%(sat,yr)

#load the gfs grid
static_inputs = pickle.load(open('./Archive/static_inputs.pkl','rb'))
lat = static_inputs['lat']#2D
lon = static_inputs['lon']#2D

#add the next grid point so the binning algorithm 
#size is 256x128
bin_lat = np.concatenate([lat[:,0],[53.25]])
bin_lon = np.concatenate([lon[0,:],[298]])

#store the sorted grid
xedge = np.sort(bin_lon)
yedge = np.sort(bin_lat)

xmid = [] #Blank array
ymid = [] #Blank array

#calcuate the midpoints for the sorting algorithm
i=0
while(i < len(xedge)-1):
    xmid.append((xedge[i]+xedge[i+1])/2) #Calculate and append midpoints
    i+=1 
i=0
while(i < len(yedge)-1):
    ymid.append((yedge[i]+yedge[i+1])/2) #Calculate and append midpoints
    i+=1

data_list = []
time_list = []

#loop through the files
for i,day in enumerate(j_days):

    #bookkeeping/progress bar
    print(i,len(j_days),yr,sat,len(xmid),len(ymid))

    #create the file name
    glm_file = '%s_%s_%s.nc'%(yr,day,sat)

    #load the xarray dataset if the file exists, otherwise move to the next day 
    try:
        ds = xr.open_dataset(glm_dir+glm_file,engine='netcdf4')
    except FileNotFoundError:
        continue
    
    #convert the dataset to a dataframe
    df = ds.to_dataframe()

    hours = {
                1:['00:00:00','01:00:00','02:00:00'],
                2:['03:00:00','04:00:00','05:00:00'],
                3:['06:00:00','07:00:00','08:00:00'],
                4:['09:00:00','10:00:00','11:00:00'],
                5:['12:00:00','13:00:00','14:00:00'],
                6:['15:00:00','16:00:00','17:00:00'],
                7:['18:00:00','19:00:00','20:00:00'],
                8:['21:00:00','22:00:00','23:00:00']
    }

    if len(df)>0:#if there are flashes across the entire domain

        #get the flashes within the latitude bounds of CONUS
        df = df.loc[df['flash_lat']>=21.25]
        df = df.loc[df['flash_lat']<=53]

        #get the west and east flashes
        if sat=='G16':#get the flashes east of 100W on GLM-Goes 16
            df_sect = df.loc[df['flash_lon']>=-100]
            df_sect = df_sect.loc[df_sect['flash_lon']<=-62.25]

        else: #get the flashes west of 100W on GLM-Goes 16
            df_sect = df.loc[df['flash_lon']<-100]
            df_sect = df_sect.loc[df_sect['flash_lon']>=-126]

        if len(df_sect)>0:#there is lightning in the BoltCast domain

            #store the date times as a datetime index
            df_sect['date_time'] = df_sect.index

            #round the times to the nearest hour
            df_sect['hour'] = df_sect['date_time'].dt.floor('h')
            
            #get the day from the last lightning strike
            ltg_valid_day = df_sect['hour'].dt.day.iloc[-1]

            #get the month of the last lightning strike
            ltg_valid_month = df_sect['hour'].dt.month.iloc[-1]

            #store it to the 00Z
            ltg_valid_hour = 0

            #generate the new valid time
            valid_time = datetime.datetime(year=int(yr),day=ltg_valid_day,month=ltg_valid_month)

            #Randy's binning code returns a 128x256 array.  You need to transpose
            #and flip it to get the lightning in the right locations. This was 
            #compared against the original dataset 

            #get the number of lightning flashes per grid box
            C = boxbin(df_sect['flash_lon']+360,df_sect['flash_lat'],xedge,yedge,mincnt=0)
            C1 = np.flip(C.transpose(),axis=0)

            #generate the binary classification, with the right orientation
            #fill nans to zero
            BC_ltg_grid = (np.flip(C.transpose(),axis=0)>=1).filled(fill_value=0)

            #store the right orientation of FED into C
            C = C1
            
        else: #there is no lightning in the BoltCast domain
            BC_ltg_grid = np.zeros((128,256))
            C = BC_ltg_grid

            #store the domain wide times
            df['date_time'] = df.index

            #floor the times down to an hour
            df['hour'] = df['date_time'].dt.floor('h')

            #get the day and the month, and set the zero hour
            ltg_valid_day = df['hour'].dt.day.iloc[-1]
            ltg_valid_month = df['hour'].dt.month.iloc[-1]
            ltg_valid_hour = 0

            #generate the datetime object
            valid_time = datetime.datetime(year=int(yr),day=ltg_valid_day,month=ltg_valid_month)
            
    else:#there are no flashes at all, across the entire domain
        BC_ltg_grid = np.zeros((128,256))
        C = BC_ltg_grid

        #store the datetime information as datetime objects
        df['date_time'] = df.index

        #floor the hour, then get the last day and month of the data
        df['hour'] = df['date_time'].dt.floor('h')
        ltg_valid_day = df['hour'].dt.day.iloc[-1]
        ltg_valid_month = df['hour'].dt.month.iloc[-1]
        ltg_valid_hour = 0

        #generate the valid time
        valid_time = datetime.datetime(year=int(yr),day=ltg_valid_day,month=ltg_valid_month)
    
    #store the counts and binary classification into an xarray dataset
    ds1 = xr.Dataset(
                    data_vars=dict(binary_ltg = (["lat","lon"],BC_ltg_grid),
                                    FED = (["lat","lon"],C)),
                    coords=dict(lon=(["lon"],xmid),lat=(["lat"],ymid)),
                    attrs=dict(description="Binary classification and FED of GLM on GFS grid. 256x128.")
                )

    #append the new dataset to the list
    data_list.append(ds1)

    #append the valid times to the list
    time_list.append(valid_time)
    
    #close the original dataset
    ds.close()

#generate a new xarray dataset from the dataset list for each day
ds2 = xr.concat(data_list, data_vars='all', dim='time')
ds2 = ds2.assign_coords(time=time_list)
ds2 = ds2.sortby('time')

#save the file to netcdf form 
output_file = '/ourdisk/hpc/ai2es/bmac87/GLM/binary_classification/Binary_Classification_%s_%s.nc'%(sat,yr)
ds2.to_netcdf(output_file, engine='netcdf4')
ds2.close()