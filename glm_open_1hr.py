import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import argparse
import pickle

"""
This file generates netcdf files of a list of the lightning flashes from the GLM sensor,
for the entire GLM domain.  You select which satellite and year you would like 
to process.  The data are stored based on year and julian day. 
"""

def main():

    sat='G16'
    yrs = ['2019','2020','2021','2022','2023','2024']

    err_file = open('glm_processing_error_'+sat+'.txt','w')
    
    #declare the coordinates you need from the glm data raw files
    coords = ['flash_lat','flash_lon','flash_id','flash_time_offset_of_first_event']
    ltg_vars = ['flash_area','flash_energy','flash_quality_flag']
    df_cols = np.concatenate([coords,ltg_vars])

    save_dir = '/ourdisk/hpc/ai2es/bmac87/OG_datasets/GLM/'+sat+'/flashes_1hr_df/'
    hours = ['00','01','02','03','04','05','06','07','08','09','10','11','12',
            '13','14','15','16','17','18','19','20','21','22','23']

    for yr in yrs: 
        #build the array of julian days. add 366 for leap year
        j_days = []
        for i in range(1,366):
            j_days.append(f"{i:03}")
        if yr=='2020' or yr=='2024':
            j_days.append('366')

        #loop through the files
        for d,day in enumerate(j_days):
            if d>=0:
                try:
                    og_nc_dir = '/ourdisk/hpc/ai2es/bmac87/OG_datasets/GLM/'+sat+'/'+yr+'/'+day+'/'
                    og_nc_files = sorted(os.listdir(og_nc_dir))

                except Exception as e:
                    err_file.write(e)
                    err_file.write("")
                    err_file.write(sat+yr+day, ' not found')
                    err_file.write("")
                    continue

    #             for h,hr in enumerate(hours):
    #                 if h>=0 and not os.path.isfile(save_dir+sat+'_'+yr+'_'+day+'_'+hr+'.pkl'):
    #                     print(sat,yr,day,hr)
    #                     #OR_GLM-L2-LCFA_G16_s20190010000000_e20190010000200_c20190010000227.nc
    #                     hr_files = sorted(glob.glob(og_nc_dir+'OR_GLM-L2-LCFA_'+sat+'_s'+yr+day+hr+'*.nc'))
                        
    #                     for f,file in enumerate(hr_files):
    #                         try:
    #                             ds = xr.open_dataset(file)
    #                             ds = ds[ltg_vars] #get the flash information using the dataset variables
    #                             df = ds.to_dataframe() #convert to dataframe
    #                             df.index = df['flash_time_offset_of_first_event']#set the indices
    #                             df = df[df_cols] #subset the flash information with Lat/Lon
    #                             if f==0:
    #                                 flashes_df = df
    #                             else:
    #                                 flashes_df = pd.concat([flashes_df,df])
    #                         #end try
    #                         except Exception as e:
    #                             err_file.write('error during reading the data')
    #                             err_file.write("")
    #                             err_file.write(e)
    #                             err_file.write("")
                                
    #                         #end_except

    #                     #end_for_files_in_hour
    #                     if not os.path.isdir(save_dir):
    #                         os.mkdir(save_dir)
    #                     if len(hr_files)>0:
    #                         pickle.dump(flashes_df,open(save_dir+sat+'_'+yr+'_'+day+'_'+hr+'.pkl','wb'))
    #                         del flashes_df, df, ds
    #                     else:
    #                         err_file.write('no files for this hour')
    #                         err_file.write('')
    #                         err_file.write(sat+yr+day+hr)
    #                         err_file.write("")
    #                 #end_hours_test
    #             #end_for_hours
    #         #end_days_test
    #     #end_for_days
    # #end_for_yrs
    err_file.close()

if __name__=="__main__":
    main()
