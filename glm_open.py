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

"""
This file generates netcdf files of a list of the lightning flashes from the GLM sensor,
for the entire GLM domain.  You select which satellite and year you would like 
to process.  The data are stored based on year and julian day. 
"""

def main():

    #create command line arguments for the satellite and year
    parser = argparse.ArgumentParser()
    parser.add_argument('--sat',type=str,default='G16',help='The model initialization time')
    parser.add_argument('--yr',type=int,default=1,help='Slurm array index for processing the years')
    args = vars(parser.parse_args())

    #determine which year to process
    sat = args['sat']
    if args['yr']==1:
        yr='2019'
    elif args['yr']==2:
        yr='2020'
    elif args['yr']==3:
        yr='2021'
    elif args['yr']==4:
        yr='2022'
    else:
        yr='2023'

    #generate a list of the julian days. add 366 if it is a leap year
    days = []
    for i in range(1,366):
        days.append(f"{i:03}")
    if yr=='2020':
        days.append('366')
    
    #book keeping, print the satellite and the year. generate a 
    #dump file for the error messages
    print('satellite_',sat,'_year_',yr)
    err_dump = open('../dump_files/err_dump_'+sat+'_'+yr+'_'+'.txt','w')

    #declare the coordinates you need from the glm data raw files
    coords = ['flash_lat','flash_lon','flash_id','flash_time_offset_of_first_event']
    ltg_vars = ['flash_area','flash_energy','flash_quality_flag']
    df_cols = np.concatenate([coords,ltg_vars])

    #loop through each julian day
    for day in days:

        #declare the directory
        glm_dir = '/ourdisk/hpc/ai2es/bmac87/OG_datasets/GLM/'+sat+'/'+yr+'/'+day+'/'

        #get a list of files for each day, if the day doesn't exist document
        #it and move on
        try:
            files = sorted(os.listdir(glm_dir))
            
        except FileNotFoundError:
            err_dump.write('file_not_found: '+ glm_dir)
            continue

        #track if it is the first file or not
        first_file=True        
        
        #loop through each GLM file
        for i,file in enumerate(files):

            #create the save directory, saving all of the flashes per day 
            save_dir = '/ourdisk/hpc/ai2es/bmac87/OG_datasets/GLM/'+sat+'/flashes/'+yr+'/'

            #create the directory if it doesn't exist
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            #generate the file name
            fname = yr+'_'+day+'_'+sat+'.nc'

            #if the file already exists, go to the next one
            if os.path.isfile(save_dir+fname):
                print(fname, ' already exists')
                continue

            #book keeping for progress
            if i%200==0:
                print('satellite_',sat,'_year_',yr,'_day_'+day,str(i)+'/'+str(len(files)))

            #load the data from the netcdf file
            try:
                ds = xr.open_dataset(glm_dir+file,engine='netcdf4')
                ds = ds[ltg_vars] #get the flash information using the dataset variables
                df = ds.to_dataframe() #convert to dataframe
                df.index = df['flash_time_offset_of_first_event']#set the indices
                df = df[df_cols] #subset the flash information with Lat/Lon

                #if the file for the day do not concatenate 
                if first_file:
                    flashes_df = df
                    first_file=False
                
                #otherwise concatenate the data
                else:
                    flashes_df = pd.concat([flashes_df,df],axis=0)

            #catch the error, but continue on to the next file if it is corrupt
            except (KeyError, ValueError):
                err_dump.write('data not available: '+ glm_dir + file)
                continue
        #end for loop of files within a day 

        if first_file==False:

            #sort the data
            flashes_df.sort_index(axis=0,inplace=True)

            #remove any QC errors
            flashes_df = flashes_df.loc[flashes_df['flash_quality_flag']==0]
            
            #convert to xr dataset for storage into .nc files
            flashes_xr = flashes_df.to_xarray()
            flashes_xr.to_netcdf(save_dir+yr+'_'+day+'_'+sat+'.nc',engine='netcdf4')

            #clear out the memory 
            del ds, df, flashes_df, flashes_xr

        else:
            continue
    #end for loop of days

if __name__=="__main__":
    main()
