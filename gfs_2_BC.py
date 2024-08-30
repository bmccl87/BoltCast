import numpy as np
import xarray as xr
import os 
import shutil 
import glob
import numpy as np
import matplotlib.pyplot as plt
import pygrib
import pickle
import argparse

"""
This file has code to downselect the GFS data to a certain domain, then save that off.  
"""

"""
This function gets the graupel mixing ratios in kg/kg from the gfs grib files. 
"""

def get_variable(label, levels, grbindx, CONUS):
    for i,level in enumerate(levels):
        if i==0:
            data_3d = grbindx.select(name=label,typeOfLevel='isobaricInhPa',level=level)[0].values[CONUS[0]:CONUS[1],CONUS[2]:CONUS[3]]
        else:
            data_3d = np.dstack([data_3d, 
                                    grbindx.select(name=label,typeOfLevel='isobaricInhPa',level=level)[0].values[CONUS[0]:CONUS[1],CONUS[2]:CONUS[3]]])
    return data_3d

def gfs_open():

    #create a parser for the initialization time and the forecast hour for easy HPC use
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_time',type=str,default='00',help='The model initialization time')
    parser.add_argument('--fcst_hour',type=int,default=2,help='Slurm array index for downloading the fcst hours simultaneously')
    args = vars(parser.parse_args())

    #declare the forecast hour based on the slurm array ID
    if args['fcst_hour']==1:
        f_hour='f147'
    elif args['fcst_hour']==2:
        f_hour='f150'
    elif args['fcst_hour']==3:
        f_hour='f153'
    elif args['fcst_hour']==4:
        f_hour='f156'
    elif args['fcst_hour']==5:
        f_hour='f159'
    elif args['fcst_hour']==6:
        f_hour='f162'
    elif args['fcst_hour']==7:
        f_hour='f165'
    elif args['fcst_hour']==8:
        f_hour='f168'
    elif args['fcst_hour']==9:
        f_hour='f171'
    elif args['fcst_hour']==10:
        f_hour='f174'
    elif args['fcst_hour']==11:
        f_hour='f177'
    elif args['fcst_hour']==12:
        f_hour='f180'
    elif args['fcst_hour']==13:
        f_hour='f183'
    elif args['fcst_hour']==14:
        f_hour='f186'
    elif args['fcst_hour']==15:
        f_hour='f189'
    else:
        f_hour='f192'
        
    #declare where the GFS grib files are stored
    gfs_dir = '/ourdisk/hpc/ai2es/bmac87/OG_datasets/GFS/'+args['init_time']+'Z/'+f_hour+'/'

    #declare the output directory
    gfs_clean_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast/GFS_processing/'+f_hour+'/'
        
    #these indices are used to downselect the CONUS domain
    north_idx=148
    south_idx=276
    west_idx=936
    east_idx=1192

    CONUS = [north_idx,south_idx,west_idx,east_idx]

    

    #these are the levels in hPa where we want the environmental information 
    levels = [200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,925,950,975,1000]

    #save of the 1-D lat and lon grids
    gfs_grid = pickle.load(open('./Archive/static_inputs.pkl','rb'))
    boltCast_lon_1d = gfs_grid['lon'][0,:]
    boltCast_lat_1d = gfs_grid['lat'][:,0]
    
    #declare the months to grab those files
    months = ['00','01','02','03','04','05','06','07','08','09','10','11','12']
    years = ['2019','2020','2021','2022','2023','2024']

    badFiles = []

    #get the files for the months and years
    for yr in years:
        for mo in months:
   
            #get the file list for each year and month. 
            file_list = sorted(glob.glob(gfs_dir+'/*'+yr+mo+'*.grib2'))

            #declare lists for the time, data, bad files
            timeList = []
            dataList = []

            print(yr,mo,f_hour)

            for file in file_list:
                

                try:
                    grbs = pygrib.open(file)
                    valid_time = grbs[1].validDate
                    timeList.append(valid_time)
                    
                    #get the variables across all of the levels
                    grbindx = pygrib.index(file,'name','typeOfLevel','level')
                    graupel_3d = get_variable('Graupel (snow pellets)',levels,grbindx,CONUS)
                    ice_3d = get_variable('Ice water mixing ratio',levels,grbindx,CONUS)
                    rain_3d = get_variable('Rain mixing ratio',levels,grbindx,CONUS)
                    snow_3d = get_variable('Snow mixing ratio',levels,grbindx,CONUS)
                    w_3d = get_variable('Geometric vertical velocity',levels,grbindx,CONUS)

                    #get the accumulated atmospheric variables
                    reflectivity = grbs.select(name='Maximum/Composite radar reflectivity',typeOfLevel='atmosphere',level=0)[0].values[north_idx:south_idx,west_idx:east_idx]
                    cape = grbs.select(name='Convective available potential energy',typeOfLevel='surface',level=0)[0].values[north_idx:south_idx,west_idx:east_idx]
                    lifted_idx = grbs.select(name='Surface lifted index',typeOfLevel='surface',level=0)[0].values[north_idx:south_idx,west_idx:east_idx]
                    precip_rate = grbs.select(name='Precipitation rate',typeOfLevel='surface',level=0)[0].values[north_idx:south_idx,west_idx:east_idx]
                    
                    ds = xr.Dataset(
                        data_vars = dict(
                            
                            graupel_q = (["lat","lon","levels"],graupel_3d),
                            ice_q = (["lat","lon","levels"],ice_3d),
                            rain_q = (["lat","lon","levels"],rain_3d),
                            snow_q = (["lat","lon","levels"],snow_3d),
                            
                            w = (["lat","lon","levels"],w_3d),

                            reflectivity = (["lat","lon"],reflectivity),
                            cape = (["lat","lon"],cape),
                            lifted_idx = (["lat","lon"],lifted_idx),
                            precip_rate = (["lat","lon"],precip_rate)
                        ),

                        coords = dict(
                            lon = (["lon"],boltCast_lon_1d),
                            lat = (["lat"],boltCast_lat_1d),
                            levels = (["levels"],levels)
                        ),

                        attrs = dict(
                            description="Inputs into BoltCast.",
                            mixing_ratio_units="kg/kg",
                            refl_units="dB",
                            cape_units="J/kg",
                            lifted_index_units="K",
                            precip_rate_units="kg/(m^2 s)",
                            w_units="m/s",
                            time="datetime in UTC/Zulu time zone",
                            lat_lon="degrees",
                            levels="Index 0 is 200mb, while index 19 is 1000mb.  In other words the first index is the highest level in the atmosphere."
                        )
                    )
                    dataList.append(ds)
                #end_try

                except Exception as e: 
                    print("Exception when down selecting GFS data")
                    print(valid_time)
                    print(file)
                    badFiles.append(file)
                    print(e)
                    print()
                    continue
                #end_except
            #end_for_files_per_month_year

            #build the final dataset from the dataList variable
            try:
                ds1 = xr.concat(dataList, data_vars='all', dim='time')
                ds1 = ds1.assign_coords(time=timeList)
                ds1 = ds1.sortby('time')

                #create the save directory and filename
                outdir = gfs_clean_dir
                if os.path.isdir(outdir) == False:
                    os.makedirs(outdir)
                fname = 'BC_GFS_%s%s.nc'%(yr,mo)
                print(outdir+fname)
                ds1.to_netcdf(outdir+fname,engine='netcdf4',mode='w')
                del ds1, dataList, timeList, ds

            except Exception as e:
                print("Exception when concatenating the data. Most likely, the files don't exist for:")
                print(mo, yr)
                continue

        #end_for_months
    #end_for_years

    print()
    print("Bad Files: ")
    print(badFiles)
    print()
    pickle.dump(badFiles,open('badFiles_GFS_downselect.pkl','wb'))

#end gfs_open()
        

if __name__=="__main__":
    gfs_open()
