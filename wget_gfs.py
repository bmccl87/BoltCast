import wget
from urllib.error import HTTPError
import os
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--init_time',type=str,default='00',help='The model initialization time')
    parser.add_argument('--fcst_hour',type=int,default=1,help='Slurm array index for downloading the fcst hours simultaneously')
    parser.add_argument('--download',action='store_true',default=False,help='Flag to download the data')

    args = vars(parser.parse_args())

    if args['fcst_hour']==1:
        f_hour='f180'
    elif args['fcst_hour']==2:
        f_hour='f183'
    elif args['fcst_hour']==3:
        f_hour='f186'
    elif args['fcst_hour']==4:
        f_hour='f189'
    else:
        f_hour='f192'
    
    
    """
    This file downloads the GFS data.  
    """
    #   'https://data.rda.ucar.edu/ds084.1/2019/20190101/gfs.0p25.2019010106.f180.grib2'

    init_hr = args['init_time']

    #declare the years of interest
    yrs = ['2023','2024']

    #declare the months
    mos = []
    for i in range(1,13):
        mos.append(f"{i:02}")
    print('months',mos)

    #build an array with the daily numbers
    julian_days = []

    yrs_dict = {}
    for yr in yrs:#for each year
        julian_day = 1 #start counting a julian day
        mos_dict = {}
        for mo in mos: #for each month
            days = []
            mo_jul = []
            if mo=='01' or mo=='03' or mo=='05' or mo=='07' or mo=='08' or mo=='10' or mo=='12':
                for t in range(1,32):
                    days.append(f"{t:02}")
                    julian_days.append(f"{julian_day:03}")
                    mo_jul.append(f"{julian_day:03}")
                    julian_day = julian_day+1
            elif mo=='02':
                if yr=='2020':
                    for t in range(1,30):
                        days.append(f"{t:02}")
                        julian_days.append(f"{julian_day:03}")
                        mo_jul.append(f"{julian_day:03}")
                        julian_day = julian_day+1
                else:
                    for t in range(1,29):
                        days.append(f"{t:02}")
                        julian_days.append(f"{julian_day:03}")
                        mo_jul.append(f"{julian_day:03}")
                        julian_day = julian_day+1
            else:
                for t in range(1,31):
                    days.append(f"{t:02}")
                    julian_days.append(f"{julian_day:03}")
                    mo_jul.append(f"{julian_day:03}")
                    julian_day = julian_day+1
            mos_dict.update({mo:days})
            mos_dict.update({mo+'_jul':mo_jul})

        yrs_dict.update({yr:mos_dict})

    #create the storage directory
    gfs_stor = '/ourdisk/hpc/ai2es/bmac87/OG_datasets/GFS/'+init_hr+'Z/'+f_hour+'/'
    if not os.path.exists(gfs_stor):
        os.makedirs(gfs_stor)

    for yr in yrs:
        for mo in mos:
            temp = yrs_dict[yr]
            days = temp[mo]
            print(yr, mo)
            for day in days:
                try:
                    gfs_url = 'https://data.rda.ucar.edu/ds084.1/'+yr+'/'+yr+mo+day+'/gfs.0p25.'+yr+mo+day+init_hr+'.'+f_hour+'.grib2'
                    fname = 'gfs.0p25.'+yr+mo+day+init_hr+'.'+f_hour+'.grib2'
                    if os.path.isfile(gfs_stor+fname):
                        pass
                    else:
                        if args['download']:
                            filename = wget.download(gfs_url,out=gfs_stor)
                except HTTPError:
                    print('no file','/gfs.0p25.'+yr+mo+day+init_hr+'.'+f_hour+'.grib2')

    # gfs_base_url = 'https://data.rda.ucar.edu/ds084.1/2019/20190101/gfs.0p25.2019010100.f180.grib2'

if __name__ == "__main__":
    main()





