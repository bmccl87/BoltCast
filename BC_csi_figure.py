import numpy as np
import matplotlib.pyplot as plt
import pickle
import xarray as xr
import pandas as pd
import matplotlib

matplotlib.rcParams['axes.labelsize'] = 18 #fontsize in pts
matplotlib.rcParams['axes.titlesize'] = 18 
matplotlib.rcParams['xtick.labelsize'] = 14 
matplotlib.rcParams['ytick.labelsize'] = 14 
matplotlib.rcParams['legend.fontsize'] = 14 
matplotlib.rcParams['legend.facecolor'] = 'w' 
matplotlib.rcParams['savefig.transparent'] = False

def main():

    fcst_hours = ['f000','f024','f048','f072','f096','f120','f144','f168','f192']

    conv_size = 4
    scores = pd.read_csv('csi_max_'+str(conv_size)+'.txt')

    rots = [0,1,2]
    for rot in rots:

        rot_scores = scores[scores['rotation']==rot]
        
        max_csi = rot_scores['max_csi'].values
        avg_csi = rot_scores['avg_csi'].values

        max_pod = rot_scores['max_pod'].values
        avg_pod = rot_scores['avg_pod'].values

        max_sr = rot_scores['max_sr'].values
        avg_sr = rot_scores['avg_sr'].values

        plt.figure(figsize=(12,8))
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        days = [0,1,2,3,4,5,6,7,8]
        ax.plot(days,max_csi,label='max_csi',color='r',linewidth=3)
        ax.scatter(days,max_csi,color='r',s=50,marker='o')
        ax.plot(days,max_pod,label='max_pod',color='b',linewidth=3)
        ax.scatter(days,max_pod,color='b',s=50,marker='o')
        ax.plot(days,max_sr,label='max_sr',color='g',linewidth=3)
        ax.scatter(days,max_sr,color='g',s=50,marker='o')

        ax.plot(days,avg_csi,label='avg_csi',color='r',linestyle='dashed',linewidth=3,alpha=.35)
        ax.scatter(days,avg_csi,color='r',s=50,marker='s',alpha=.35)
        ax.plot(days,avg_pod,label='avg_pod',color='b',linestyle='dashed',linewidth=3,alpha=.35)
        ax.scatter(days,avg_pod,color='b',s=50,marker='s',alpha=.35)
        ax.plot(days,avg_sr,label='avg_sr',color='g',linestyle='dashed',linewidth=3,alpha=.35)
        ax.scatter(days,avg_sr,color='g',s=50,marker='s',alpha=.35)

        ax.grid()
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xticks(days)
        ax.set_xticklabels(fcst_hours)
        ax.set_xlabel('Forecast Hour')
        ax.set_ylabel('Metrics - CSI/POD/SR')
        ax.set_title('Metrics: Rotation: '+str(rot)+', Conv_Size: '+str(conv_size))

        plt.savefig('metric_vs_time_rot_'+str(rot)+'_conv_size_'+str(conv_size)+'.png')
        plt.savefig('metric_vs_time_rot_'+str(rot)+'_conv_size_'+str(conv_size)+'.pdf')
        plt.close()

if __name__=="__main__":
    main()