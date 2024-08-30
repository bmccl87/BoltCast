import sys
import argparse
import pickle
import pandas as pd
import wandb
import socket
import matplotlib.pyplot as plt
import shutil 
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from BC_data_loader2 import *
from BC_parser import *
import os
from gewitter_functions import *

#load contingency_table func
from gewitter_functions import get_contingency_table,make_performance_diagram_axis,get_acc,get_pod,get_sr,csi_from_sr_and_pod
#plot parameters that I personally like, feel free to make these your own.
import matplotlib
import matplotlib.patheffects as path_effects

#outlines for text 
pe1 = [path_effects.withStroke(linewidth=1.5,
                            foreground="k")]
pe2 = [path_effects.withStroke(linewidth=1.5,
                            foreground="w")]

matplotlib.rcParams['axes.facecolor'] = [0.9,0.9,0.9] #makes a grey background to the axis face
matplotlib.rcParams['axes.labelsize'] = 14 #fontsize in pts
matplotlib.rcParams['axes.titlesize'] = 14 
matplotlib.rcParams['xtick.labelsize'] = 12 
matplotlib.rcParams['ytick.labelsize'] = 12 
matplotlib.rcParams['legend.fontsize'] = 12 
matplotlib.rcParams['legend.facecolor'] = 'w' 
matplotlib.rcParams['savefig.transparent'] = False


def main():

    visible_devices = tf.config.get_visible_devices('GPU') 
    n_visible_devices = len(visible_devices)
    print(n_visible_devices)
    tf.config.set_visible_devices([], 'GPU')
    print('GPU turned off')  

    results_dir = "/ourdisk/hpc/ai2es/bmac87/results/fcst_hour_runs/pickles/"
    files = os.listdir(results_dir)
    print(files)

    rotations = [1]
    conv_size=16
    csi_max = open('csi_max_'+str(conv_size)+'.txt','w')
    header = 'rotation,f_hour,max_csi,avg_csi,min_csi,max_pod,avg_pod,min_pod,max_sr,avg_sr,min_sr'
    csi_max.write(header)
    csi_max.write('\n')


    

    ax_idx_r = 0
    ax_idx_c = 0
   
    hours = ['f000','f024','f048','f072','f096','f120','f144','f168','f192']
    fig, axes = plt.subplots(3,3,figsize=(20,20))
    i=0
    for f_hour in hours:

        colors = ['red','blue','green']
    
        #plot it up  
        axes[ax_idx_r,ax_idx_c] = make_performance_diagram_axis(axes[ax_idx_r,ax_idx_c])

        for rot in rotations:
            tf_ds, ds = load_test_data(rotation=rot,
                            f_hour=f_hour,
                            base_dir='/scratch/bmac87/dataset/')

            if conv_size!=16:
                pickle_dir = '/ourdisk/hpc/ai2es/bmac87/results/fcst_hour_runs/pickles/BC_rot_'+str(rot)+'_LR_0.000010000_deep_3_nconv_3_conv_size_'+str(conv_size)+'_stride_1_'+f_hour+'_results.pkl'
            else:
                pickle_dir = '/scratch/bmac87/results/BC_rot_'+str(rot)+'_LR_0.000010000_deep_3_nconv_3_conv_size_'+str(conv_size)+'_stride_1_'+f_hour+'16_conv_reulu__results.pkl'
            results = pickle.load(open(pickle_dir,'rb'))
            for key in results: 
                print(key)
            model_output = np.asarray(results['test_prediction'])
            
            labels = np.asarray(results['test_y'])

            #probability threholds 
            thresh = np.arange(0.05,1.05,0.05)
            plt.figure()
            plt.hist(model_output.ravel(),bins=thresh,label='Predictions')
            plt.hist(labels.ravel(),bins=[0,.05,.95,1],label='Labels')
            plt.xlabel('Prob. Threshold (decimal)')
            plt.ylabel('Num. Pixels per Threshold')
            plt.legend()
            plt.savefig('./pdfs/thresh_dist_rot_'+str(rot)+'_conv_size_'+str(conv_size)+'_'+f_hour+'.pdf')
            plt.savefig('./images/thresh_dist_rot_'+str(rot)+'_conv_size_'+str(conv_size)+'_'+f_hour+'.png')
            plt.close()


            #statsitcs we need for performance diagram 
            tp = tf.keras.metrics.TruePositives(thresholds=thresh.tolist())
            fp = tf.keras.metrics.FalsePositives(thresholds=thresh.tolist())
            fn = tf.keras.metrics.FalseNegatives(thresholds=thresh.tolist())

            #get performance diagram line by getting tp,fp and fn 
            tp.reset_state()
            fp.reset_state()
            fn.reset_state()

            tps = tp(labels.ravel(),model_output.ravel())

            fps = fp(labels.ravel(),model_output.ravel())

            fns = fn(labels.ravel(),model_output.ravel())

            #calc x,y of performance diagram 
            pods = tps/(tps + fns)
            srs = tps/(tps + fps)
            csis = tps/(tps + fns + fps)

            print(np.asarray(srs))
            string = str(rot)+','+f_hour+','+str(np.max(np.asarray(csis)))+','+str(np.mean(np.asarray(csis)))+','+str(np.min(np.asarray(csis)))+','+str(np.max(np.asarray(pods)))+','+str(np.mean(np.asarray(pods)))+','+str(np.min(np.asarray(pods)))+','+str(np.nanmax(np.asarray(srs)))+','+str(np.nanmean(np.asarray(srs)))+','+str(np.nanmin(np.asarray(srs)))

            csi_max.write(string)
            csi_max.write('\n')
            
            axes[ax_idx_r,ax_idx_c].plot(np.asarray(srs),np.asarray(pods),'-s',color=colors[rot],markerfacecolor='w',label='Rot: '+str(rot))

            for i,t in enumerate(thresh):
                if i==5:
                    text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
                    axes[ax_idx_r,ax_idx_c].text(np.asarray(srs)[i]+0.02,np.asarray(pods)[i]+0.02,text,path_effects=pe1,fontsize=9,color='white')
                if i==10:
                    text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
                    axes[ax_idx_r,ax_idx_c].text(np.asarray(srs)[i]+0.02,np.asarray(pods)[i]+0.02,text,path_effects=pe1,fontsize=9,color='white')
                if i==15:
                    text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
                    axes[ax_idx_r,ax_idx_c].text(np.asarray(srs)[i]+0.02,np.asarray(pods)[i]+0.02,text,path_effects=pe1,fontsize=9,color='white')

        axes[ax_idx_r,ax_idx_c].legend()
        axes[ax_idx_r,ax_idx_c].set_title(f_hour)
        ax_idx_c=ax_idx_c+1
        if f_hour=='f048':
            ax_idx_c=0
            ax_idx_r=1
        if f_hour=='f120':
            ax_idx_c=0
            ax_idx_r=2


    plt.tight_layout()
    plt.savefig('./images/gewitter_plot_subplot_all_conv_size_'+str(conv_size)+'.png')
    plt.savefig('./pdfs/gewitter_plot_subplot_all_conv_size_'+str(conv_size)+'.pdf')
    plt.close()
    csi_max.close()

if __name__=="__main__":
    main()