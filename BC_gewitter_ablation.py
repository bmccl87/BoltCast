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
    csi_max = open('csi_max_ablation.txt','w')

    results_dir = "../results/"
    files = os.listdir(results_dir)
    print(files)

    rotations = [0]

    conv_size=4

    visible_devices = tf.config.get_visible_devices('GPU') 
    n_visible_devices = len(visible_devices)
    print(n_visible_devices)
    tf.config.set_visible_devices([], 'GPU')
    print('GPU turned off')
   
    hours = ['f000']
    fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(10,8))

    i=0
    for f_hour in hours:

        colors = ['red','blue','green']
    
        #plot it up  
        ax = make_performance_diagram_axis(axes[0,0])

        for rot in rotations:

            pickle_dir = '/scratch/bmac87/results/sensitivity/'
            f_static = 'BC_rot_'+str(rot)+'_LR_0.000010000_deep_3_nconv_3_conv_size_'+str(conv_size)+'_stride_1_'+f_hour+'static__results.pkl'
            f_atm = 'BC_rot_'+str(rot)+'_LR_0.000010000_deep_3_nconv_3_conv_size_'+str(conv_size)+'_stride_1_'+f_hour+'static_atm__results.pkl'
            f_pres = 'BC_rot_'+str(rot)+'_LR_0.000010000_deep_3_nconv_3_conv_size_'+str(conv_size)+'_stride_1_'+f_hour+'static_pres__results.pkl'
            
            for i,file in enumerate([f_static,f_atm,f_pres]): 

                results = pickle.load(open(pickle_dir+file,'rb'))
                model_output = np.asarray(results['test_prediction'])
                
                labels = np.asarray(results['test_y'])
                
                #probability threholds 
                thresh = np.arange(0.05,1.05,0.05)
                
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

                string = 'rotation= '+str(rot)+' fcst hr= '+f_hour+ 'max csi= '+str(np.max(np.asarray(csis)))+' file: '+file
                csi_max.write(string)
                csi_max.write('\n')
                if i==0:
                    label='Static'
                    r_idx = 0
                    c_idx=1
                elif i==1:
                    label='Static + Atm'
                    r_idx = 1
                    c_idx=0
                else:
                    label = 'Static + Pres'
                    r_idx=1
                    c_idx=1
                ax.plot(np.asarray(srs),np.asarray(pods),'-s',color=colors[i],markerfacecolor='w',label=label)
    #           
                axes[r_idx,c_idx].hist(model_output.ravel(),bins=thresh,label='Output')
                axes[r_idx,c_idx].hist(labels.ravel(),bins=[0,.05,.95,1],label='Labels')
                axes[r_idx,c_idx].set_xlabel('Prob. Threshold (decimal)')
                axes[r_idx,c_idx].set_ylabel('Num. Pixels per Threshold')
                axes[r_idx,c_idx].set_title(label)
                axes[r_idx,c_idx].legend()


                # for i,t in enumerate(thresh):
                #     if i==5:
                #         text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
                #         axes[ax_idx_r,ax_idx_c].text(np.asarray(srs)[i]+0.02,np.asarray(pods)[i]+0.02,text,path_effects=pe1,fontsize=9,color='white')
                #     if i==10:
                #         text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
                #         axes[ax_idx_r,ax_idx_c].text(np.asarray(srs)[i]+0.02,np.asarray(pods)[i]+0.02,text,path_effects=pe1,fontsize=9,color='white')
                #     if i==15:
                #         text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
                #         axes[ax_idx_r,ax_idx_c].text(np.asarray(srs)[i]+0.02,np.asarray(pods)[i]+0.02,text,path_effects=pe1,fontsize=9,color='white')
    ax.legend()
    plt.tight_layout()
    plt.savefig('./images/gewitter_sensitivity.png')
    plt.savefig('./pdfs/gewitter_sensitivity.pdf')
    plt.close()
    csi_max.close()
if __name__=="__main__":
    main()