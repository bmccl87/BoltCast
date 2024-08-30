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

    f_hour = 'f000'
    rot = 2
    conv_size=4


    visible_devices = tf.config.get_visible_devices('GPU') 
    n_visible_devices = len(visible_devices)
    print(n_visible_devices)
    tf.config.set_visible_devices([], 'GPU')
    print('GPU turned off')

    tf_ds, ds = load_test_data(rotation=rot,
                    f_hour=f_hour,
                    base_dir='/scratch/bmac87/dataset/')
    
    binary_ltg = ds['binary_ltg']
    # for i in range(30):
    #     plt.figure()
    #     plt.imshow(binary_ltg.values[i,:,:])
    #     print(ds['time'].values[i])
    #     plt.title(f_hour+' '+str(i))
    #     plt.savefig('test_'+str(i)+'_'+f_hour+'.png')
    #     plt.close()
    # print('binary_ltg_type',type(binary_ltg.values))
   

    pbase = 'BC_rot_'+str(rot)+'_LR_0.000010000_deep_3_nconv_3_conv_size_'+str(conv_size)+'_stride_1_'+f_hour+'_results.pkl'
    p_dir = '/scratch/bmac87/results/fcst_hour_runs/pickles/'
    results = pickle.load(open(p_dir+pbase,'rb'))
    model_output = results['test_prediction']
    for key in results:
        print(key)
    # print('model_output_type',type(np.asarray(model_output)))
    
    map_res = {'labels':binary_ltg.values,'model_output':np.asarray(model_output),'lat':ds['lat'],'lon':ds['lon'],'time':ds['time']}

    pickle.dump(map_res,open('map_res_rot_'+str(rot)+'_conv_size_'+str(conv_size)+'_'+f_hour+'.pkl','wb'))
    

    # for i in range(30):
        
    #     f, (ax1, ax2) = plt.subplots(1, 2)
    #     ax1.imshow(model_output[i,:,:])
    #     ax2.imshow(labels[i,:,:])
    #     plt.savefig('./images/test_'+str(i)+'.png')
    #     plt.close()

    # plt.figure()
    # plt.hist(model_output.ravel())
    # plt.xlabel('prob of lightning')
    # plt.ylabel('count')
    # plt.savefig('/images/output_hist.png')
    # plt.close()

    

if __name__=="__main__":
    main()