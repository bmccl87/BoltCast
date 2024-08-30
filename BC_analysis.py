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

    print("hello world")

    results_dir = "../results/"
    files = os.listdir(results_dir)
    print(files)

    f_hour = 'f000'
    rot = 0


    visible_devices = tf.config.get_visible_devices('GPU') 
    n_visible_devices = len(visible_devices)
    print(n_visible_devices)
    tf.config.set_visible_devices([], 'GPU')
    print('GPU turned off')

    tf_ds, ds = load_test_data(rotation=rot,
                    f_hour=f_hour,
                    base_dir='/scratch/bmac87/dataset/')

    print('loading the model')
    model_dir = './models/BC_rot_0_LR_0.000010000_deep_3_nconv_1__model/'
    model =  tf.keras.models.load_model(model_dir)
    print(model)

    for input_layers, labels in tf_ds.take(1):
        print(input_layers.shape)
        print(labels.shape)

    print('print out the model output shape')
    model_output = model.predict(input_layers)
    print(model_output.shape)
    for i in range(30):
        
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(model_output[i,:,:])
        ax2.imshow(labels[i,:,:])
        plt.savefig('./test_images/test_'+str(i)+'.png')
        plt.close()

    plt.figure()
    plt.hist(model_output.ravel())
    plt.xlabel('prob of lightning')
    plt.ylabel('count')
    plt.savefig('./test_images/output_hist.png')
    plt.close()

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

    labels = ds['binary_ltg']

    tps = tp(labels.values.ravel(),model_output.ravel())
    fps = fp(labels.values.ravel(),model_output.ravel())
    fns = fn(labels.values.ravel(),model_output.ravel())

    #calc x,y of performance diagram 
    pods = tps/(tps + fns)
    srs = tps/(tps + fps)
    csis = tps/(tps + fns + fps)

    
    #plot it up  
    ax = make_performance_diagram_axis()
    ax.plot(np.asarray(srs),np.asarray(pods),'-s',color='dodgerblue',markerfacecolor='w',label='UNET')

    for i,t in enumerate(thresh):
        text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
        ax.text(np.asarray(srs)[i]+0.02,np.asarray(pods)[i]+0.02,text,path_effects=pe1,fontsize=9,color='white')

    plt.tight_layout()
    plt.savefig('gewitter_plot.png')
    plt.close()

if __name__=="__main__":
    main()