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
import os
from gewitter_functions import *
import xarray as xr

#load contingency_table func
from gewitter_functions import get_contingency_table,make_performance_diagram_axis,get_acc,get_pod,get_sr,csi_from_sr_and_pod
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

    # visible_devices = tf.config.get_visible_devices('GPU') 
    # n_visible_devices = len(visible_devices)
    # print(n_visible_devices)
    # tf.config.set_visible_devices([], 'GPU')
    # print('GPU turned off')

    rots = [0,1,2,3,4]

    for r,rot in enumerate(rots):
        if r>=1:
            model_dir = '/scratch/bmac87/BoltCast_scratch/results/UNet/all_rots/models/'
            model_file = 'BC_UNet_rot_%s_LR_0.000010000_deep_3_nconv_3_conv_size_4_stride_1_epochs_500__binary_batch_32_symmetric__SD_0.2_conv_relu__last_sigmoid__model.keras'%(rot)
            model = tf.keras.models.load_model(model_dir+model_file)

            print("loading the test data, tf and ds")
            data_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/binary/'
            test_tf = tf.data.Dataset.load(data_dir+'rot_%s_test.tf'%(rot))

            c=0
            print("loading the test_tf.take()")
            for inputs,labels in test_tf:
                if c==0:
                    inputs_all = inputs
                    labels_all = labels
                else:
                    inputs_all = np.concatenate([inputs_all,inputs],axis=0)
                    labels_all = np.concatenate([labels_all,labels],axis=0)
                c+=1

            print("running the model.predict(test_tf)")
            model_output = model.predict(test_tf)
            del test_tf

            dict_out = {'labels':np.float32(labels_all),
                            'model_output':np.float32(model_output)}

            print('saving off the output and labels')
            print('rotation: ',rot)
            save_dir = '/scratch/bmac87/BoltCast_scratch/results/UNet/all_rots/labels_outputs/'
            if os.path.isdir(save_dir)==False:
                os.makedirs(save_dir)
            fsave = 'UNet_symmetric_rot_%s_output.pkl'%(rot)
            pickle.dump(dict_out,open(save_dir+fsave,'wb'))
            del dict_out, model_output, inputs_all, labels_all

if __name__=="__main__":
    main()