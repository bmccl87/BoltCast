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

#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=18
plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
#################################################################

pickle_dir = '/scratch/bmac87/results/hyper_parameter/pickles/'
model_dir = '/scratch/bmac87/results/hyper_parameter/models/'

visible_devices = tf.config.get_visible_devices('GPU') 
n_visible_devices = len(visible_devices)
print(n_visible_devices)
tf.config.set_visible_devices([], 'GPU')
print('GPU turned off')

lrate_files = [
    'BC_rot_0_LR_0.000010000_deep_3_nconv_3__results.pkl',
    'BC_rot_0_LR_0.000100000_deep_3_nconv_1__results.pkl',
    'BC_rot_0_LR_0.001000000_deep_3_nconv_1__results.pkl',
]

nconv_files = [
    'BC_rot_0_LR_0.000010000_deep_3_nconv_1__results.pkl',
    'BC_rot_0_LR_0.000010000_deep_3_nconv_2__results.pkl',
    'BC_rot_0_LR_0.000010000_deep_3_nconv_3_conv_size_2_stride_1__results.pkl'
]

conv_size_files = [
    'BC_rot_0_LR_0.000010000_deep_3_nconv_3_conv_size_2_stride_1__results.pkl',
    'BC_rot_0_LR_0.000010000_deep_3_nconv_3_conv_size_3_stride_1__results.pkl',
    'BC_rot_0_LR_0.000010000_deep_3_nconv_3_conv_size_4_stride_1__results.pkl'
]

plt.figure(figsize=(18,10))
subplot_idx = 1

param = 'val_accuracy'
param_1 = 'accuracy'
colors = ['blue','red','green']
c_idx=0
#make the lrate plot 
ax1 = plt.subplot(1,3,1)
for file in lrate_files:
    results = pickle.load(open(pickle_dir+file,'rb'))
    history = results['history']
    val_acc = history[param]
    trng_acc = history[param_1]
    ax1.plot(val_acc,color=colors[c_idx],label = 'Val: LR = '+ file[12:20])
    ax1.plot(trng_acc,color=colors[c_idx],linestyle='dashed',label='Trng: LR = '+file[12:20])
    ax1.legend()
    ax1.grid()
    ax1.set_ylim([.8,1])
    ax1.set_xlim([0,200])  
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Varied Learning Rate')
    c_idx=c_idx+1

c_idx=0
ax2 = plt.subplot(1,3,2)
for file in nconv_files:
    print(file)
    results = pickle.load(open(pickle_dir+file,'rb'))
    history = results['history']
    val_acc = history[param]
    trng_acc = history[param_1]
    ax2.plot(val_acc,color=colors[c_idx],label = 'Val: n_conv = '+ file[37:38])
    ax2.plot(trng_acc,color=colors[c_idx],linestyle='dashed',label = 'Trng: n_conv = '+ file[37:38])
    ax2.legend()
    ax2.grid()
    ax2.set_ylim([.8,1])
    ax2.set_xlim([0,200])  
    ax2.set_xlabel('Epochs')
    ax2.set_yticklabels([])
    ax2.set_title('Varied n_conv_per_step')
    c_idx=c_idx+1

c_idx=0
ax3 = plt.subplot(1,3,3)
for file in conv_size_files:
    results = pickle.load(open(pickle_dir+file,'rb'))
    history = results['history']
    val_acc = history[param]
    trng_acc = history[param_1]
    ax3.plot(val_acc,color=colors[c_idx],label ='conv_size = '+ file[48:50])
    ax3.plot(trng_acc,color=colors[c_idx],linestyle='dashed',label ='conv_size = '+ file[48:50])
    ax3.legend()
    ax3.grid()
    ax3.set_ylim([.8,1])
    ax3.set_xlim([0,200])  
    ax3.set_xlabel('Epochs')
    ax3.set_yticklabels([])
    ax3.set_title('Varied conv_size')
    c_idx=c_idx+1

plt.suptitle('Rotation 0, Forecast Hour 00 Hyper-parameters')
plt.savefig('hyper_parameter_paper.png')
plt.savefig('hyper_parameter_paper.pdf')
plt.close()