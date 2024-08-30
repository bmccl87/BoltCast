import argparse
import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

def load_data(rotation=0,
            f_hour='f000',
            base_dir='/scratch/bmac87/dataset/'):

     train_xr = pickle.load(open(base_dir+'train/'+f_hour+'_xr_norm_r'+str(rotation)+'.pkl','rb'))
     val_xr = pickle.load(open(base_dir+'val/'+f_hour+'_xr_norm_r'+str(rotation)+'.pkl','rb'))
     test_xr = pickle.load(open(base_dir+'test/'+f_hour+'_xr_norm_r'+str(rotation)+'.pkl','rb'))

     print('xarray_train',train_xr)
     print('xarray_val',val_xr)
     print('xarray_test',test_xr)
     print("")
     
     print('building tensorflow datasets')
     print(train_xr.norm_features.values.shape)
     tf_train = tf.data.Dataset.from_tensor_slices((train_xr.norm_features.values,train_xr.binary_ltg.values))
     tf_train = tf_train.batch(32)
     print('tf_train',tf_train)

     tf_val = tf.data.Dataset.from_tensor_slices((val_xr.norm_features.values,val_xr.binary_ltg.values))
     tf_val = tf_val.batch(32)
     print('tf_val',tf_val)

     tf_test = tf.data.Dataset.from_tensor_slices((test_xr.norm_features.values,test_xr.binary_ltg.values))
     tf_test = tf_test.batch(32)
     print('tf_test', tf_test)
     
     return tf_train, tf_val,tf_test

def load_test_data(rotation=0,
            f_hour='f000',
            base_dir='/scratch/bmac87/dataset/'):
     
     test_xr = pickle.load(open(base_dir+'test/'+f_hour+'_xr_norm_r'+str(rotation)+'.pkl','rb'))
     tf_test = tf.data.Dataset.from_tensor_slices((test_xr.norm_features.values,test_xr.binary_ltg.values))
     tf_test = tf_test.batch(32)
     return tf_test, test_xr

if __name__=="__main__":
     load_data(rotation=0,
               f_hour='f000',
               base_dir='/scratch/bmac87/dataset/')
