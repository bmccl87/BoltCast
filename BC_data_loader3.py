import argparse
import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

def load_data(rotation=0,
            f_hour='f000',
            base_dir='/scratch/bmac87/dataset/',
            batch=16):

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
    #  tf_train = tf_train.shuffle(buffer_size=500)
     tf_train = tf_train.batch(batch)
     tf_train = tf_train.cache()
     print('tf_train',tf_train)

     tf_val = tf.data.Dataset.from_tensor_slices((val_xr.norm_features.values,val_xr.binary_ltg.values))
    #  tf_val = tf_val.shuffle(buffer_size=30)
     tf_val = tf_val.batch(batch)
     tf_val = tf_val.cache()
     print('tf_val',tf_val)

     tf_test = tf.data.Dataset.from_tensor_slices((test_xr.norm_features.values,test_xr.binary_ltg.values))
    #  tf_test = tf_test.shuffle(buffer_size=30)
     tf_test = tf_test.batch(batch)
     tf_test = tf_test.cache()
     print('tf_test', tf_test)
     
     return tf_train, tf_val,tf_test

def load_static_data(rotation=0,
            f_hour='f000',
            base_dir='/ourdisk/hpc/ai2es/bmac87/BoltCast/dataset/',
            batch=16):

     ch_idx1=17
     ch_idx2=20
     print("loading static data")
     #
     
     test_xr = pickle.load(open(base_dir+'test/'+f_hour+'_xr_norm_r'+str(rotation)+'.pkl','rb'))
     time_test = test_xr['time']
     y = test_xr['lat'].values
     x = test_xr['lon'].values
     print(test_xr['n_channel'].values)
     temp_features = test_xr['norm_features'][:,:,:,ch_idx1:ch_idx2]
     temp_nchannel = test_xr['n_channel'].values[ch_idx1:ch_idx2]
     xr_tf_ds = xr.Dataset(data_vars=dict(
                                            norm_features=(["n_samples", "y", "x","n_channel"],temp_features.values),
                                            binary_ltg=(["n_samples", "y", "x"],test_xr['binary_ltg'].values)
                                            ),
                                    coords=dict(
                                        time=("n_samples",time_test.values),
                                        n_channel=("n_channel",temp_nchannel),
                                        lat=("y",y),
                                        lon=("x",x)
                                    ))

     tf_test = tf.data.Dataset.from_tensor_slices((xr_tf_ds.norm_features.values,xr_tf_ds.binary_ltg.values))
     tf_test = tf_test.batch(batch)

     train_xr = pickle.load(open(base_dir+'train/'+f_hour+'_xr_norm_r'+str(rotation)+'.pkl','rb'))
     temp_features = train_xr['norm_features'][:,:,:,ch_idx1:ch_idx2]
     time_train = train_xr['time'].values
     xr_tf_ds = xr.Dataset(data_vars=dict(
                                            norm_features=(["n_samples", "y", "x","n_channel"],temp_features.values),
                                            binary_ltg=(["n_samples", "y", "x"],train_xr['binary_ltg'].values)
                                            ),
                                    coords=dict(
                                        time=("n_samples",time_train),
                                        n_channel=("n_channel",temp_nchannel),
                                        lat=("y",y),
                                        lon=("x",x)
                                    ))

     tf_train = tf.data.Dataset.from_tensor_slices((xr_tf_ds.norm_features.values,xr_tf_ds.binary_ltg.values))
     tf_train = tf_train.batch(batch)

     val_xr = pickle.load(open(base_dir+'val/'+f_hour+'_xr_norm_r'+str(rotation)+'.pkl','rb'))
     temp_features = val_xr['norm_features'][:,:,:,ch_idx1:ch_idx2]
     time_val = val_xr['time'].values
     xr_tf_ds = xr.Dataset(data_vars=dict(
                                            norm_features=(["n_samples", "y", "x","n_channel"],temp_features.values),
                                            binary_ltg=(["n_samples", "y", "x"],val_xr['binary_ltg'].values)
                                            ),
                                    coords=dict(
                                        time=("n_samples",time_val),
                                        n_channel=("n_channel",temp_nchannel),
                                        lat=("y",y),
                                        lon=("x",x)
                                    ))

     tf_val = tf.data.Dataset.from_tensor_slices((xr_tf_ds.norm_features.values,xr_tf_ds.binary_ltg.values))
     tf_val = tf_val.batch(batch)

     return tf_train, tf_val, tf_test
     

def load_static_pres_data(rotation=0,
            f_hour='f000',
            base_dir='/ourdisk/hpc/ai2es/bmac87/BoltCast/dataset/',
            batch=16):

     ch_idx1=0
     ch_idx2=14

     test_xr = pickle.load(open(base_dir+'test/'+f_hour+'_xr_norm_r'+str(rotation)+'.pkl','rb'))
     time_test = test_xr['time'].values
     y = test_xr['lat'].values
     x = test_xr['lon'].values
     temp_nchannel = np.concatenate([test_xr['n_channel'].values[0:14],test_xr['n_channel'].values[17:20]])
     temp_features = np.concatenate([test_xr['norm_features'][:,:,:,ch_idx1:ch_idx2].values,test_xr['norm_features'][:,:,:,17:20].values],axis=3)

     xr_tf_ds = xr.Dataset(data_vars=dict(
                                            norm_features=(["n_samples", "y", "x","n_channel"],temp_features),
                                            binary_ltg=(["n_samples", "y", "x"],test_xr['binary_ltg'].values)
                                            ),
                                    coords=dict(
                                        time=("n_samples",time_test),
                                        n_channel=("n_channel",temp_nchannel),
                                        lat=("y",y),
                                        lon=("x",x)
                                    ))

     tf_test = tf.data.Dataset.from_tensor_slices((xr_tf_ds.norm_features.values,xr_tf_ds.binary_ltg.values))
     tf_test = tf_test.batch(batch)
     
     val_xr = pickle.load(open(base_dir+'val/'+f_hour+'_xr_norm_r'+str(rotation)+'.pkl','rb'))
     time_val = val_xr['time'].values
     temp_nchannel = np.concatenate([val_xr['n_channel'].values[0:14],val_xr['n_channel'].values[17:20]])
     temp_features = np.concatenate([val_xr['norm_features'][:,:,:,ch_idx1:ch_idx2].values,val_xr['norm_features'][:,:,:,17:20].values],axis=3)
     xr_tf_ds = xr.Dataset(data_vars=dict(
                                            norm_features=(["n_samples", "y", "x","n_channel"],temp_features),
                                            binary_ltg=(["n_samples", "y", "x"],val_xr['binary_ltg'].values)
                                            ),
                                    coords=dict(
                                        time=("n_samples",time_val),
                                        n_channel=("n_channel",temp_nchannel),
                                        lat=("y",y),
                                        lon=("x",x)
                                    ))
     tf_val = tf.data.Dataset.from_tensor_slices((xr_tf_ds.norm_features.values,xr_tf_ds.binary_ltg.values))
     tf_val =tf_val.batch(batch)

     train_xr = pickle.load(open(base_dir+'train/'+f_hour+'_xr_norm_r'+str(rotation)+'.pkl','rb'))
     time_train = train_xr['time'].values
     temp_nchannel = np.concatenate([train_xr['n_channel'].values[0:14],train_xr['n_channel'].values[17:20]])
     temp_features = np.concatenate([train_xr['norm_features'][:,:,:,ch_idx1:ch_idx2].values,train_xr['norm_features'][:,:,:,17:20].values],axis=3)
     xr_tf_ds = xr.Dataset(data_vars=dict(
                                            norm_features=(["n_samples", "y", "x","n_channel"],temp_features),
                                            binary_ltg=(["n_samples", "y", "x"],train_xr['binary_ltg'].values)
                                            ),
                                    coords=dict(
                                        time=("n_samples",time_train),
                                        n_channel=("n_channel",temp_nchannel),
                                        lat=("y",y),
                                        lon=("x",x)
                                    ))

     tf_train =  tf.data.Dataset.from_tensor_slices((xr_tf_ds.norm_features.values,xr_tf_ds.binary_ltg.values))
     tf_train = tf_train.batch(batch)

     print(xr_tf_ds)
     print(tf_train)

     return tf_train, tf_val, tf_test

def load_static_atm_data(rotation=0,
            f_hour='f000',
            base_dir='/ourdisk/hpc/ai2es/bmac87/BoltCast/dataset/',
            batch=16):

     ch_idx1=14
     ch_idx2=20
     print("loading static data")
     #
     
     test_xr = pickle.load(open(base_dir+'test/'+f_hour+'_xr_norm_r'+str(rotation)+'.pkl','rb'))
     time_test = test_xr['time']
     y = test_xr['lat'].values
     x = test_xr['lon'].values
     print(test_xr['n_channel'].values)
     temp_features = test_xr['norm_features'][:,:,:,ch_idx1:ch_idx2]
     temp_nchannel = test_xr['n_channel'].values[ch_idx1:ch_idx2]
     print(temp_nchannel)
     xr_tf_ds = xr.Dataset(data_vars=dict(
                                            norm_features=(["n_samples", "y", "x","n_channel"],temp_features.values),
                                            binary_ltg=(["n_samples", "y", "x"],test_xr['binary_ltg'].values)
                                            ),
                                    coords=dict(
                                        time=("n_samples",time_test.values),
                                        n_channel=("n_channel",temp_nchannel),
                                        lat=("y",y),
                                        lon=("x",x)
                                    ))

     tf_test = tf.data.Dataset.from_tensor_slices((xr_tf_ds.norm_features.values,xr_tf_ds.binary_ltg.values))
     tf_test = tf_test.batch(batch)

     train_xr = pickle.load(open(base_dir+'train/'+f_hour+'_xr_norm_r'+str(rotation)+'.pkl','rb'))
     temp_features = train_xr['norm_features'][:,:,:,ch_idx1:ch_idx2]
     time_train = train_xr['time'].values
     xr_tf_ds = xr.Dataset(data_vars=dict(
                                            norm_features=(["n_samples", "y", "x","n_channel"],temp_features.values),
                                            binary_ltg=(["n_samples", "y", "x"],train_xr['binary_ltg'].values)
                                            ),
                                    coords=dict(
                                        time=("n_samples",time_train),
                                        n_channel=("n_channel",temp_nchannel),
                                        lat=("y",y),
                                        lon=("x",x)
                                    ))

     tf_train = tf.data.Dataset.from_tensor_slices((xr_tf_ds.norm_features.values,xr_tf_ds.binary_ltg.values))
     tf_train = tf_train.batch(batch)

     val_xr = pickle.load(open(base_dir+'val/'+f_hour+'_xr_norm_r'+str(rotation)+'.pkl','rb'))
     temp_features = val_xr['norm_features'][:,:,:,ch_idx1:ch_idx2]
     time_val = val_xr['time'].values
     xr_tf_ds = xr.Dataset(data_vars=dict(
                                            norm_features=(["n_samples", "y", "x","n_channel"],temp_features.values),
                                            binary_ltg=(["n_samples", "y", "x"],val_xr['binary_ltg'].values)
                                            ),
                                    coords=dict(
                                        time=("n_samples",time_val),
                                        n_channel=("n_channel",temp_nchannel),
                                        lat=("y",y),
                                        lon=("x",x)
                                    ))

     tf_val = tf.data.Dataset.from_tensor_slices((xr_tf_ds.norm_features.values,xr_tf_ds.binary_ltg.values))
     tf_val = tf_val.batch(batch)

     return tf_train, tf_val, tf_test



def load_test_data(rotation=0,
            f_hour='f000',
            base_dir='/ourdisk/hpc/ai2es/bmac87/BoltCast/dataset/',
            batch=16):
     
     test_xr = pickle.load(open(base_dir+'test/'+f_hour+'_xr_norm_r'+str(rotation)+'.pkl','rb'))
     tf_test = tf.data.Dataset.from_tensor_slices((test_xr.norm_features.values,test_xr.binary_ltg.values))
     tf_test = tf_test.batch(batch)
     return tf_test, test_xr

if __name__=="__main__":

      
     visible_devices = tf.config.get_visible_devices('GPU') 
     n_visible_devices = len(visible_devices)
     print(n_visible_devices)
     tf.config.set_visible_devices([], 'GPU')
     print('NO VISIBLE DEVICES!!!!')

     load_static_atm_data(rotation=0,
               f_hour='f000',
               base_dir='/ourdisk/hpc/ai2es/bmac87/BoltCast/dataset/',
               batch=16)
