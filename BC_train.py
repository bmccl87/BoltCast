import sys
import argparse
import pickle
import pandas as pd
import wandb
import socket
import matplotlib.pyplot as plt
import shutil 
import os

import tensorflow as tf
from tensorflow import keras

#import BoltCast specific code
from BC_parser import *
from BC_unet_classifier2 import *
from BC_data_loader3 import * 



#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=18
plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
#################################################################

#################################################################
def check_args(args):
    '''
    Check that the input arguments are rational

    '''
    assert (args.lrate > 0.0 and args.lrate < 1), "Lrate must be between 0 and 1"
    assert (args.cpus_per_task is None or args.cpus_per_task > 1), "cpus_per_task must be positive or None"
    
 
    
#################################################################

def generate_fname(args):
    '''
    Generate the base file name for output files/directories.
    
    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.
    '''
    lrate = f"{args.lrate:09.9f}"
    fname = 'BC_rot_%s_LR_%s_deep_%s_nconv_%s_conv_size_%s_stride_%s_%s'%(args.rotation, lrate, args.deep, args.n_conv_per_step,args.conv_size,args.stride,args.fcst_hour)

    # Label
    if args.label is None:
        label_str = ""
    else:
        label_str = "%s_"%args.label

    if args.batch_normalization is False:
        bn_str = ""
    else:
        bn_str = "_BN_"
    
    if args.L2_reg is None:
        L2_str = ""
    else: 
        L2_str = "_L2reg_%s_"%(args.L2_reg)
        
    
    # Put it all together, including #of training folds and the experiment rotation
    return fname+label_str+bn_str+L2_str

def execute_exp(args=None, multi_gpus=False):

    #Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])

    # Scale the batch size with the number of GPUs
    if multi_gpus > 1:
        args.batch = args.batch*multi_gpus

    print('Batch size', args.batch)

    ####################################################
    # Create the TF datasets for training, validation, testing

    if args.verbose >= 3:
        print('Starting data flow')

    
    image_size=args.image_size[0:2]
    print('image_size')
    print(image_size)

    if args.load_data:
        #load the data
        print('loading the data in BC_train.py')
        ds_train, ds_val, ds_test = load_data(rotation=args.rotation, f_hour=args.fcst_hour,base_dir='/ourdisk/hpc/ai2es/bmac87/BoltCast/dataset/',batch=args.batch)

        if args.static:
            #load static tf datasets
            ds_train, ds_val, ds_test = load_static_data(rotation=args.rotation, f_hour=args.fcst_hour, base_dir='/ourdisk/hpc/ai2es/bmac87/BoltCast/dataset/',batch=args.batch)

        if args.static_pres:
            #load static and pres level datasets
            ds_train, ds_val, ds_test = load_static_pres_data(rotation=args.rotation, f_hour=args.fcst_hour, base_dir='/ourdisk/hpc/ai2es/bmac87/BoltCast/dataset/',batch=args.batch)
        
        if args.static_atm:
            #load static and atm datasets
            ds_train, ds_val, ds_test = load_static_atm_data(rotation=args.rotation, f_hour=args.fcst_hour, base_dir='/ourdisk/hpc/ai2es/bmac87/BoltCast/dataset/',batch=args.batch)

    ####################################################

    

    # Output file base and pkl file
    fbase = generate_fname(args)
    print(fbase)
    fname_out = "%s_results.pkl"%fbase
    print(fname_out)

    if args.build_model:
        print('building the model')
        print('image_size',image_size)
        model = create_unet(args,
                        image_size=image_size,
                        filters=args.conv_nfilters,
                        conv_size=args.conv_size,
                        pool_size=args.pool,
                        deep=args.deep,
                        n_conv_per_step=args.n_conv_per_step,
                        lrate=args.lrate,
                        loss=tf.keras.losses.BinaryCrossentropy(),#tensor flow loss function
                        metrics=tf.keras.metrics.BinaryCrossentropy(),#tensor flow metrics
                        padding=args.padding,#string, same,valid,etc.
                        strides=args.stride,#int, pixel stride
                        conv_activation=args.activation_conv,
                        last_activation=args.activation_last,
                        batch_normalization=args.batch_normalization,
                        dropout=args.spatial_dropout,
                        skip=args.skip)
        if args.static:
            model = create_unet_static(args,
                        image_size=image_size,
                        filters=args.conv_nfilters,
                        conv_size=args.conv_size,
                        pool_size=args.pool,
                        deep=args.deep,
                        n_conv_per_step=args.n_conv_per_step,
                        lrate=args.lrate,
                        loss=tf.keras.losses.BinaryCrossentropy(),#tensor flow loss function
                        metrics=tf.keras.metrics.BinaryCrossentropy(),#tensor flow metrics
                        padding=args.padding,#string, same,valid,etc.
                        strides=args.stride,#int, pixel stride
                        conv_activation=args.activation_conv,
                        last_activation=args.activation_last,
                        batch_normalization=args.batch_normalization,
                        dropout=args.spatial_dropout,
                        skip=args.skip)

        if args.static_atm:
            model = create_unet_static_atm(args,
                        image_size=image_size,
                        filters=args.conv_nfilters,
                        conv_size=args.conv_size,
                        pool_size=args.pool,
                        deep=args.deep,
                        n_conv_per_step=args.n_conv_per_step,
                        lrate=args.lrate,
                        loss=tf.keras.losses.BinaryCrossentropy(),#tensor flow loss function
                        metrics=tf.keras.metrics.BinaryCrossentropy(),#tensor flow metrics
                        padding=args.padding,#string, same,valid,etc.
                        strides=args.stride,#int, pixel stride
                        conv_activation=args.activation_conv,
                        last_activation=args.activation_last,
                        batch_normalization=args.batch_normalization,
                        dropout=args.spatial_dropout,
                        skip=args.skip)
        
        if args.static_pres:
            model = create_unet_static_pres(args,
                        image_size=image_size,
                        filters=args.conv_nfilters,
                        conv_size=args.conv_size,
                        pool_size=args.pool,
                        deep=args.deep,
                        n_conv_per_step=args.n_conv_per_step,
                        lrate=args.lrate,
                        loss=tf.keras.losses.BinaryCrossentropy(),#tensor flow loss function
                        metrics=tf.keras.metrics.BinaryCrossentropy(),#tensor flow metrics
                        padding=args.padding,#string, same,valid,etc.
                        strides=args.stride,#int, pixel stride
                        conv_activation=args.activation_conv,
                        last_activation=args.activation_last,
                        batch_normalization=args.batch_normalization,
                        dropout=args.spatial_dropout,
                        skip=args.skip)
                        
        print(model.summary())
      
    # Plot the model if the model is built
    if args.render and args.build_model:
        render_fname = '/scratch/bmac87/results/%s_model_plot.png'%fbase
        plot_model(model, to_file=render_fname, show_shapes=True, show_layer_names=True)

     # Perform the experiment?
    if args.nogo:
        # No!
        print("NO GO")
        return

    # Check if output file already exists
    if not args.force and os.path.exists(fname_out):

        # Results file does exist: exit
        print("File %s already exists"%fname_out)
        return

    #####
    # Start wandb
    run = wandb.init(project=args.project, name=fbase, notes=fbase, config=vars(args))

    # Log hostname
    wandb.log({'hostname': socket.gethostname()})

    # Log model design image
    if args.render:
        wandb.log({'model architecture': wandb.Image(render_fname)})

    #####
    # Callbacks
    cbs = []

    if args.early_stopping:
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True,
                                                        min_delta=args.min_delta, monitor=args.monitor)
        cbs.append(early_stopping_cb)
    
    cbs.append(tf.keras.callbacks.ModelCheckpoint(filepath='/scratch/bmac87/'+args.fcst_hour+'_rot_'+str(args.rotation)+'_checkpoint.model.keras',
                                                    monitor='val_loss',
                                                    mode='auto',
                                                    save_best_only=False,
                                                    save_freq='epoch'))

    # Weights and Biases
    wandb_metrics_cb = wandb.keras.WandbMetricsLogger()
    cbs.append(wandb_metrics_cb)

    if args.verbose >= 3:
        print('Fitting model')
    
    #train the model
    history = model.fit(ds_train,
                        batch_size = args.batch,
                        epochs=args.epochs,
                        use_multiprocessing=True, 
                        verbose=args.verbose>=2,
                        validation_data = ds_val,
                        callbacks=cbs)

    # Done training
    print('Done Training')
    # Generate results data
    results = {}
    results['history'] = history.history

    print('Predicting the model')
    for test_inputs, test_labels in ds_test.take(1):
        print(test_inputs.shape)
        print(test_labels.shape)
    
    test_results = model.predict(test_inputs)
    results['test_x'] = test_inputs
    results['test_y'] = test_labels
    results['test_prediction'] = test_results

    # Save results
    fbase = generate_fname(args)
    results['fname_base'] = fbase
    with open("/scratch/bmac87/results/%s_results.pkl"%(fbase), "wb") as fp:
        pickle.dump(results, fp)
    
    # Save model
    if args.save_model:
        print('saving the model')
        model.save("/scratch/bmac87/results/models/%s_model"%(fbase))

    wandb.finish()

    return model



if __name__ == "__main__":

    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    #n_physical_devices = 0

    if args.verbose >= 3:
        print('Arguments parsed')

    #GPU check
    visible_devices = tf.config.get_visible_devices('GPU') 
    n_visible_devices = len(visible_devices)
    print(n_visible_devices)
    print('GPUS:', visible_devices)
    if n_visible_devices > 0:
        for device in visible_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print('We have %d GPUs\n'%n_visible_devices)
    else:
        print('NO GPU')

    # Turn off GPU?
    # if not args.gpu or "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
    #     visible_devices = tf.config.get_visible_devices('GPU') 
    #     n_visible_devices = len(visible_devices)
    #     print(n_visible_devices)
    #     tf.config.set_visible_devices([], 'GPU')
    #     print('NO VISIBLE DEVICES!!!!')

    # Set number of threads, if it is specified
    if args.cpus_per_task is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)
    
    #lrate, n_conv, roation, conv_size,stride
    ids = { 1: 'f000',
        2: 'f024',
        3: 'f048',
        4: 'f072',
        5: 'f096',
        6: 'f120',
        7: 'f144',
        8: 'f168',
        9: 'f192'}
        
    
    exp_id = args.exp
    hypes = ids[exp_id]
    args.fcst_hour = hypes

    execute_exp(args, multi_gpus=n_visible_devices)