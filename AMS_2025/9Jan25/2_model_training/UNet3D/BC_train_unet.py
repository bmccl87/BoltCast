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
from BC_unet_3Dconv_only import *
from BC_data_loader5 import * 



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
    fname = 'BC_UNet_rot_%s_LR_%s_deep_%s_nconv_%s_conv_size_%s_stride_%s_epochs_%s_'%(args.rotation, lrate, args.deep, args.n_conv_per_step,args.conv_size,args.stride,args.epochs)

    # Label
    if args.label is None:
        label_str = ""
    else:
        print("label not none")
        label_str = "_%s_"%args.label

    if args.batch_normalization is False:
        bn_str = ""
    else:
        bn_str = "_BN_"
    
    if args.L2_reg>0.0:
        L2_str = "_L2reg_%s"%(args.L2_reg)
    else: 
        L2_str = ""
    
    if args.spatial_dropout>0.0:
        SD_str="_SD_%s"%(args.spatial_dropout)
    else:
        SD_str="_SD_%s"%(args.spatial_dropout)

    conv_str = '_conv_%s_'%args.activation_conv
    last_str = '_last_%s_'%args.activation_last

    # Put it all together, including #of training folds and the experiment rotation
    return fname+label_str+bn_str+L2_str+SD_str+conv_str+last_str

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

    if args.load_data:
        #load the data
        print('loading the data in BC_train.py')
        ds_train, ds_val, ds_test = load_data_scratch(rotation=args.rotation, 
                                                base_dir='/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/binary/')

    ####################################################
    # Output file base and pkl file
    fbase = generate_fname(args)
    print(fbase)
    fname_out = "%s_results.pkl"%fbase
    print(fname_out)

    if args.build_model:
        print('building the model')
        model = create_stacked_unet(args)
        print(model.summary())

    # Plot the model if the model is built
    if args.render and args.build_model:
        if os.path.isdir(args.results_path)==False:
            os.makedirs(args.results_path)

        render_fname = args.results_path+'%s_model_plot.png'%fbase
        plot_model(model, to_file=render_fname, show_shapes=True, show_layer_names=True)

    # Check if output file already exists
    if not args.force and os.path.exists(fname_out):
        # Results file does exist: exit
        print("File %s already exists"%fname_out)
        return

    #load the slurm environment variables into dictionary variable for documenting into wandb
    slurm_dict = {}
    slurm_dict['slurm_job_id'] = os.environ.get('SLURM_JOB_ID')
    slurm_dict['slurm_job_name'] = os.environ.get('SLURM_JOB_NAME')
    slurm_dict['slurm_job_account'] = os.environ.get('SLURM_JOB_ACCOUNT')
    slurm_dict['slurm_cpus_per_task'] = os.environ.get('SLURM_CPUS_PER_TASK')
    slurm_dict['slurm_nodelist'] = os.environ.get('SLURM_JOB_NODELIST')
    slurm_dict['slurm_partition'] = os.environ.get('SLURM_JOB_PARTITION')
    slurm_dict['slurm_num_nodes'] = os.environ.get('SLURM_JOB_NUM_NODES')
    slurm_dict['slurm_array_job_id'] = os.environ.get('SLURM_ARRAY_JOB_ID')
    slurm_dict['slurm_task_id'] = os.environ.get('SLURM_ARRAY_TASK_ID')

    #load the variables into dictionary for passing into wandb
    args_dict = vars(args)
    config_dict = {}
    for key in args_dict:
        config_dict[key] = args_dict[key]
    for key in slurm_dict:
        config_dict[key] = slurm_dict[key]

    #####
    # Start wandb
    wandb_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/wandb/'
    if os.path.isdir(wandb_dir)==False:
        os.makedirs(wandb_dir)
    run = wandb.init(dir=wandb_dir,
                    project=args.project, 
                    name=fbase, 
                    notes=fbase, 
                    config=config_dict)

    # Log hostname
    wandb.log({'hostname': socket.gethostname()})

    # Log the code
    wandb.run.log_code(".")

    # Log model design image
    if args.render:
        wandb.log({'model architecture': wandb.Image(render_fname)})
    del slurm_dict, config_dict, args_dict

    #####
    # Callbacks
    cbs = []

    if args.early_stopping:
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True,
                                                        min_delta=args.min_delta, monitor=args.monitor)
        cbs.append(early_stopping_cb)
    
    ckpt_dir = args.ckpt_path
    if os.path.isdir(ckpt_dir)==False:
        os.makedirs(ckpt_dir)
    ckpt_fname = fname_out[:-12]+'_checkpoint.model.keras'
    print('checkpoint info')
    print(ckpt_dir+ckpt_fname)
    cbs.append(tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_dir+ckpt_fname,
                                                    monitor='val_loss',
                                                    mode='auto',
                                                    save_best_only=False,
                                                    save_freq='epoch'))

    # Weights and Biases
    wandb_metrics_cb = wandb.keras.WandbMetricsLogger()
    cbs.append(wandb_metrics_cb)

    if args.verbose >= 3:
        print('Fitting model')
    
    # Perform the experiment?
    if args.nogo:
        # No!
        print("NO GO")
        return

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
    
    # Save results
    fbase = generate_fname(args)
    results['fname_base'] = fbase
    if os.path.isdir(args.results_path)==False:
        os.makedirs(args.results_path)
    with open(args.results_path+"%s_results.pkl"%(fbase), "wb") as fp:
        pickle.dump(results, fp)
    
    # Save model
    if args.save_model:
        print('saving the model')
        model_dir = args.results_path+'models/'
        if os.path.isdir(model_dir)==False:
            os.makedirs(model_dir)
        model.save(model_dir+"%s_model.keras"%(fbase))
    wandb.finish()

    return model

if __name__ == "__main__":

    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    if args.verbose >= 3:
        print('Arguments parsed')

    #GPU check
    visible_devices = tf.config.get_visible_devices('GPU') 
    n_visible_devices = len(visible_devices)
    print("number of  GPUs: ",n_visible_devices )
    print('GPU info:', visible_devices)
    if n_visible_devices > 0:
        for device in visible_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print('We have %d GPUs\n'%n_visible_devices)
    else:
        print('NO GPU')

    # Turn off GPU?
    if not args.gpu or "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        visible_devices = tf.config.get_visible_devices('GPU') 
        n_visible_devices = len(visible_devices)
        tf.config.set_visible_devices([], 'GPU')
        print('GPUs turned off')
    print()

    # Set number of threads, if it is specified
    if args.cpus_per_task is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)

    execute_exp(args, multi_gpus=n_visible_devices)