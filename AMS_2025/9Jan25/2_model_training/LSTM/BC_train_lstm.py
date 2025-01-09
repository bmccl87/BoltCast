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
from BC_parser_LSTM import *
from BC_convLSTM_depth import *
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
    fname = 'BC_LSTM_rot_%s_LR_%s_lstm_deep_%s_conv_deep_%s_conv_size_%s_stride_%s_epochs_%s_'%(args.rotation, 
                                                                                        lrate, 
                                                                                        args.lstm_deep,
                                                                                        args.conv_deep, 
                                                                                        args.conv_size,
                                                                                        args.stride,
                                                                                        args.epochs)

    # Label
    if args.label is None:
        label_str = ""
    else:
        print("label not none")
        label_str = "_%s_"%args.label
    
    #L2 regularization
    if args.L2_reg>0.0:
        L2_str = "_L2reg_%s"%(args.L2_reg)
    else: 
        L2_str = ""
    
    #dropout
    dropout_str="_dropout_%s"%(args.dropout)

    #convolutional information
    conv_str = '_conv_%s_'%args.activation_conv
    last_str = '_last_%s_'%args.activation_last

    # Put it all together, including #of training folds and the experiment rotation
    return fname+label_str+L2_str+dropout_str+conv_str+last_str

def execute_exp(args=None, multi_gpus=False):

    #Check the arguments
    if args is None:
        parser = create_parser()
        args = parser.parse_args([])

    # Scale the batch size with the number of GPUs
    if multi_gpus > 1:
        args.batch = args.batch*multi_gpus
    print('Batch size', args.batch)

    ####################################################
    # Output file base and pkl file
    fbase = generate_fname(args)
    print(fbase)
    fname_out = "%s_results.pkl"%(fbase)
    print(fname_out)

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

    if args.load_data:
        print('loading the data in BC_train_LSTM.py')
        ds_train, ds_val, ds_test = load_data_scratch(rotation=args.rotation, 
                                                base_dir=args.data_path)
    if args.build_model:
        print('building the model')
        model = create_LSTM(args)
        print(model.summary())
        
    # Perform the experiment?
    if args.nogo:
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
    
    # Save results
    results['fname_base'] = fbase
    if os.path.isdir(args.results_path)==False:
        os.makedirs(args.results_path)
    with open(args.results_path+'%s.pkl'%(fbase), "wb") as fp:
        pickle.dump(results, fp)
    
    # Save model
    if args.save_model:
        print('saving the model')
        model_path = args.results_path+'models/'
        if os.path.isdir(model_path)==False:
            os.makedirs(model_path)
        model.save(model_path+'%s_model.keras'%(fbase))
    wandb.finish()

    return model

if __name__ == "__main__":

    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

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