'''
Advanced Machine Learning, 2024

Argument parser needed by multiple programs.

Author: Andrew H. Fagg (andrewhfagg@gmail.com)
Modified by: Brandon T. McClung (bmac7167@ou.edu), 17 Sep 24
'''

import argparse

def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='BoltCast', fromfile_prefix_chars='@')

    # High-level info for WandB
    parser.add_argument('--project', type=str, default='BoltCast_LSTM', help='WandB project name')

    # Project configuration
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--force', action='store_true', help='Perform the experiment even if the it was completed previously')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    parser.add_argument('--load_data',action='store_true',default=False,help='Flag to load the data')
    parser.add_argument('--no-load_data',action='store_false',dest='load_data')
    parser.add_argument('--build_model',action='store_true',default=False,help='Flag to build the model')
    parser.add_argument('--no-build_model',action='store_false',dest='build_model')
    parser.add_argument('--rotation',type=int, default=0,help='The rotation for cross validation')
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--no-gpu', action='store_false', dest='gpu', help='Do not use the GPU')
    parser.add_argument('--image_size', nargs=4, type=int, default=[128,256,104], help="Size of input images (rows, cols, channels)")
    parser.add_argument('--results_path', type=str, default='/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/', help='Results directory')
    parser.add_argument('--ckpt_path',type=str,default='/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/model_checkpoints/',help='Model checkpoint directory')
    parser.add_argument('--render', action='store_true', default=False, help='Write model image')
    parser.add_argument('--save_model', action='store_true', default=True, help='Save a model file')
    parser.add_argument('--no-save_model', action='store_false', dest='save_model', help='Do not save a model file')
    parser.add_argument('--data_path',type=str,default='/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/10_folds_ds/')
    
    # Specific experiment configuration
    parser.add_argument('--label', type=str, default=None, help="Extra label to add to output files")
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lrate', type=float, default=0.00001, help="Learning rate")
    parser.add_argument('--loss',type=str,default='binary_cross_entropy',help='The loss function of the optimizer')
    
    parser.add_argument('--dropout',type=float,default=0.0,help='Amount of spatial dropout')
    parser.add_argument('--L2_reg',type=float,default=None,help='L2 Regularization rate')
    parser.add_argument('--early_stopping',action='store_true',default=False,help='Use Early Stopping')
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early termination")
    parser.add_argument('--monitor', type=str, default="val_loss", help="Metric to monitor for early termination")
    
    parser.add_argument('--batch', type=int, default=16, help="Training set batch size")
    parser.add_argument('--cache', type=str, default=None, help="Cache (default: none; RAM: specify empty string; else specify file")
    parser.add_argument('--shuffle', type=int, default=0, help="Size of the shuffle buffer (0 = no shuffle")
    
    # LSTM
    parser.add_argument('--conv_deep',type=int,default=2,help='How many 3D conv layers')
    parser.add_argument('--lstm_deep',type=int,default=2,help='How many ConvLSTM2D layers')
    parser.add_argument('--conv_size', type=int, default=3, help='Convolution filter size per layer')
    parser.add_argument('--pool',type=int,default=2,help='Pooling size for the Conv3D inputs')
    parser.add_argument('--stride',type=int,default=1,help='Stride pixels')
    parser.add_argument('--padding', type=str, default='same', help='Padding type for convolutional layers')
    parser.add_argument('--activation_conv', type=str, default='relu', help='Activation function for convolutional layers')
    parser.add_argument('--activation_last',type=str,default='sigmoid',help='Last activation function')
    parser.add_argument('--return_sequences',action='store_true',default=False,help='Whether or not to return the sequences from the LSTM layer')
    parser.add_argument('--no-return_sequences',action='store_false',dest='return_sequences')
    parser.add_argument('--return_state',action='store_true',default=False,help='Whether or not to return the state of the LSTM layer')
    parser.add_argument('--no-return_state',action='store_false',dest='return_state')

    return parser

