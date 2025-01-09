import argparse
from keras.layers import Input, Conv2D, Conv2DTranspose, Conv3D, ConvLSTM2D, AveragePooling2D, AveragePooling3D, UpSampling3D, TimeDistributed, Concatenate 
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import keras
import numpy as np
from BC_parser_LSTM import *

def create_LSTM(args):
    print("creating the LSTM")

    lrate = args.lrate
    image_size = args.image_size
    padding = args.padding
    activation = args.activation_conv
    activation_last = args.activation_last
    conv_size = args.conv_size
    pool = args.pool
    conv_deep = args.conv_deep
    lstm_deep = args.lstm_deep
    ret_state = args.return_state
    ret_seq = args.return_sequences
    loss_str = args.loss
    num_filters = 16

    input_tensor = Input(shape=(image_size[0],image_size[1],image_size[2],image_size[3]),
                                    dtype=tf.dtypes.float32,
                                    name='input')

    tensor = Conv3D(filters=num_filters,
                        input_shape = input_tensor.shape,
                        activation=activation,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_size = (conv_size,conv_size,conv_size),
                        name='en_conv3d_'+activation+'_'+padding+'_input')(input_tensor)
    
    en_conv_count=1
    for d in range(conv_deep):
        tensor = Conv3D(filters=num_filters,
                        input_shape = input_tensor.shape,
                        activation=activation,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_size = (conv_size,conv_size,conv_size),
                        name='en_conv3d_'+activation+'_'+padding+'_'+str(en_conv_count))(tensor)

        tensor = AveragePooling3D(pool_size=(1,pool,pool),
                            name='en_pool_'+str(en_conv_count))(tensor)
        en_conv_count+=1
        num_filters=num_filters*2

    ##########################################CONV_LSTM_Layers#######################################################
    print("building the ConvLSTM layer")
    for d in range(lstm_deep):
        tensor, h_tensor, c_tensor = ConvLSTM2D(filters=16, 
                                kernel_size=(conv_size, conv_size), 
                                padding=padding, 
                                return_sequences=ret_seq,
                                return_state=ret_state,
                                input_shape = tensor.shape,
                                name='clstm_'+str(d))(tensor)

    ######################################################################################################################
    for d in range(conv_deep):
        tensor = UpSampling3D(size=(1,pool,pool),
                            name='de_up3D_'+str(d))(tensor)
        
        tensor = Conv3D(filters=8,
                            input_shape=tensor.shape,
                            activation=activation,
                            padding=padding,
                            dtype=tf.dtypes.float32,
                            kernel_size=(conv_size,conv_size,conv_size),
                            name='de_Conv3D_'+activation+'_'+padding+'_'+str(d))(tensor)
    
    output_tensor = Conv3D(filters=1,
                        input_shape=tensor.shape,
                        activation=activation_last,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_size=(conv_size,conv_size,conv_size),
                        name='output_Conv3D_'+activation+'_'+padding)(tensor)

    #complete the model
    model = Model(inputs=input_tensor,outputs=output_tensor)
    opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)
    if loss_str=='binary_cross_entropy':
        loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=opt,loss=loss)
    return model 

if __name__ == "__main__":
    print('BC_convLSTM.py main function')

    visible_devices = tf.config.get_visible_devices('GPU') 
    n_visible_devices = len(visible_devices)
    print(n_visible_devices)
    tf.config.set_visible_devices([], 'GPU')
    print('NO VISIBLE DEVICES!!!!')

    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    print(args)

    if args.build_model:
        print('building the model')
        model = create_LSTM(args)
        print(model.summary())
        plot_model(model, to_file='test_csltm.png', show_shapes=True, show_layer_names=True)
        