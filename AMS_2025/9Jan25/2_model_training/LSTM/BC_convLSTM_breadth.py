import argparse
from keras.layers import Input, Conv2D, Conv2DTranspose, Conv3D, ConvLSTM2D, AveragePooling2D, AveragePooling3D, UpSampling3D, TimeDistributed, Concatenate 
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import keras
import numpy as np
from BC_parser import *

def create_LSTM(args):
    print("creating the LSTM")

    lrate = args.lrate
    image_size = args.image_size
    padding = args.padding
    activation = args.activation_conv
    activation_last = args.activation_last
    conv_size = args.conv_size
    pool = args.pool

    input1 = Input(shape=(1,image_size[1],image_size[2],image_size[3]),
                                    dtype=tf.dtypes.float32,
                                    name='input_day1')

    conv1 = Conv3D(filters=16,
                        input_shape = input1.shape,
                        activation=activation,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_size = (1,conv_size,conv_size),
                        name='day1_conv3d_'+activation+'_'+padding)(input1)

    pool1 = AveragePooling3D(pool_size=(1,pool,pool),
                            name='day1_pooling')(conv1)

    input2 = Input(shape=(1,image_size[1],image_size[2],image_size[3]),
                    dtype=tf.dtypes.float32,
                    name='input_day2')

    conv2 = Conv3D(filters=16,
                        input_shape=input2.shape,
                        activation=activation,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_size=(1,conv_size,conv_size),
                        name='day2_conv3d_'+activation+'_'+padding)(input2)
    
    pool2 = AveragePooling3D(pool_size=(1,pool,pool),
                            name='day2_pooling')(conv2)

    input3 = Input(shape=(1,image_size[1],image_size[2],image_size[3]),
                            dtype=tf.dtypes.float32,
                            name='input_day3')
    
    conv3 = Conv3D(filters=16,
                        input_shape=input3.shape,
                        activation=activation,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_size=(1,conv_size,conv_size),
                        name='day3_conv3d_'+activation+'_'+padding)(input3)
    
    pool3 = AveragePooling3D(pool_size=(1,pool,pool),
                            name='day3_pooling')(conv3)

    input4 = Input(shape=(1,image_size[1],image_size[2],image_size[3]),
                    dtype=tf.dtypes.float32,
                    name='input_day4')

    conv4 = Conv3D(filters=16,
                        input_shape=input4.shape,
                        activation=activation,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_size=(1,conv_size,conv_size),
                        name='day4_conv3d_'+activation+'_'+padding)(input4)
    
    pool4 = AveragePooling3D(pool_size=(1,pool,pool),
                            name='day4_pooling')(conv4)

    ##########################################CONV_LSTM_Layers#######################################################
    print("building day1 convLSTM layer")
    cl_1, h1, c1 = ConvLSTM2D(filters=16, 
                                kernel_size=(4, 4), 
                                padding='same', 
                                return_sequences=True,
                                return_state=True,
                                name='clstm_day1')(pool1)

    print("building day2 convLSTM layer")
    cl_2, h2, c2 = ConvLSTM2D(filters=16, 
                                kernel_size=(4, 4),
                                padding='same', 
                                return_sequences=True,
                                return_state=True,name='clstm_day2')([pool2,h1,c1])
    
    print("building day3 convLSTM layer")
    cl_3, h3, c3 = ConvLSTM2D(filters=16, 
                                kernel_size=(4, 4), 
                                padding='same', 
                                return_sequences=True,
                                return_state=True,
                                name='clstm_day3')([pool3, h2, c2])

    print("building day4 convLSTM layer")
    cl_4, h4, c4 = ConvLSTM2D(filters=16, 
                                kernel_size=(4, 4), 
                                padding='same', 
                                return_sequences=True,
                                return_state=True,
                                name='clstm_day4')([pool4, h3, c3])
    ######################################################################################################################
    up1 = UpSampling3D(size=(1,pool,pool),
                        name='day1_up')(cl_1)
    print(up1.shape)
    up1 = tf.squeeze(up1,axis=1)
    print(up1.shape)
    
    de_conv_1 = Conv2D(filters=8,
                        input_shape=up1.shape,
                        activation=activation,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_size=(conv_size,conv_size),
                        name='day1_conv2d_'+activation+'_'+padding)(up1)

    de_conv_out1 = Conv2D(filters=1,
                        input_shape=de_conv_1.shape,
                        activation=activation_last,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_size=(2,2),
                        name='day1_output_'+activation_last+'_'+padding)(de_conv_1)
    
    up2 = UpSampling3D(size=(1,pool,pool),
                        name='day2_up')(cl_2)
    up2 = tf.squeeze(up2,axis=1)

    de_conv_2 = Conv2D(filters=8,
                        input_shape=up2.shape,
                        activation=activation,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_size=(conv_size,conv_size),
                        name='day2_conv2d_'+activation+'_'+padding)(up2)
    
    de_conv_out2 = Conv2D(filters=1,
                            input_shape=de_conv_2.shape,
                            activation=activation_last,
                            padding=padding,
                            dtype=tf.dtypes.float32,
                            kernel_size=(2,2),
                            name='day2_output_'+activation_last+'_'+padding)(de_conv_2)

    up3 = UpSampling3D(size=(1,pool,pool),
                        name='day3_up')(cl_3)
    up3 = tf.squeeze(up3,axis=1)

    de_conv_3 = Conv2D(filters=8,
                        input_shape=up3.shape,
                        activation=activation,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_size=(conv_size,conv_size),
                        name='day3_conv2d_'+activation+'_'+padding)(up3)

    de_conv_out3 = Conv2D(filters=1,
                            input_shape=de_conv_3.shape,
                            activation=activation_last,
                            padding=padding,
                            dtype=tf.dtypes.float32,
                            kernel_size=(2,2),
                            name='day3_output_'+activation_last+'_'+padding)(de_conv_3)

    up4 = UpSampling3D(size=(1,pool,pool),
                        name='day4_up')(cl_4)
    up4 = tf.squeeze(up4,axis=1)

    de_conv_4 = Conv2D(filters=8,
                        input_shape=up4.shape,
                        activation=activation,
                        padding=padding,
                        dtype=tf.dtypes.float32,
                        kernel_size=(conv_size,conv_size),
                        name='day4_conv2d_'+activation+'_'+padding)(up4)
    
    de_conv_out4 = Conv2D(filters=1,
                            input_shape=de_conv_4.shape,
                            activation=activation_last,
                            padding=padding,
                            dtype=tf.dtypes.float32,
                            kernel_size=(conv_size,conv_size),
                            name='day4_output_'+activation_last+'_'+padding)(de_conv_4)

    output_tensor = [de_conv_out1,de_conv_out2,de_conv_out3,de_conv_out4]

    #complete the model
    model = Model(inputs=[input1,input2,input3,input4],outputs=[de_conv_out1,de_conv_out2,de_conv_out3,de_conv_out4])
    opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)
    model.compile(optimizer=opt,loss='mse')
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
        