import argparse
import numpy as np
import tensorflow as tf
import keras
from keras.layers import InputLayer, Dense, Activation, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model

from BC_parser import *

def create_unet(args,
                image_size=[128,256,1],
                filters=[8,16,32],
                conv_size=2,
                pool_size=2,
                deep=4,
                n_conv_per_step=2,
                lrate=.0001,
                n_types=1,
                loss='binary-cross-entropy',#string or tensor flow object
                metrics='accuracy', #string or tensor flow object 
                padding='same', #string
                strides=1, #int
                conv_activation='elu', #string, consider relu
                last_activation='sigmoid', #predicting the probability
                batch_normalization=False, 
                dropout=0.0,
                skip=False): 

    '''
    This functions builds a U-Net for BoltCast.  It builds down the number of layers 
    determined by the deep parameter.  It performs MaxPooling after each convolution block 
    described by the n_conv_step parameter.  Activation functions throughout the U are 
    provided, along with the activation function for the last layer.  Strides, padding, 
    loss functions, and metrics are provided too.  The filters is a list of the number 
    of filters to use, for every convolution down.  The filters are flipped along the way 
    up.  Flags/booleans for KSC data sets and regularization
    are also provided.  
    '''

    input_tensor = tf.keras.Input(shape=(image_size[0],image_size[1],20),
                                    dtype=tf.dtypes.float64,
                                    name='Input_Tensor')                                    
    tensor = input_tensor

    lambda_regularization = args.L2_reg
    #add l2 regularization if needed
    if lambda_regularization is not None:
        lambda_regularization = tf.keras.regularizers.l2(lambda_regularization)(tensor)
    
    skips = []
    if args.skip==True:
        skips.append(input_tensor)

    #go down the U-net
    for d in range(deep):
        for n in range(n_conv_per_step):

                #create the convolutional blocks
                tensor = Conv2D(filters=filters[d],
                    padding=args.padding,
                    kernel_size=(conv_size,conv_size),
                    strides=(strides,strides),
                    dtype=tf.dtypes.float64,
                    activation=conv_activation,
                    name='Conv_Down_%s_%s_%s'%(d,n,conv_activation),
                    kernel_regularizer=lambda_regularization,
                    use_bias=True)(tensor)
                
                if (args.skip==True) and (n==(n_conv_per_step-1)):
                    skips.append(tensor)

                #add batch_normalization if true
                if batch_normalization==True:
                    tensor = tf.keras.layers.BatchNormalization()(tensor)
                
                #add dropout if desired
                if dropout>0:
                    tensor = tf.keras.layers.SpatialDropout2D(dropout)(tensor)

        #add the max or average pooling layer
        tensor = tf.keras.layers.MaxPooling2D(pool_size=(pool_size,pool_size),name='MaxPool_%s'%d)(tensor)
    
    #learn at the bottom of the U 
    tensor = Conv2D(filters=filters[d]*2,
        padding=args.padding,
        kernel_size=(conv_size,conv_size),
        strides=(strides,strides),
        dtype=tf.dtypes.float64,
        activation=conv_activation,
        name='Conv_Bottom',
        kernel_regularizer=lambda_regularization,
        use_bias=True)(tensor)
    
    #add batch_normalization if true
    if batch_normalization==True:
        tensor = tf.keras.layers.BatchNormalization()(tensor)
    
    #add dropout if desired
    if dropout>0:
        tensor = tf.keras.layers.SpatialDropout2D(dropout)(tensor)

    filters = np.flip(filters)
    for d in range(deep):
        #UpSample
        tensor = tf.keras.layers.UpSampling2D(dtype=tf.dtypes.float64)(tensor)

        if args.skip:
                tensor = tf.concat([tensor,skips.pop()],axis=3)

        for n in range(n_conv_per_step): 

            #create the convolutional blocks
            tensor = Conv2D(filters=filters[d],
                padding=args.padding,
                kernel_size=(conv_size,conv_size),
                strides=(strides,strides),
                dtype=tf.dtypes.float64,
                activation=conv_activation,
                name='Conv_Up_%s_%s_%s'%(d,n,conv_activation),
                kernel_regularizer=lambda_regularization,
                use_bias=True)(tensor)

            

            #add batch_normalization if true
            if batch_normalization==True:
                tensor = tf.keras.layers.BatchNormalization()(tensor)
                
            #add dropout if desired
            if dropout>0:
                tensor = tf.keras.layers.SpatialDropout2D(dropout)(tensor)

    output_tensor=Conv2D(filters=1,
                        padding = args.padding,
                        kernel_size=1,
                        dtype=tf.dtypes.float64,
                        name='binary_ltg',
                        activation=last_activation,
                        use_bias=True)(tensor)

    #compile the model 
    model = Model(inputs=input_tensor,outputs=output_tensor)
    
    opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)
    model.compile(optimizer=opt,loss=loss,metrics='accuracy')

    return model

def create_unet_static(args,
                image_size=[128,256,1],
                filters=[8,16,32],
                conv_size=2,
                pool_size=2,
                deep=4,
                n_conv_per_step=2,
                lrate=.0001,
                n_types=1,
                loss='binary-cross-entropy',#string or tensor flow object
                metrics='accuracy', #string or tensor flow object 
                padding='same', #string
                strides=1, #int
                conv_activation='elu', #string, consider relu
                last_activation='sigmoid', #predicting the probability
                batch_normalization=False, 
                dropout=0.0,
                skip=False): 

    '''
    This functions builds a U-Net for BoltCast.  It builds down the number of layers 
    determined by the deep parameter.  It performs MaxPooling after each convolution block 
    described by the n_conv_step parameter.  Activation functions throughout the U are 
    provided, along with the activation function for the last layer.  Strides, padding, 
    loss functions, and metrics are provided too.  The filters is a list of the number 
    of filters to use, for every convolution down.  The filters are flipped along the way 
    up.  Flags/booleans for KSC data sets and regularization
    are also provided.  
    '''

    input_tensor = tf.keras.Input(shape=(image_size[0],image_size[1],3),
                                    dtype=tf.dtypes.float64,
                                    name='Input_Tensor')                                    
    tensor = input_tensor

    lambda_regularization = args.L2_reg
    #add l2 regularization if needed
    if lambda_regularization is not None:
        lambda_regularization = tf.keras.regularizers.l2(lambda_regularization)(tensor)
    
    skips = []
    if args.skip==True:
        skips.append(input_tensor)

    #go down the U-net
    for d in range(deep):
        for n in range(n_conv_per_step):

                #create the convolutional blocks
                tensor = Conv2D(filters=filters[d],
                    padding=args.padding,
                    kernel_size=(conv_size,conv_size),
                    strides=(strides,strides),
                    dtype=tf.dtypes.float64,
                    activation=conv_activation,
                    name='Conv_Down_%s_%s_%s'%(d,n,conv_activation),
                    kernel_regularizer=lambda_regularization,
                    use_bias=True)(tensor)
                
                if (args.skip==True) and (n==(n_conv_per_step-1)):
                    skips.append(tensor)

                #add batch_normalization if true
                if batch_normalization==True:
                    tensor = tf.keras.layers.BatchNormalization()(tensor)
                
                #add dropout if desired
                if dropout>0:
                    tensor = tf.keras.layers.SpatialDropout2D(dropout)(tensor)

        #add the max or average pooling layer
        tensor = tf.keras.layers.MaxPooling2D(pool_size=(pool_size,pool_size),name='MaxPool_%s'%d)(tensor)
    
    #learn at the bottom of the U 
    tensor = Conv2D(filters=filters[d]*2,
        padding=args.padding,
        kernel_size=(conv_size,conv_size),
        strides=(strides,strides),
        dtype=tf.dtypes.float64,
        activation=conv_activation,
        name='Conv_Bottom',
        kernel_regularizer=lambda_regularization,
        use_bias=True)(tensor)
    
    #add batch_normalization if true
    if batch_normalization==True:
        tensor = tf.keras.layers.BatchNormalization()(tensor)
    
    #add dropout if desired
    if dropout>0:
        tensor = tf.keras.layers.SpatialDropout2D(dropout)(tensor)

    filters = np.flip(filters)
    for d in range(deep):
        #UpSample
        tensor = tf.keras.layers.UpSampling2D(dtype=tf.dtypes.float64)(tensor)

        if args.skip:
                tensor = tf.concat([tensor,skips.pop()],axis=3)

        for n in range(n_conv_per_step): 

            #create the convolutional blocks
            tensor = Conv2D(filters=filters[d],
                padding=args.padding,
                kernel_size=(conv_size,conv_size),
                strides=(strides,strides),
                dtype=tf.dtypes.float64,
                activation=conv_activation,
                name='Conv_Up_%s_%s_%s'%(d,n,conv_activation),
                kernel_regularizer=lambda_regularization,
                use_bias=True)(tensor)

            

            #add batch_normalization if true
            if batch_normalization==True:
                tensor = tf.keras.layers.BatchNormalization()(tensor)
                
            #add dropout if desired
            if dropout>0:
                tensor = tf.keras.layers.SpatialDropout2D(dropout)(tensor)

    output_tensor=Conv2D(filters=1,
                        padding = args.padding,
                        kernel_size=1,
                        dtype=tf.dtypes.float64,
                        name='binary_ltg',
                        activation=last_activation,
                        use_bias=True)(tensor)

    #compile the model 
    model = Model(inputs=input_tensor,outputs=output_tensor)
    
    opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)
    model.compile(optimizer=opt,loss=loss,metrics='accuracy')

    return model

def create_unet_static_atm(args,
                image_size=[128,256,1],
                filters=[8,16,32],
                conv_size=2,
                pool_size=2,
                deep=4,
                n_conv_per_step=2,
                lrate=.0001,
                n_types=1,
                loss='binary-cross-entropy',#string or tensor flow object
                metrics='accuracy', #string or tensor flow object 
                padding='same', #string
                strides=1, #int
                conv_activation='elu', #string, consider relu
                last_activation='sigmoid', #predicting the probability
                batch_normalization=False, 
                dropout=0.0,
                skip=False): 

    '''
    This functions builds a U-Net for BoltCast.  It builds down the number of layers 
    determined by the deep parameter.  It performs MaxPooling after each convolution block 
    described by the n_conv_step parameter.  Activation functions throughout the U are 
    provided, along with the activation function for the last layer.  Strides, padding, 
    loss functions, and metrics are provided too.  The filters is a list of the number 
    of filters to use, for every convolution down.  The filters are flipped along the way 
    up.  Flags/booleans for KSC data sets and regularization
    are also provided.  
    '''

    input_tensor = tf.keras.Input(shape=(image_size[0],image_size[1],6),
                                    dtype=tf.dtypes.float64,
                                    name='Input_Tensor')                                    
    tensor = input_tensor

    lambda_regularization = args.L2_reg
    #add l2 regularization if needed
    if lambda_regularization is not None:
        lambda_regularization = tf.keras.regularizers.l2(lambda_regularization)(tensor)
    
    skips = []
    if args.skip==True:
        skips.append(input_tensor)

    #go down the U-net
    for d in range(deep):
        for n in range(n_conv_per_step):

                #create the convolutional blocks
                tensor = Conv2D(filters=filters[d],
                    padding=args.padding,
                    kernel_size=(conv_size,conv_size),
                    strides=(strides,strides),
                    dtype=tf.dtypes.float64,
                    activation=conv_activation,
                    name='Conv_Down_%s_%s_%s'%(d,n,conv_activation),
                    kernel_regularizer=lambda_regularization,
                    use_bias=True)(tensor)
                
                if (args.skip==True) and (n==(n_conv_per_step-1)):
                    skips.append(tensor)

                #add batch_normalization if true
                if batch_normalization==True:
                    tensor = tf.keras.layers.BatchNormalization()(tensor)
                
                #add dropout if desired
                if dropout>0:
                    tensor = tf.keras.layers.SpatialDropout2D(dropout)(tensor)

        #add the max or average pooling layer
        tensor = tf.keras.layers.MaxPooling2D(pool_size=(pool_size,pool_size),name='MaxPool_%s'%d)(tensor)
    
    #learn at the bottom of the U 
    tensor = Conv2D(filters=filters[d]*2,
        padding=args.padding,
        kernel_size=(conv_size,conv_size),
        strides=(strides,strides),
        dtype=tf.dtypes.float64,
        activation=conv_activation,
        name='Conv_Bottom',
        kernel_regularizer=lambda_regularization,
        use_bias=True)(tensor)
    
    #add batch_normalization if true
    if batch_normalization==True:
        tensor = tf.keras.layers.BatchNormalization()(tensor)
    
    #add dropout if desired
    if dropout>0:
        tensor = tf.keras.layers.SpatialDropout2D(dropout)(tensor)

    filters = np.flip(filters)
    for d in range(deep):
        #UpSample
        tensor = tf.keras.layers.UpSampling2D(dtype=tf.dtypes.float64)(tensor)

        if args.skip:
                tensor = tf.concat([tensor,skips.pop()],axis=3)

        for n in range(n_conv_per_step): 

            #create the convolutional blocks
            tensor = Conv2D(filters=filters[d],
                padding=args.padding,
                kernel_size=(conv_size,conv_size),
                strides=(strides,strides),
                dtype=tf.dtypes.float64,
                activation=conv_activation,
                name='Conv_Up_%s_%s_%s'%(d,n,conv_activation),
                kernel_regularizer=lambda_regularization,
                use_bias=True)(tensor)

            

            #add batch_normalization if true
            if batch_normalization==True:
                tensor = tf.keras.layers.BatchNormalization()(tensor)
                
            #add dropout if desired
            if dropout>0:
                tensor = tf.keras.layers.SpatialDropout2D(dropout)(tensor)

    output_tensor=Conv2D(filters=1,
                        padding = args.padding,
                        kernel_size=1,
                        dtype=tf.dtypes.float64,
                        name='binary_ltg',
                        activation=last_activation,
                        use_bias=True)(tensor)

    #compile the model 
    model = Model(inputs=input_tensor,outputs=output_tensor)
    
    opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)
    model.compile(optimizer=opt,loss=loss,metrics='accuracy')

    return model
    

def create_unet_static_pres(args,
                image_size=[128,256,1],
                filters=[8,16,32],
                conv_size=2,
                pool_size=2,
                deep=4,
                n_conv_per_step=2,
                lrate=.0001,
                n_types=1,
                loss='binary-cross-entropy',#string or tensor flow object
                metrics='accuracy', #string or tensor flow object 
                padding='same', #string
                strides=1, #int
                conv_activation='elu', #string, consider relu
                last_activation='sigmoid', #predicting the probability
                batch_normalization=False, 
                dropout=0.0,
                skip=False): 

    '''
    This functions builds a U-Net for BoltCast.  It builds down the number of layers 
    determined by the deep parameter.  It performs MaxPooling after each convolution block 
    described by the n_conv_step parameter.  Activation functions throughout the U are 
    provided, along with the activation function for the last layer.  Strides, padding, 
    loss functions, and metrics are provided too.  The filters is a list of the number 
    of filters to use, for every convolution down.  The filters are flipped along the way 
    up.  Flags/booleans for KSC data sets and regularization
    are also provided.  
    '''

    input_tensor = tf.keras.Input(shape=(image_size[0],image_size[1],17),
                                    dtype=tf.dtypes.float64,
                                    name='Input_Tensor')                                    
    tensor = input_tensor

    lambda_regularization = args.L2_reg
    #add l2 regularization if needed
    if lambda_regularization is not None:
        lambda_regularization = tf.keras.regularizers.l2(lambda_regularization)(tensor)
    
    skips = []
    if args.skip==True:
        skips.append(input_tensor)

    #go down the U-net
    for d in range(deep):
        for n in range(n_conv_per_step):

                #create the convolutional blocks
                tensor = Conv2D(filters=filters[d],
                    padding=args.padding,
                    kernel_size=(conv_size,conv_size),
                    strides=(strides,strides),
                    dtype=tf.dtypes.float64,
                    activation=conv_activation,
                    name='Conv_Down_%s_%s_%s'%(d,n,conv_activation),
                    kernel_regularizer=lambda_regularization,
                    use_bias=True)(tensor)
                
                if (args.skip==True) and (n==(n_conv_per_step-1)):
                    skips.append(tensor)

                #add batch_normalization if true
                if batch_normalization==True:
                    tensor = tf.keras.layers.BatchNormalization()(tensor)
                
                #add dropout if desired
                if dropout>0:
                    tensor = tf.keras.layers.SpatialDropout2D(dropout)(tensor)

        #add the max or average pooling layer
        tensor = tf.keras.layers.MaxPooling2D(pool_size=(pool_size,pool_size),name='MaxPool_%s'%d)(tensor)
    
    #learn at the bottom of the U 
    tensor = Conv2D(filters=filters[d]*2,
        padding=args.padding,
        kernel_size=(conv_size,conv_size),
        strides=(strides,strides),
        dtype=tf.dtypes.float64,
        activation=conv_activation,
        name='Conv_Bottom',
        kernel_regularizer=lambda_regularization,
        use_bias=True)(tensor)
    
    #add batch_normalization if true
    if batch_normalization==True:
        tensor = tf.keras.layers.BatchNormalization()(tensor)
    
    #add dropout if desired
    if dropout>0:
        tensor = tf.keras.layers.SpatialDropout2D(dropout)(tensor)

    filters = np.flip(filters)
    for d in range(deep):
        #UpSample
        tensor = tf.keras.layers.UpSampling2D(dtype=tf.dtypes.float64)(tensor)

        if args.skip:
                tensor = tf.concat([tensor,skips.pop()],axis=3)

        for n in range(n_conv_per_step): 

            #create the convolutional blocks
            tensor = Conv2D(filters=filters[d],
                padding=args.padding,
                kernel_size=(conv_size,conv_size),
                strides=(strides,strides),
                dtype=tf.dtypes.float64,
                activation=conv_activation,
                name='Conv_Up_%s_%s_%s'%(d,n,conv_activation),
                kernel_regularizer=lambda_regularization,
                use_bias=True)(tensor)

            

            #add batch_normalization if true
            if batch_normalization==True:
                tensor = tf.keras.layers.BatchNormalization()(tensor)
                
            #add dropout if desired
            if dropout>0:
                tensor = tf.keras.layers.SpatialDropout2D(dropout)(tensor)

    output_tensor=Conv2D(filters=1,
                        padding = args.padding,
                        kernel_size=1,
                        dtype=tf.dtypes.float64,
                        name='binary_ltg',
                        activation=last_activation,
                        use_bias=True)(tensor)

    #compile the model 
    model = Model(inputs=input_tensor,outputs=output_tensor)
    
    opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)
    model.compile(optimizer=opt,loss=loss,metrics='accuracy')

    return model


if __name__ == "__main__":

    print('BC_unet_classifier.py main function')

    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()

    image_size=[128,256,1]

    if args.build_model:
        print('building the model')
        model = create_unet(args,
                        image_size=image_size,
                        filters=args.conv_nfilters,
                        conv_size=args.conv_size,
                        pool_size=args.pool,
                        deep=args.deep,
                        n_conv_per_step=args.n_conv_per_step,
                        lrate=args.lrate,
                        loss=tf.keras.losses.BinaryCrossentropy(),#tensor flow loss function
                        metrics=tf.keras.metrics.SparseCategoricalAccuracy(),#tensor flow metrics
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
                        metrics=tf.keras.metrics.SparseCategoricalAccuracy(),#tensor flow metrics
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
                        metrics=tf.keras.metrics.SparseCategoricalAccuracy(),#tensor flow metrics
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
                        metrics=tf.keras.metrics.SparseCategoricalAccuracy(),#tensor flow metrics
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
        render_fname = 'LC_model_test.png'
        plot_model(model, to_file=render_fname, show_shapes=True, show_layer_names=True)

    


