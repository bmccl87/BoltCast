import xarray as xr
import numpy as np
import tensorflow as tf
import keras
import copy
import pickle
import sys
import argparse

def main():
    print('BC_perm_importance.py')
    print('the modules loaded successfully')

    parser = argparse.ArgumentParser(description='BoltCast', fromfile_prefix_chars='@')
    parser.add_argument('--perm_num',type=int,default=0)
    parser.add_argument('--feature',type=str,default='cape')
    parser.add_argument('--model_type',type=str,default='UNet')
    parser.add_argument('--rotation',type=int,default=4)
    parser.add_argument('--lrate',type=float,default=.00001)

    args = parser.parse_args()
    print('setting the model type, rotation, single feature, and permutation numbers')
    print('perm_num',args.perm_num)
    print('feature',args.feature)
    print('lrate',args.lrate)
    print('rotation',args.rotation)
    print('model_type',args.model_type)

    if args.model_type=='LSTM':
        lstm_layers = ['1','2','3']
    else:
        lstm_layers=['0']

    thresh=.35
    print('generating the metrics list')
    opt = keras.optimizers.Adam(learning_rate=args.lrate, amsgrad=False)
    loss_tf = tf.keras.losses.BinaryCrossentropy()
    auc_roc_tf = tf.keras.metrics.AUC(name='auc_ROC',curve='ROC')
    auc_pr_tf = tf.keras.metrics.AUC(name='auc_PR',curve='PR')
    acc_tf = tf.keras.metrics.BinaryAccuracy(name='binary_accuracy',threshold=thresh)
    prec_tf = tf.keras.metrics.Precision(name='precision',thresholds=thresh)
    recall_tf = tf.keras.metrics.Recall(name='recall',thresholds=thresh)
    all_metrics = [auc_roc_tf,auc_pr_tf,acc_tf,prec_tf,recall_tf]
    
    print('loading the test dataset')
    load_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/binary/'
    load_file = 'rot_%s_test.nc'%(args.rotation)
    test_ds = xr.open_dataset(load_dir+load_file,engine='netcdf4')
    X = np.float32(test_ds['x'].values)
    y_true = np.float32(test_ds['y'].values)
    features = test_ds['features'].values
    del test_ds

    for lstm_deep in lstm_layers:
        if args.model_type=='UNet':
            model_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/UNet_symmetric/all_rots/models/'
            model_file = 'BC_UNet_rot_%s_LR_0.000010000_deep_3_nconv_3_conv_size_4_stride_1_epochs_500__binary_batch_32_symmetric__SD_0.2_conv_relu__last_sigmoid__model.keras'%(args.rotation)
        else:
            model_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/AMS_2025/Conv2DLSTM/all_rots/models/'
            model_file = 'BC_LSTM_rot_%s_LR_0.000010000_lstm_deep_%s_conv_deep_2_conv_size_4_stride_1_epochs_500__binary_batch_32__dropout_0.0_conv_relu__last_sigmoid__model.keras'%(args.rotation,lstm_deep)
        model = tf.keras.models.load_model(model_dir+model_file)

        print('compiling the model')
        model.compile(optimizer=opt,loss=loss_tf,metrics=all_metrics)

        print('evaluating the model, un-permuted')
        y_eval_dict = model.evaluate(x=X,y=y_true,return_dict=True,verbose=0)
        print('y_eval_dict',y_eval_dict)

        for f,feature in enumerate(features):
            if feature==args.feature:
                print('permuting,',f,feature)

                #get shuffled indices along the samples
                idx_shuffled = np.random.choice(np.arange(0,X.shape[0]),replace=False,size=X.shape[0])
                X_shuffled = np.float32(copy.deepcopy(X))
                X_shuffled[:,:,:,:,f] = X_shuffled[:,:,:,:,f][idx_shuffled]

                print('evaluating')
                print('X_shuffled.shape',X_shuffled.shape)
                print(type(X_shuffled), type(y_true))
                print('size of X_shuffled',sys.getsizeof(X_shuffled)/1e9,'GB')
                metrics_shuffled = model.evaluate(X_shuffled,y_true,return_dict=True,verbose=0)
                print('metrics_shuffled',metrics_shuffled)

                metrics_dict = {'auc_ROC':metrics_shuffled['auc_ROC'],
                                'auc_PR':metrics_shuffled['auc_PR'],
                                'binary_accuracy':metrics_shuffled['binary_accuracy'],
                                'precision':metrics_shuffled['precision'],
                                'recall':metrics_shuffled['recall'],
                                'No_Perm':y_eval_dict}
                print('saving the permutation results')
                if args.model_type=='UNet':
                    fsave = './single_perm_results_pkl/%s_perm_num_%s_%s_rot_%s_threshold_%s.pkl'%(args.feature,args.perm_num,args.model_type,args.rotation,thresh)
                else:
                    fsave = './single_perm_results_pkl/%s_perm_num_%s_%s_rot_%s_lstm_deep_%s_threshold_%s.pkl'%(args.feature,args.perm_num,args.model_type,args.rotation,lstm_deep,thresh)
                pickle.dump(metrics_dict,open(fsave,'wb'))
                print('saved metrics_dict successfully')

if __name__=="__main__":
    main()
    print('main method complete')