import argparse
import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt

def load_data(rotation=0,
            f_hour='f000',
            base_dir='/scratch/bmac87/rotations/'):

    
     input_labels = ['graupel_250',
         'graupel_500',
         'graupel_700',
         'ice_250',
         'ice_500',
         'ice_700',
         'rain_250',
         'rain_500',
         'rain_700',
         'snow_250',
         'snow_500',
         'snow_700',
         'w_500',
         'w_700',
         'reflectivity',
         'cape',
         'lifted_index']

     print("loading the data in BC_data_loader.py")
     data_file = base_dir+'%s/r%s.pkl'%(f_hour,rotation)
     print(data_file)
     all_data = pickle.load(open(data_file,'rb'))

     print('training data shapes')
     X_train = all_data['train']['x']#dataset
     y_train = all_data['train']['y']#dataset
     print(X_train.dims)
     print(y_train.dims)

     print('validation data shapes')
     X_val = all_data['val']['x']#dataset
     y_val = all_data['val']['y']#dataset
     print(X_val.dims)
     print(y_val.dims)


     print('test data shapes')
     X_test = all_data['test']['x']#dataset
     y_test = all_data['test']['y']#dataset
     print(X_test.dims)
     print(y_test.dims)

     print('normalize each variable in the training, validation, and testing datasets')
    

     X_tr_list = []
     X_test_list = []
     X_val_list = []

     input_labels = ['graupel_250',
         'graupel_500',
         'graupel_700',
         'ice_250',
         'ice_500',
         'ice_700',
         'rain_250',
         'rain_500',
         'rain_700',
         'snow_250',
         'snow_500',
         'snow_700',
         'w_500',
         'w_700',
         'reflectivity',
         'cape',
         'lifted_index',
         'lat',
         'lon',
         'terrain']


     trng_dict = {'graupel_250':np.zeros([len(X_train['time']),128,256]),
         'graupel_500':np.zeros([len(X_train['time']),128,256]),
         'graupel_700':np.zeros([len(X_train['time']),128,256]),
         'ice_250':np.zeros([len(X_train['time']),128,256]),
         'ice_500':np.zeros([len(X_train['time']),128,256]),
         'ice_700':np.zeros([len(X_train['time']),128,256]),
         'rain_250':np.zeros([len(X_train['time']),128,256]),
         'rain_500':np.zeros([len(X_train['time']),128,256]),
         'rain_700':np.zeros([len(X_train['time']),128,256]),
         'snow_250':np.zeros([len(X_train['time']),128,256]),
         'snow_500':np.zeros([len(X_train['time']),128,256]),
         'snow_700':np.zeros([len(X_train['time']),128,256]),
         'w_500':np.zeros([len(X_train['time']),128,256]),
         'w_700':np.zeros([len(X_train['time']),128,256]),
         'reflectivity':np.zeros([len(X_train['time']),128,256]),
         'cape':np.zeros([len(X_train['time']),128,256]),
         'lifted_index':np.zeros([len(X_train['time']),128,256]),
         'lat':np.zeros([len(X_train['time']),128,256]),
         'lon':np.zeros([len(X_train['time']),128,256]),
         'terrain':np.zeros([len(X_train['time']),128,256])}

     tnv_dict = {'graupel_250':np.zeros([len(X_test['time']),128,256]),
         'graupel_500':np.zeros([len(X_test['time']),128,256]),
         'graupel_700':np.zeros([len(X_test['time']),128,256]),
         'ice_250':np.zeros([len(X_test['time']),128,256]),
         'ice_500':np.zeros([len(X_test['time']),128,256]),
         'ice_700':np.zeros([len(X_test['time']),128,256]),
         'rain_250':np.zeros([len(X_test['time']),128,256]),
         'rain_500':np.zeros([len(X_test['time']),128,256]),
         'rain_700':np.zeros([len(X_test['time']),128,256]),
         'snow_250':np.zeros([len(X_test['time']),128,256]),
         'snow_500':np.zeros([len(X_test['time']),128,256]),
         'snow_700':np.zeros([len(X_test['time']),128,256]),
         'w_500':np.zeros([len(X_test['time']),128,256]),
         'w_700':np.zeros([len(X_test['time']),128,256]),
         'reflectivity':np.zeros([len(X_test['time']),128,256]),
         'cape':np.zeros([len(X_test['time']),128,256]),
         'lifted_index':np.zeros([len(X_test['time']),128,256]),
         'lat':np.zeros([len(X_test['time']),128,256]),
         'lon':np.zeros([len(X_test['time']),128,256]),
         'terrain':np.zeros([len(X_test['time']),128,256])}

     X_train_dict = trng_dict
     X_val_dict = tnv_dict
     X_test_dict = tnv_dict

     #normalize the training dataset
     for l,label in enumerate(input_labels):
          
          if l<17:
               print('normalizing, ',label, l)

               #use self written norm function to normalize the data
               X_train_dict[label] = norm(X_train[label])
          else:
               print('normalizing static, ',label, l)
               static_inputs = pickle.load(open('static_inputs.pkl','rb'))
               X_train_dict[label] = norm1(static_inputs[label])

     y_train_dict = {'binary_ltg':y_train['binary_ltg'].values}
     y_val_dict = {'binary_ltg':y_val['binary_ltg'].values}
     y_test_dict = {'binary_ltg':y_test['binary_ltg'].values}
     

     #normalize the testing and validation datasets
     for l,label in enumerate(input_labels):
          if l<17:
               print('normalizing TNV, ',label, l)
               X_val_dict[label] = norm(X_val[label])
               X_test_dict[label] = norm(X_test[label])

          else:
               print('normalizing TNV static, ',label, l)

               static_inputs = pickle.load(open('tnv_static_inputs.pkl','rb'))
               X_val_dict[label] = norm1(static_inputs[label])
               X_test_dict[label] = norm1(static_inputs[label])
     
     return X_train_dict, y_train_dict, X_val_dict, y_val_dict, X_test_dict, y_test_dict

def norm(input_data):

     input_avg = np.mean(np.mean(np.mean(input_data,axis=2),axis=1),axis=0).values
     input_max = np.max(np.max(np.max(input_data,axis=2),axis=1),axis=0).values
     input_min = np.min(np.min(np.min(input_data,axis=2),axis=1),axis=0).values

     norm_num = input_data.values-input_min*np.ones(input_data.values.shape)
     norm_den = np.ones(input_data.values.shape)*input_max - np.ones(input_data.values.shape)*input_min
     norm_input = np.divide(norm_num,norm_den)

     return norm_input

#written specifically for the static inputs
def norm1(input_data):
     input_avg = np.mean(np.mean(np.mean(input_data,axis=2),axis=1),axis=0)
     input_max = np.max(np.max(np.max(input_data,axis=2),axis=1),axis=0)
     input_min = np.min(np.min(np.min(input_data,axis=2),axis=1),axis=0)

     norm_num = input_data-input_min*np.ones(input_data.shape)
     norm_den = np.ones(input_data.shape)*input_max - np.ones(input_data.shape)*input_min
     norm_input = np.divide(norm_num,norm_den)

     return norm_input

def build_figures(X_norm_np,X_train_da,label):

     print(label,' building figure')
     static_inputs = pickle.load(open('tnv_static_inputs.pkl','rb'))
     lat = np.squeeze(static_inputs['lat'][0,:,:])
     lon = np.squeeze(static_inputs['lon'][0,:,:])
     
     num_times = X_norm_np.shape[0]
     for i in range(num_times):
          if i%50==0:
               fig = plt.figure(figsize=(20,10))
               data_norm = np.squeeze(X_norm_np[i,:,:])
               data_orig = X_train_da.values[i,:,:]
               #plot the temperature
               ax = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
               cf_norm = ax.contourf(lon,lat,data_norm,transform=ccrs.PlateCarree())
               ax.add_feature(cfeature.COASTLINE, edgecolor="black")
               ax.add_feature(cfeature.STATES,edgecolor="black")
               cbar_ax = fig.add_axes([0.2, 0.2, 0.6, 0.02])
               cbar=fig.colorbar(cf_norm, cax=cbar_ax,orientation='horizontal',label='Scaled')
               ax.set_title('Min/Max Feature Scaled Data')

               ax = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
               cf_orig = ax.contourf(lon,lat,data_orig,transform=ccrs.PlateCarree())
               ax.add_feature(cfeature.COASTLINE, edgecolor="black")
               ax.add_feature(cfeature.STATES,edgecolor="black")
               ax.set_title('Original Data')
               cbar1_ax = fig.add_axes([0.2, 0.1, 0.6, 0.02])
               cbar1 = fig.colorbar(cf_orig,cax=cbar1_ax,orientation='horizontal',label='Original')
               plt.suptitle(label,fontsize=24)
               plt.savefig('../images/test_norm/'+label+'_'+str(i)+'.png')
               plt.close()

if __name__=="__main__":
     load_data(rotation=0,
            f_hour='f000',
            base_dir='/scratch/bmac87/rotations/')
