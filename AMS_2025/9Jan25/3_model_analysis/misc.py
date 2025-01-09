def generate_output_samples(test_ds,#un-normalized xarray dataset, xarray
                            test_tf, #tensorflow dataset, tf
                            model, #tensorflow model, tf
                            results_dir,#directory to store the images, str
                            model_file,#filename, str
                            rotation):#rotation, int

    lat = test_ds['lat'].values
    lon = test_ds['lon'].values

    take_idx = 3
    test_results = model.predict(test_tf.take(take_idx))
    print(type(test_results))
    print(test_results.shape)
    
    take_count=0
    for inputs,labels in test_tf.take(take_idx):
        inputs_np = inputs.numpy()
        labels_np = labels.numpy()
        if take_count==0:
            inputs_stack = inputs_np
            labels_stack = labels_np
        else:
            print(inputs_np.shape)
            print(inputs_stack.shape)
            inputs_stack = np.concatenate([inputs_stack,inputs_np],axis=0)
            labels_stack = np.concatenate([labels_stack,labels_np],axis=0)
        take_count+=1        
    print('labels shape')
    print(labels_stack.shape)

    axis0_len = labels_stack.shape[0]
    for i in range(axis0_len):
        if i%4==0:
            print('plotting the figures')
            fig, axes = plt.subplots(nrows=4,
                                    ncols=2,
                                    subplot_kw={'projection': ccrs.PlateCarree()})
            
            axes[0,0].set_title('Predictions')
            axes[0,0].set_ylabel('Day 1')
            axes[0,0].set_xticks([])
            axes[0,0].set_yticks([])
            data = np.squeeze(test_results[i,0,:,:])
            cb0 = axes[0,0].pcolormesh(lon,lat,data,vmin=0,vmax=1,cmap='coolwarm')
            plt.colorbar(cb0,ax=axes[0,0])
            axes[0,0].add_feature(cfeature.COASTLINE,edgecolor='white',linewidth=.25)
            axes[0,0].add_feature(cfeature.STATES,edgecolor='white',linewidth=.25)

            axes[1,0].set_ylabel('Day 2')
            axes[1,0].set_xticks([])
            axes[1,0].set_yticks([])
            data = np.squeeze(test_results[i,1,:,:])
            cb1 = axes[1,0].pcolormesh(lon,lat,data,vmin=0,vmax=1,cmap='coolwarm')
            plt.colorbar(cb1,ax=axes[1,0])
            axes[1,0].add_feature(cfeature.STATES,edgecolor='white',linewidth=.25)
            axes[1,0].add_feature(cfeature.COASTLINE,edgecolor='white',linewidth=.25)

            axes[2,0].set_ylabel('Day 3')
            axes[2,0].set_xticks([])
            axes[2,0].set_yticks([])
            data = np.squeeze(test_results[i,2,:,:])
            cb2 = axes[2,0].pcolormesh(lon,lat,data,vmin=0,vmax=1,cmap='coolwarm')
            plt.colorbar(cb2,ax=axes[2,0])
            axes[2,0].add_feature(cfeature.STATES,edgecolor='white',linewidth=.25)
            axes[2,0].add_feature(cfeature.COASTLINE,edgecolor='white',linewidth=.25)
            
            axes[3,0].set_ylabel('Day 4')
            axes[3,0].set_xticks([])
            axes[3,0].set_yticks([])
            data = np.squeeze(test_results[i,3,:,:])
            cb3 = axes[3,0].pcolormesh(lon,lat,data,vmin=0,vmax=1,cmap='coolwarm')
            plt.colorbar(cb3,ax=axes[3,0])
            axes[3,0].add_feature(cfeature.STATES,edgecolor='white',linewidth=.25)
            axes[3,0].add_feature(cfeature.COASTLINE,edgecolor='white',linewidth=.25)
            
            axes[0,1].set_title('Labels')
            axes[0,1].set_xticks([])
            axes[0,1].set_yticks([])
            data = np.squeeze(labels_stack[i,0,:,:])
            cb4 = axes[0,1].pcolormesh(lon,lat,data,vmin=0,vmax=1,cmap='coolwarm')
            plt.colorbar(cb4,ax=axes[0,1])
            axes[0,1].add_feature(cfeature.COASTLINE,edgecolor='white',linewidth=.25)
            axes[0,1].add_feature(cfeature.STATES,edgecolor='white',linewidth=.25)

            axes[1,1].set_xticks([])
            axes[1,1].set_yticks([])
            data = np.squeeze(labels_stack[i,1,:,:])
            cb5 = axes[1,1].pcolormesh(lon,lat,data,vmin=0,vmax=1,cmap='coolwarm')
            plt.colorbar(cb5,ax=axes[1,1])
            axes[1,1].add_feature(cfeature.COASTLINE,edgecolor='white',linewidth=.25)
            axes[1,1].add_feature(cfeature.STATES,edgecolor='white',linewidth=.25)

            axes[2,1].set_xticks([])
            axes[2,1].set_yticks([])
            data = np.squeeze(labels_stack[i,2,:,:])
            cb6 = axes[2,1].pcolormesh(lon,lat,data,vmin=0,vmax=1,cmap='coolwarm')
            plt.colorbar(cb6,ax=axes[2,1])
            axes[2,1].add_feature(cfeature.COASTLINE,edgecolor='white',linewidth=.25)
            axes[2,1].add_feature(cfeature.STATES,edgecolor='white',linewidth=.25)

            axes[3,1].set_xticks([])
            axes[3,1].set_yticks([])
            data = np.squeeze(labels_stack[i,3,:,:])
            cb7 = axes[3,1].pcolormesh(lon,lat,data,vmin=0,vmax=1,cmap='coolwarm')
            plt.colorbar(cb7,ax=axes[3,1])
            axes[3,1].add_feature(cfeature.COASTLINE,edgecolor='white',linewidth=.25)
            axes[3,1].add_feature(cfeature.STATES,edgecolor='white',linewidth=.25)

            sup_title = model_file[8:-6]
            plt.suptitle(sup_title)
            save_dir = './test_tf_output/LSTM/'
            if os.path.isdir(save_dir)==False:
                os.makedirs(save_dir)
            plt.savefig(save_dir+sup_title+'_'+str(i)+'.png')
            plt.close()

def main():

    rotation=0
    LSTM = True
    if LSTM==True:
        exp_name = 'depth_exp1'
        model_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/Conv2DLSTM/depth_exp1/'
        model_file = 'BC_LSTM_rot_0_LR_0.000010000_lstm_deep_3_conv_deep_2_conv_size_4_stride_1_epochs_500__dropout_0.0_conv_relu__last_sigmoid__model'

    else: 
        print('loading the model')  
        model_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/results/models/'
        model_file = 'BC_UNet_rot_0_LR_0.000010000_deep_3_nconv_3_conv_size_4_stride_1_epochs_500__SD_0.2_conv_relu__last_sigmoid__model'
    
    model = tf.keras.models.load_model(model_dir+model_file)

    print("loading the test data")
    data_dir = '/scratch/bmac87/BoltCast_scratch/'
    test_tf = tf.data.Dataset.load(data_dir+'rot_%s_test.tf'%(rotation))
    test_ds = xr.open_dataset(data_dir+'rot_%s_test.nc'%(rotation),engine='netcdf4')

    generate_output_samples(test_ds,test_tf,model,model_dir,model_file,rotation)