import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def main():

    # visible_devices = tf.config.get_visible_devices('GPU') 
    # n_visible_devices = len(visible_devices)
    # print(n_visible_devices)
    # tf.config.set_visible_devices([], 'GPU')
    # print('GPU turned off')
    
    print('BC_Train_cor.py')
    for rot in [0,1,2,3,4]:
        load_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/binary/'
        fload = 'rot_%s_train.tf'%(rot)

        tfds = tf.data.Dataset.load(load_dir+fload)
        print(tfds)

        c=0
        for in_for,out_for in tfds:
            print(c)
            if c==0:
                inputs=in_for
                print('inputs.shape',inputs.shape)
            else:
                data = in_for
                print('data.shape',data.shape)
                inputs = np.concatenate([inputs,data],axis=0)
                print('inputs.shape',inputs.shape)
            c+=1

        ravel_np = np.zeros([inputs.shape[0]*inputs.shape[1]*inputs.shape[2]*inputs.shape[3],9])
        
        for i in range(9):
            ravel_np[:,i] = np.ravel(inputs[:,:,:,:,i])
        print('ravel_np.shape:', ravel_np.shape)

        vars = ['CAPE','LI','Z','PRate','W','Ice_q','Sn_q','Gr_q','Rn_q']
        corr_coef_np = np.corrcoef(ravel_np,rowvar=False)
        print(corr_coef_np)
        print(corr_coef_np.shape)

        plt.matshow(corr_coef_np,cmap='coolwarm')
        plt.xticks(range(9),vars,fontsize=8,rotation=90)
        plt.gca().xaxis.tick_bottom()
        plt.yticks(range(9),vars,fontsize=8,rotation=0)
        for i in range(9):
            for j in range(9):
                corr = corr_coef_np[i,j]
                r_str = f"{corr:.2f}"
                plt.text(i-.4,j+.2,r_str)
        plt.title('Training Dataset: Rotation '+str(rot),fontsize=14)
        plt.savefig('corr_coef_np_rot_%s.png'%(rot))
        plt.close()
        del tfds, inputs, corr_coef_np, ravel_np, data 


if __name__=="__main__":
    main()