import os
import sys

features = ['cape','precip_rate','lifted_idx','reflectivity','w','ice_q','snow_q','rain_q','graupel_q']
for perm_num in range(25):
    print(perm_num)
if model_type=='UNet':
    fsave = './single_perm_results_pkl/%s_perm_num_%s_%s_rot_%s_threshold_%s.pkl'%(args.feature,args.perm_num,args.model_type,args.rotation,thresh)
else:
    fsave = './single_perm_results_pkl/%s_perm_num_%s_%s_rot_%s_lstm_deep_%s_threshold_%s.pkl'%(args.feature,args.perm_num,args.model_type,args.rotation,lstm_deep,thresh)