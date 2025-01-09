import numpy as np
import matplotlib.pyplot as plt
import pickle
static_inputs = pickle.load(open('../2_model_training/static_inputs.pkl','rb'))

lat = static_inputs['lat']
lon = static_inputs['lon']
terrain = static_inputs['terrain']

fig,axes = plt.subplots(ncols = 1,
                        nrows=3)
cb = axes[0].imshow(lat)
axes[0].set_title('lat')
plt.colorbar(cb,ax=axes[0])

cb=axes[1].imshow(lon)
axes[1].set_title('lon')
plt.colorbar(cb,ax=axes[1])

cb = axes[2].imshow(terrain)
axes[2].set_title('terrain')
plt.colorbar(cb,ax=axes[2])
plt.savefig('test.png')
plt.close()

stacked = np.dstack([lat,lon,terrain])
stacked = np.swapaxes(np.swapaxes(stacked,2,0),1,2)#4,128,256 
fig,axes = plt.subplots(ncols=1,
                        nrows=3)
cb = axes[0].imshow(stacked[0,:,:])
cb = axes[1].imshow(stacked[1,:,:])
cb = axes[2].imshow(stacked[2,:,:])
plt.savefig('test2.png')
plt.close()