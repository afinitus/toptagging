import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import math

file_path = '../scripts/evaluate_ttbar.h5'

f = h5py.File(file_path, 'r')
raw_data = np.array(f['data'])
pid = np.array(f['pid'])
data = np.array([])
#data_arr = []
#for i in range(len(pid)):
#    if pid[i] == 1.:
#        data_arr.append(raw_data[i])
#data = np.array(data_arr)
print(data.shape)
label_list = ['E', 'p_x', 'p_y', 'p_z']
sigbkg = pid.ravel()
energy_log = data[:,:,3].ravel()
pt_log = data[:,:,2].ravel()
phi = data[:,:,1].ravel()
eta = data[:,:,0].ravel()
energy = []
px = []
py = []
pz = []

for i in range(len(pt_log)):
    energy.append(math.exp(energy_log[i]))
    px_val = math.exp(pt_log[i]) * np.cos(phi[i])
    py_val = np.tan(phi[i]) * px_val
    pz_val = np.sinh(eta[i]) * math.exp(pt_log[i])
    if pt_log[i] == 0:
        px.append(0)
        py.append(0)
        pz.append(0)
    else:
        px.append(px_val)
        py.append(py_val)
        pz.append(pz_val)


fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
for i, (name, ax) in enumerate(zip(label_list, np.array(axs).ravel())):
    if i == 0:
        ax.hist(energy, bins=100)
    elif i == 1:
        ax.hist(px, bins=100)
    elif i == 2:
        ax.hist(py, bins=100)
    elif i == 3:
        ax.hist(pz, bins=100)
    ax.set_yscale("log")
    ax.set_title(name)

fig.savefig('plots')