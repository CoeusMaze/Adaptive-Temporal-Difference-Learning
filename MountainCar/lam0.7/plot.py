import pickle
import numpy as np
import matplotlib.pyplot as plt

# parameters
name = 'MountainCar-v0'
optimizers = ('ATD', 'ALRR', 'TD')
end = 300

# load mc loss histories
mc_curves = []
for i in range(len(optimizers)):
    with open('mc_'+optimizers[i]+'_linear_'+name+'.pkl', 'rb') as f:
        mc_curves.append(pickle.load(f))

# plotting
plt.style.use('seaborn')
plt.rc('axes', labelsize=19)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=17)

fig = plt.figure(figsize=(6.4, 5.6))
plt.xscale('log')
plt.xlabel('Episode')
plt.ylabel('RMSBE')
for i in range(len(optimizers)):
    curves = mc_curves[i]
    ys = []
    for j in range(curves.shape[0]):
        curve = curves[j][:end]
        y = []
        for i in range(len(curve)):
            x = i + 1
            y.append(np.mean(curve[0:x]))
        ys.append(y)
    y_mean = np.mean(ys, axis = 0)
    y_std = np.std(ys, axis = 0)
    x = np.linspace(0, len(y)-1, num = len(y), dtype = int)+1
    plt.plot(x, y_mean)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    plt.fill_between(x, y_mean+y_std, y_mean-y_std, alpha=0.3, antialiased=True)
plt.legend(('AdaTD(0.7)','ALRR-TD(0.7)','TD(0.7)'))
fig.savefig('All_Runtime_Loss_linear_'+name+'.png', dpi=fig.dpi)
plt.show()
