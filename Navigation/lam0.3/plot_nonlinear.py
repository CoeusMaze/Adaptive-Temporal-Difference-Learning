import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# parameters
name = 'simple_customized'
optimizers = ('ATD', 'TD')
color = ("windows blue", "pale red")
start = 0
end = 300

# load mc loss histories
optimizers_curves = []
for i in range(len(optimizers)):
    with open('mc_'+optimizers[i]+'_nonlinear_'+name+'.pkl', 'rb') as f:
        optimizers_curves.append(pickle.load(f))


# plotting
plt.style.use('seaborn')
plt.rc('axes', labelsize=19)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=17)

fig = plt.figure(figsize=(6.4, 5.6))
plt.xlabel('Episode')
plt.ylabel('RMSBE')
for i in range(len(optimizers)):
    curves = optimizers_curves[i][:,start:end+1]
    y_mean = np.mean(curves, axis = 0)
    y_std = np.std(curves, axis = 0)
    y_std[start:1] = 0
    x = np.linspace(start, len(y_mean), num = len(y_mean), dtype=int)
    plt.plot(x, y_mean, sns.xkcd_rgb[color[i]])
    plt.fill_between(x, y_mean+y_std, y_mean-y_std, color=sns.xkcd_rgb[color[i]], alpha=0.4, antialiased=True)
plt.legend(('AdaTD(0.3)','TD(0.3)'))
fig.savefig('All_Runtime_Loss_nonlinear_'+name+'.png', dpi=fig.dpi)
plt.show()
