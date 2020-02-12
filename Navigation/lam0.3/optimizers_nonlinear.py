import os
import pickle
import numpy as np
import tensorflow as tf
from make_env import make_env
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras.initializers as ki
from tensorflow.keras.utils import to_categorical
os.environ["CUDA_VISIBLE_DEVICES"]=""
        

class v_model(tf.keras.Model):
    
    def __init__(self):
        super().__init__('mlp_value')
        self.hidden1 = kl.Dense(64, activation='relu', kernel_initializer = ki.RandomNormal(mean=0.0, stddev=1e-2, seed=1))
        self.hidden2 = kl.Dense(64, activation = 'relu', kernel_initializer = ki.RandomNormal(mean=0.0, stddev=1e-2, seed=2))
        self.values = kl.Dense(1, name='critic_value', kernel_initializer = ki.RandomNormal(mean=0.0, stddev=1e-2, seed=3))


    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        hidden_logs1 = self.hidden1(x)
        hidden_logs2 = self.hidden2(hidden_logs1)
        
        return self.values(hidden_logs2)
        
        
class TD_optimizers:

    def __init__(self):
        
        self.params = {
            'gamma': 0.95,
        }
    
    
    def train(self, env, ATD_params, TD_params, mc=10, batch_sz=32, max_episode=500, reset_rate=100, lam=0.0, runtime_range = 100, render=False):
        
        for j in range(mc):
            
            # initialize value models
            vmodel_atd = v_model()
            vmodel_td = v_model()
            
            # initialize containers
            ep_loss_atd, runtime_atd, ep_loss_td, runtime_td = [], [], [], []
            actions = np.zeros((bsize), dtype = np.int32)
            rewards, dones, values_atd, values_td = np.empty((4, bsize))
            observations = np.zeros((bsize,) + env.observation_space[0].shape, dtype = np.float32)
            
            # initialize optimizers
            lr_atd, beta, delt = ATD_params
            n = 6
            m = [0] * n
            v = [0] * n
            z_atd = [0] * n
            lr_td = TD_params
            z_td = [0] * n
        
            # start training
            next_obs = np.array(env.reset())
            for episode in range(max_episode):
                if (episode+1) % reset_rate == 0:
                    next_obs = np.array(env.reset())
                for step in range(bsize):
                    observations[step] = next_obs.copy()
                    actions[step] = np.random.randint(env.action_space[0].n)
                    values_atd[step] = vmodel_atd.predict(next_obs)
                    values_td[step] = vmodel_td.predict(next_obs)
                    # we assume uniform action space size
                    onehot = to_categorical(actions[step], env.action_space[0].n)[None,:]
                    next_obs, rewards[step], dones[step], _ = [np.array(ret) for ret in env.step(onehot)]
                    if render:
                        env.render()
                    if dones[step]:
                        next_obs = np.array(env.reset())
                
                # ATD update
                next_value = vmodel_atd.predict(next_obs)
                targets = self._targets_(rewards, dones, values_atd, next_value)
                msbe_atd = kls.mean_squared_error(targets, values_atd)
                for step in range(bsize):
                    with tf.GradientTape() as tape1:
                        pred_atd = vmodel_atd(observations[step][None,:], training = True)
                        vloss_atd = pred_atd
                    gradients = tape1.gradient(vloss_atd, vmodel_atd.trainable_variables)
                    for i in range(n):
                        z_atd[i] = self.params['gamma'] * lam * z_atd[i] + gradients[i].numpy()
                        gradients[i] = (values_atd[step] - targets[step]) * z_atd[i]
                        m[i] = beta * m[i] + (1-beta) * gradients[i]
                        v[i] = v[i] + np.linalg.norm(gradients[i])**2
                        vmodel_atd.trainable_variables[i].assign(vmodel_atd.trainable_variables[i].numpy() - (lr_atd*m[i]/np.sqrt(delt+v[i]))/bsize)
                    if dones[step]:
                        z_atd = [0] * n

                # TD update 
                next_value = vmodel_td.predict(next_obs)
                targets = self._targets_(rewards, dones, values_td, next_value)
                msbe_td = kls.mean_squared_error(targets, values_td)
                for step in range(bsize):
                    with tf.GradientTape() as tape2:
                        pred_td = vmodel_td(observations[step][None,:], training = True)
                        vloss_td = pred_td
                    gradients = tape2.gradient(vloss_td, vmodel_td.trainable_variables)
                    for i in range(n):
                        z_td[i] = self.params['gamma'] * lam * z_td[i] + gradients[i].numpy()
                        gradients[i] = (values_td[step] - targets[step]) * z_td[i]
                        vmodel_td.trainable_variables[i].assign(vmodel_td.trainable_variables[i].numpy() - (lr_td * gradients[i])/bsize)
                    if dones[step]:
                        z_td = [0] * n
                
                ep_loss_atd.append(np.mean(msbe_atd.numpy()))
                runtime_atd.append(np.mean(ep_loss_atd[-runtime_range:]))
                ep_loss_td.append(np.mean(msbe_td.numpy()))
                runtime_td.append(np.mean(ep_loss_td[-runtime_range:]))
                print("MC: %d, Episode: %d, ATD runtime loss: %.3f, TD runtime loss: %.3f" 
                      % (j+1, episode+1, runtime_atd[-1], runtime_td[-1]))
          
            if j == 0:
                mc_loss_atd = np.array(ep_loss_atd)
                mc_runtime_atd = np.array(runtime_atd)
                mc_loss_td = np.array(ep_loss_td)
                mc_runtime_td = np.array(runtime_td)
            else:
                mc_loss_atd = np.vstack((mc_loss_atd, np.array(ep_loss_atd)))
                mc_runtime_atd = np.vstack((mc_runtime_atd, np.array(runtime_atd)))
                mc_loss_td = np.vstack((mc_loss_td, np.array(ep_loss_td)))
                mc_runtime_td = np.vstack((mc_runtime_td, np.array(runtime_td)))
                
        return mc_loss_atd, mc_runtime_atd, mc_loss_td, mc_runtime_td


    def _targets_(self, rewards, dones, values, next_value):
        next_values = np.append(values, next_value)
        targets = np.zeros(len(rewards))
        for t in reversed(range(rewards.shape[0])):
            targets[t] = rewards[t] + self.params['gamma'] * next_values[t+1] * (1-dones[t])
        
        return targets


if __name__ == '__main__':

    # common params
    name = 'simple_customized'
    mc = 10
    epi_max = 200
    bsize = 20
    reset_rate = 1
    render = False
    runtime_range = 50
    lam = 0.3
    
    # custom param
    ATD_params = [0.7, 0.2, 0.01]
    TD_params = 2e-1
    
    #initialize env and models
    env = make_env(name)
    trainer = TD_optimizers()
    
    # training
    mc_loss_atd, mc_runtime_atd, mc_loss_td, mc_runtime_td = trainer.train(env, ATD_params, TD_params, mc, bsize, 
                                                                           epi_max, reset_rate, lam, runtime_range, render)
    
    # save results
    with open('mc_ATD_nonlinear_'+name+'.pkl', 'wb') as f:
        pickle.dump(mc_runtime_atd, f)
    with open('mc_TD_nonlinear_'+name+'.pkl', 'wb') as f:
        pickle.dump(mc_runtime_td, f)
