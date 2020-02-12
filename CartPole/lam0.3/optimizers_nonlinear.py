import os
import gym
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.initializers as ki
os.environ["CUDA_VISIBLE_DEVICES"]=""
        

class v_model(tf.keras.Model):
    
    def __init__(self):
        super().__init__('mlp_value')
        self.hidden1 = kl.Dense(128, activation='relu', kernel_initializer = ki.RandomNormal(mean=0.0, stddev=0.01, seed=1))
        self.hidden2 = kl.Dense(128, activation = 'relu', kernel_initializer = ki.RandomNormal(mean=0.0, stddev=0.01, seed=1))
        self.values = kl.Dense(1, name='critic_value', kernel_initializer = ki.RandomNormal(mean=0.0, stddev=0.01, seed=1))


    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        hidden_logs1 = self.hidden1(x)
        hidden_logs2 = self.hidden2(hidden_logs1)
        
        return self.values(hidden_logs2)
        
        
class TD_optimizers:

    def __init__(self):
        # hyperparameters for loss terms, gamma is the discount coefficient
        self.params = {
            'gamma': 0.99,
        }
    
    
    def train(self, env, ATD_params, TD_params, mc=10, batch_sz=32, max_episode=500, lam=0.0, runtime_range=100):
        
        for j in range(mc):
            vmodel_atd = v_model()
            vmodel_td = v_model()
            # initialize containers
            actions = np.empty((batch_sz,), dtype=np.int32)
            rewards, dones, values_atd, values_td = np.empty((4, batch_sz))
            observations = np.empty((batch_sz,) + env.observation_space.shape)
            ep_loss_atd, ep_loss_td, runtime_td, runtime_atd = [], [], [], []
            
            # initialize optimizers
            lr_atd, beta, delt = ATD_params
            n = 6
            m = [0] * n
            v = [0] * n
            z_atd = [0] * n
            lr_td = TD_params
            z_td = [0] * n
        
            # start training
            next_obs = env.reset()
            for episode in range(max_episode):
                step = 0
                for step in range(bsize):
                    observations[step] = next_obs.copy()
#                    actions[step] = env.action_space.sample()
                    actions[step] = 0
                    values_atd[step] = vmodel_atd.predict(next_obs[None, :])
                    values_td[step] = vmodel_td.predict(next_obs[None, :])
                    next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                    if dones[step]:
                        next_obs = env.reset()
                    
                # ATD update
                next_value = vmodel_atd.predict(next_obs[None, :])
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
                next_value = vmodel_td.predict(next_obs[None, :])
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
        #next_values = np.append(values, next_value)
        #targets = np.zeros(len(rewards))
        #for t in reversed(range(rewards.shape[0])):
            #targets[t] = rewards[t] + self.params['gamma'] * next_values[t+1] * (1-dones[t])

        targets = np.append(np.zeros_like(rewards), next_value)
        for t in reversed(range(rewards.shape[0])):
            targets[t] = rewards[t] + self.params['gamma'] * targets[t+1] * (1-dones[t])
        targets = targets[:-1]

        return targets


if __name__ == '__main__':

    # common params
    name = 'CartPole-v0'
    mc = 10
    epi_max = 500
    bsize = 32
    runtime_range = 10
    lam = 0.3
    
    # custom param
    ATD_params = [1.5, 0.2, 0.01]
    TD_params = 2e-2
    
    #initialize env and models
    env = gym.make(name)
    trainer = TD_optimizers()
    
    # training
    mc_loss_atd, mc_runtime_atd, mc_loss_td, mc_runtime_td = trainer.train(env, ATD_params, TD_params, mc, 
                                                                           bsize, epi_max, lam, runtime_range)
    
    # save results
    with open('mc_ATD_nonlinear_'+name+'.pkl', 'wb') as f:
        pickle.dump(mc_runtime_atd, f)
    with open('mc_TD_nonlinear_'+name+'.pkl', 'wb') as f:
        pickle.dump(mc_runtime_td, f)
