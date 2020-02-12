import os
import gym
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
os.environ["CUDA_VISIBLE_DEVICES"]=""


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class p_model(tf.keras.Model):   
    
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        self.logits = kl.Dense(num_actions, name='policy_logits', use_bias=False)
        self.sample_action = ProbabilityDistribution()


    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        
        return self.logits(x)


    def action_sampler(self, obs):
        logits = self.predict(obs)
        action = self.sample_action.predict(logits)

        return np.squeeze(action, axis=-1)
 
    
class TD_Optimizers:

    def __init__(self, vparam, pmodel):
        # hyperparameters for loss terms, gamma is the discount coefficient
        self.params = {
            'gamma': 0.95,
        }
        self.vparam = vparam
        self.pmodel = pmodel
        
    
    def train(self, env, ATD_params, ALRR_params, TD_params, mc=10, max_episode=500, bsize=32, pparam=None):
        
        for j in range(mc):
            
            ''' initialization '''
            # ATD initialization
            vparam_atd = np.copy(self.vparam)
            lr_atd, beta, delt = ATD_params
            m, v, z = 0, 0, 0
            mean_losses_atd, losses_atd = [], []
            
            # ALRR initialization
            vparam_alrr = np.copy(self.vparam)
            lr_alrr, sig, epsilon = ALRR_params
            slope = 2 * sig * (lr_alrr)**(1 - lam/2)
            x = np.arange(bsize)
            param_ini = np.copy(self.vparam)
            vparams = [self.vparam]
            mean_losses_alrr, losses_alrr = [], []
            
            # TD initialization
            vparam_td = np.copy(self.vparam)
            lr_td = TD_params
            mean_losses_td, losses_td = [], []
            
            # estimate expected initial loss
            next_obs = env.reset()
            losses_ini = []
            step = 0
            while step < 5*bsize:
                obs = next_obs.copy()
                action = self.pmodel.action_sampler(next_obs[None, :])
                # use custom initializer to control policy
                if step == 0:
                    if type(pparam) == np.ndarray:
                        self.pmodel.trainable_variables[0].assign(pparam)
                    else:
                        np.savetxt('new_policy.txt', self.pmodel.trainable_variables[0].numpy())    
                next_obs, reward, done, _ = env.step(action)
                if done:
                    next_obs = env.reset()
                    z = 0
                    continue
                step += 1
                loss_ini = (obs @ self.vparam- (reward + self.params['gamma'] * next_obs @ self.vparam))**2
                losses_ini.append(loss_ini)
            mean_loss_ini = np.mean(losses_ini)
            mean_losses_atd.append(mean_loss_ini)
            mean_losses_alrr.append(mean_loss_ini)
            mean_losses_td.append(mean_loss_ini)
            print("MC: %d, Episode: %d, ATD loss: %.5f, ALRR loss: %.5f, TD loss: %.5f" 
                  % (j+1, 0, mean_loss_ini, mean_loss_ini, mean_loss_ini))
        
            ''' start training '''
            next_obs = env.reset()
            for episode in range(max_episode):
                step = 0
                while step < bsize:
                    obs = next_obs.copy()
                    action = self.pmodel.action_sampler(next_obs[None, :])
                    next_obs, reward, done, _ = env.step(action)
                    if done:
                        next_obs = env.reset()
                        z = 0
                        continue
                    step += 1
                    z = self.params['gamma'] * lam * z + obs  
                    
                    # ATD update
                    gradient = reward * z + self.params['gamma'] * (next_obs @ vparam_atd) * z - obs @ vparam_atd * z
                    m = beta * m + (1 - beta) * gradient
                    v = v + np.linalg.norm(gradient)**2
                    vparam_atd = vparam_atd + lr_atd * m / np.sqrt(delt + v)
                    loss_atd = (obs @ vparam_atd - (reward + self.params['gamma'] * next_obs @ vparam_atd))**2
                    losses_atd.append(loss_atd)
                    
                    # ALRR update
                    gradient = reward * z + self.params['gamma'] * (next_obs @ vparam_alrr) * z - obs @ vparam_alrr * z
                    vparam_alrr = vparam_alrr + lr_alrr * gradient
                    vparams.append(vparam_alrr)
                    if len(vparams) >= bsize:
                        l = len(vparams)
                        y = np.linalg.norm(np.array(vparams[l-bsize:]) - param_ini, axis=-1)
                        slope = np.polyfit(x, y, deg=1)[0]
                    if slope < ((sig * (lr_alrr)**(1 - lam/2)) / bsize):
                        lr_alrr /= epsilon
                        param_ini = np.copy(vparam_alrr)
                    loss_alrr = (obs @ vparam_alrr - (reward + self.params['gamma'] * next_obs @ vparam_alrr))**2
                    losses_alrr.append(loss_alrr)
                    
                    # TD update
                    gradient = reward * z + self.params['gamma'] * (next_obs @ vparam_td) * z - obs @ vparam_td * z
                    vparam_td = vparam_td + lr_td * gradient
                    loss_td = (obs @ vparam_td - (reward + self.params['gamma'] * next_obs @ vparam_td))**2
                    losses_td.append(loss_td)
                 
                    
                mean_loss_atd = np.mean(losses_atd[(len(losses_atd)-bsize):])
                mean_losses_atd.append(mean_loss_atd)
                mean_loss_alrr = np.mean(losses_alrr[(len(losses_alrr)-bsize):])
                mean_losses_alrr.append(mean_loss_alrr)
                mean_loss_td = np.mean(losses_td[(len(losses_td)-bsize):])
                mean_losses_td.append(mean_loss_td)
                print("MC: %d, Episode: %d, ATD loss: %f, ALRR loss: %f, TD loss: %f" 
                      % (j+1, episode+1, mean_loss_atd, mean_loss_alrr, mean_loss_td))
            
            if j == 0:
                mc_atd = np.array(mean_losses_atd)
                mc_alrr = np.array(mean_losses_alrr)
                mc_td = np.array(mean_losses_td)
            else:
                mc_atd = np.vstack((mc_atd, np.array(mean_losses_atd)))
                mc_alrr = np.vstack((mc_alrr, np.array(mean_losses_alrr)))
                mc_td = np.vstack((mc_td, np.array(mean_losses_td)))
        
        return mc_atd, mc_alrr, mc_td


if __name__ == '__main__':
    
    # common params
    name = 'Acrobot-v1'
    mc = 10
    epi_max = 1000
    bsize = 48
    lam = 0.3
    
    # custom param
    ATD_params = [9, 0.5, 1]
    ALRR_params = [1e-3, 0.001, 1.2]
    TD_params = [4e-2]
    
    # initialize env and models
    env = gym.make(name)
    vparam = np.zeros(env.observation_space.shape[0])
    try:
        pparam = np.loadtxt('policy.txt', dtype=np.float32)
    except:
        pparam = None
    pmodel = p_model(env.action_space.n)
    optimizers = TD_Optimizers(vparam, pmodel)
    
    # training
    mc_atd, mc_alrr, mc_td = optimizers.train(env, ATD_params, ALRR_params, TD_params, mc, 
                                                    epi_max, bsize, pparam)
    
    # save results
    with open('mc_ATD_linear_'+name+'.pkl', 'wb') as f:
        pickle.dump(mc_atd, f)
    with open('mc_ALRR_linear_'+name+'.pkl', 'wb') as f:
        pickle.dump(mc_alrr, f)
    with open('mc_TD_linear_'+name+'.pkl', 'wb') as f:
        pickle.dump(mc_td, f)
