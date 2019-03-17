import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt
import gym

#-----------------------------------------------------------------------------

class DataCollector:
    def __init__(self,config):
        self.config = config
        self.env = gym.make(self.config['env_name'])
        
    def collect(self):
        S0 = []
        A = []
        S1 = []
        R = []
        for run in range(self.config['n_runs']):
            done = False
            step = 0
            state = self.env.reset()
            while (step!=200) and (not done):
                S0.append(state)
                action = self.env.action_space.sample()
                A.append(action)
                state,reward,done,info = self.env.step(action)
                S1.append(state)
                R.append(reward)                
        S0 = np.array(S0)
        A = np.array(A).reshape((len(A),1))
        S1 = np.array(S1)
        R = np.array(R).reshape((len(R),1))
        return S0,A,S1,R

#-----------------------------------------------------------------------------

if __name__ == '__main__':
    cartpole_config = {
        'env_name': 'CartPole-v0',
        'n_runs':100
    }
    cartpole = DataCollector(cartpole_config)
    S0,A,S1,R = cartpole.collect()
    x = np.concatenate((S0,A),axis=1)
    y = S1
    print(x.shape,y.shape)

