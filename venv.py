import numpy as np
import mxnet as mx
import gym
from collectdata import DataCollector
from model import StateModel, RewardModel

default_config = {
    'env_name': 'CartPole-v0',
    'n_runs':1000,
}

class VirtualEnv(gym.Env):
    def __init__(self,config=default_config,premade=False):
        self.env_name = config['env_name']
        self.originalEnv = gym.make(config['env_name'])
        self.observation_space = self.originalEnv.observation_space
        self.action_space = self.originalEnv.action_space
        self.max_step_count = 200
        if not premade:
            print('Collecting Data')    
            data_collector = DataCollector(config)
            S0,A,S1,R = data_collector.collect()
            S0A = np.concatenate((S0,A),axis=1)
            S0AS1 = np.concatenate((S0,A,S1),axis=1)
            r = R
            modeller_config = {
                'max_epochs': 100,
                'batch_size': 256,
                'learning_rate': 1e-3,
                'weight_decay': 1e-5,
                'verbosity': 25,
                'S0A': S0A,
                'S1': S1,
                'S0AS1': S0AS1,
                'R': R
            }

            print('Fitting State Transition Model')
            self.sm = StateModel(modeller_config)
            self.sm.fit()
            
            print('Fitting Reward Function Model')
            self.rm = RewardModel(modeller_config)
            self.rm.fit()
            sm_file_name = self.env_name+'_'+'sm.params'
            self.sm.save_parameters(sm_file_name)
            rm_file_name = self.env_name+'_'+'rm.params'
            self.rm.save_parameters(rm_file_name)            

    def seed(self,value):
        self.originalEnv.seed(value)

    def reset(self):
        self.step_count = 0
        self.state = self.originalEnv.reset()
        return self.state

    def step(self,action):
        self.step_count += 1
        s0a = mx.nd.array(np.atleast_2d(np.append(self.state,action)))
        next_state = self.sm(s0a)
        s0as1 = mx.nd.concat(s0a,next_state,dim=1)
        reward = self.rm(s0as1)
        self.state = next_state.asnumpy()
        if self.step_count < self.max_step_count:
	        done = False or self.check_done(self.state)
        else:
	        done = True
        return self.state,reward.asnumpy(),done,{}

    def check_done(self,state):
        """
            Need to be implemented for each environment seperately
        """
        if self.env_name == 'Pendulum-v0':
	        return False
        if self.env_name == 'Acrobot-v1':
	        return False

        if self.env_name == 'CartPole-v0':
	        if (abs(state[0,0]) > 2.4) or (abs(state[0,2]) > 1.2e-1):
		        return True
	        else:
		        return False

if __name__ == '__main__':
    v = VirtualEnv()
    original = gym.make('CartPole-v0')
    done_original = False
    v.seed(0)
    original.seed(0)
    v.reset()
    original.reset()
    while not doneo:
        action = v.action_space.sample()
        s,r,done,i = v.step(action)
        s_original,r_original,done_original,i_original = original.step(action)
        print(done,done_original)
