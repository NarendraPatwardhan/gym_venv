import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------

class StateModel(mx.gluon.Block):
    def __init__(self,config):
        super(StateModel, self).__init__()
        self.config = config
        x = mx.nd.array(self.config['S0A'])
        y = mx.nd.array(self.config['S1'])
        self.dataset = mx.gluon.data.dataset.ArrayDataset(x,y)
        self.dataloader = mx.gluon.data.DataLoader(self.dataset,batch_size=self.config['batch_size'])
        with self.name_scope():
            self.state_transition = mx.gluon.nn.Sequential('state_transition_')
            with self.state_transition.name_scope():
                self.state_transition.add(mx.gluon.nn.Dense(10, activation='relu'))
                self.state_transition.add(mx.gluon.nn.Dense(20, activation='relu'))
                self.state_transition.add(mx.gluon.nn.Dense(10, activation='relu'))
                self.state_transition.add(mx.gluon.nn.Dense(self.config['S1'].shape[1]))

    def forward(self, x):
        return self.state_transition(x)

    def fit(self):
        self.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
        criterion = mx.gluon.loss.HuberLoss()
        optimizer = mx.gluon.Trainer(self.collect_params(), 'adam',{'learning_rate': self.config['learning_rate'],'wd': self.config['weight_decay']})
        errors = []
        for epoch in range(self.config['max_epochs']):
            running_loss = 0.0
            n_total = 0.0
            for data in self.dataloader:
                x, y = data
                with mx.autograd.record():
                    output = self.forward(x)
                    loss = criterion(output, y)
                loss.backward()
                optimizer.step(self.config['batch_size'])
                running_loss += mx.nd.sum(loss).asscalar()
                n_total += x.shape[0]
            errors.append(running_loss / n_total)
            if epoch%self.config['verbosity']==0:
                print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, self.config['max_epochs'], running_loss / n_total))
        fig,ax = plt.subplots()
        ax.plot(range(len(errors)),np.array(errors))
        ax.set_title('State Modelling')
        ax.set_ylabel('Huber Loss')
        ax.set_xlabel('Epoch')
        fig.savefig('state_modelling')

#-----------------------------------------------------------------------------

class RewardModel(mx.gluon.Block):
    def __init__(self,config):
        super(RewardModel, self).__init__()
        self.config = config
        x = mx.nd.array(self.config['S0AS1'])
        y = mx.nd.array(self.config['R'])
        self.dataset = mx.gluon.data.dataset.ArrayDataset(x,y)
        self.dataloader = mx.gluon.data.DataLoader(self.dataset,batch_size=self.config['batch_size'])
        with self.name_scope():
            self.reward_function = mx.gluon.nn.Sequential('reward_function_')
            with self.reward_function.name_scope():
                self.reward_function.add(mx.gluon.nn.Dense(10, activation='relu'))
                self.reward_function.add(mx.gluon.nn.Dense(20, activation='relu'))
                self.reward_function.add(mx.gluon.nn.Dense(10, activation='relu'))
                self.reward_function.add(mx.gluon.nn.Dense(1))

    def forward(self, x):
        return self.reward_function(x)

    def fit(self):
        self.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
        criterion = mx.gluon.loss.HuberLoss()
        optimizer = mx.gluon.Trainer(self.collect_params(), 'adam',{'learning_rate': self.config['learning_rate'],'wd': self.config['weight_decay']})
        errors = []
        for epoch in range(self.config['max_epochs']):
            running_loss = 0.0
            n_total = 0.0
            for data in self.dataloader:
                x, y = data
                with mx.autograd.record():
                    output = self.forward(x)
                    loss = criterion(output, y)
                loss.backward()
                optimizer.step(self.config['batch_size'])
                running_loss += mx.nd.sum(loss).asscalar()
                n_total += x.shape[0]
            errors.append(running_loss / n_total)
            if epoch%self.config['verbosity']==0:
                print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, self.config['max_epochs'], running_loss / n_total))
        fig,ax = plt.subplots()
        ax.plot(range(len(errors)),np.array(errors))
        ax.set_title('Reward Modelling')
        ax.set_ylabel('Huber Loss')
        ax.set_xlabel('Epoch')
        fig.savefig('reward_modelling')

#-----------------------------------------------------------------------------

if __name__ == '__main__':
    x = np.random.randn(100,4)
    xt = np.random.randn(100,4)
    y = x[:,:3]
    yt = xt[:,:3]
    random_config = {
        'max_epochs': 5000,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'verbosity': 25,
        'S0A': x,
        'S1': y
    }
    random_sm = StateModel(random_config)
    random_sm.fit()
    yp = random_sm(mx.nd.array(xt))
    print(abs(yp.asnumpy() - yt).sum())

