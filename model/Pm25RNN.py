# -*- coding: utf-8 -*-
import theano, theano.tensor as T
import numpy as np
import theano_lstm
import random
from theano_lstm import LSTM, RNN, StackedCells, Layer, create_optimization_updates, masked_loss
theano.config.compute_test_value = 'off'
        
class Model:
    """
    Simple predictive model for forecasting words from
    sequence using LSTMs. Choose how many LSTMs to stack
    what size their memory should be, and how many
    words can be predicted.
    """
    def __init__(self, hidden_size, input_size, output_size, stack_size=1, celltype=RNN):
        # declare model
        self.model = StackedCells(input_size, celltype=celltype, layers =[hidden_size] * stack_size)
        # add a classifier:
        self.model.layers.append(Layer(hidden_size, output_size, activation = T.tanh))
        # inputs are matrices of indices,
        # each row is a sentence, each column a timestep
        self.steps=T.iscalar('steps')
        self.gfs=T.matrix('gfs')#输入gfs数据
        self.pm25in=T.matrix('pm25in')#pm25初始数据部分
        self.pm25target=T.matrix('pm25target')#输出的目标target
        self.srng = T.shared_randomstreams.RandomStreams(np.random.randint(0, 1024))
        # create symbolic variables for prediction:(就是做一次整个序列完整的进行预测，得到结果是prediction)
        self.predictions = self.create_prediction()
        # create gradient training functions:
        self.create_cost_fun()
        self.create_training_function()
        self.create_predict_function()
        '''上面几步的意思就是先把公式写好'''
        
        
    @property
    def params(self):
        return self.model.params      
        
    def create_prediction(self):
        def oneStep(gfs_tm2,gfs_tm1,gfs_t,pm25_tm2,pm25_tm1,*prev_hiddens):
            input_x=T.concatenate([gfs_tm2,gfs_tm1,gfs_t,pm25_tm2,pm25_tm1],axis=0)
            new_states = self.model.forward(input_x, prev_hiddens)
            #错位之后返回
            return [new_states[-1]]+new_states[:-1]
        
        gfs=self.gfs
        initial_predict=self.pm25in
            
        result, updates = theano.scan(oneStep,
                          n_steps=self.steps,
                          sequences=[dict(input=gfs, taps=[-2,-1,-0])],
                          outputs_info=[dict(initial=initial_predict, taps=[-2,-1])] + [dict(initial=layer.initial_hidden_state, taps=[-1]) for layer in self.model.layers if hasattr(layer, 'initial_hidden_state')])
        #根据oneStep，result的结果list有两个元素，result[0]是new_stats[-1]即最后一层输出的array，result[1]是之前层
        return result[0]
        
    def create_cost_fun (self):                                 
        self.cost = (self.predictions - self.pm25target).norm(L=2) / self.steps
        
    def create_predict_function(self):
        self.pred_fun = theano.function(inputs=[self.gfs,self.pm25in,self.steps],outputs =self.predictions,allow_input_downcast=True)
                                 
    def create_training_function(self):
        updates, gsums, xsums, lr, max_norm = create_optimization_updates(self.cost, self.params, method="adadelta")#这一步Gradient Decent!!!!
        self.update_fun = theano.function(
            inputs=[self.gfs,self.pm25in, self.pm25target,self.steps],
            outputs=self.cost,
            updates=updates,
            allow_input_downcast=True)
        
    def __call__(self, gfs,pm25in,steps):
        return self.pred_fun(gfs,pm25in,steps)
        
# construct model & theano functions:
RNNobj = Model(
    input_size=18+2,
    hidden_size=10,
    output_size=1,
    stack_size=1, # make this bigger, but makes compilation slow
    celltype=RNN, # use RNN or LSTM
)

gfs=np.arange(24).reshape(4,6)
pm25in=np.arange(4).reshape(4,1)
#看输出a可以发现shape是（2，1），第一维2是两次step，第二维就是每次的结果了
#于是target要对应，第一维是对应每个小时，第二维是具体结果；这还是只一个example的target
pm25target=np.arange(2).reshape(2,1)
a=RNNobj(gfs,pm25in,2)
RNNobj.update_fun(gfs,pm25in,np.arange(2).reshape(2,1),2)

#for i in xrange(inputs.shape[0]):
