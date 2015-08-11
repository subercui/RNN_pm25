# -*- coding: utf-8 -*-
import theano, theano.tensor as T
import numpy as np
import theano_lstm
import random

from theano_lstm import LSTM, RNN, StackedCells, Layer, create_optimization_updates, masked_loss

class Model:
    """
    Simple predictive model for forecasting words from
    sequence using LSTMs. Choose how many LSTMs to stack
    what size their memory should be, and how many
    words can be predicted.
    """
    def __init__(self, hidden_size, input_size, output_size, stack_size=1, celltype=RNN,steps=40):
        # declare model
        self.model = StackedCells(input_size, celltype=celltype, layers =[hidden_size] * stack_size)
        # add a classifier:
        self.model.layers.append(Layer(hidden_size, output_size, activation = T.tanh))
        # inputs are matrices of indices,
        # each row is a sentence, each column a timestep
        self.steps=steps
        self.gfs=T.matrix()#输入gfs数据
        self.pm25in=T.matrix()#pm25初始数据部分
        self.pm25target=T.matrix()#输出的目标target
        self.layerstatus=None
        self.results=None
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
        
    def create_prediction(self):#做一次predict的方法
        gfs=self.gfs
        pm25in=self.pm25in
        #初始第一次前传
        self.layerstatus=self.model.forward(T.concatenate([gfs[0],gfs[1],gfs[2],pm25in[0],pm25in[1]],axis=0))
        #self.results=T.concatenate([self.results,self.layerstatus[-1]],axis=0)
        #前传之后step-1次
        for i in xrange(1,self.steps):
            pm25in=T.concatenate([pm25in,T.shape_padright(self.layerstatus[-1])],axis=0)
            self.layerstatus=self.model.forward(T.concatenate([gfs[i],gfs[i+1],gfs[i+2],pm25in[i],pm25in[i+1]],axis=0),self.layerstatus)
        app=self.layerstatus
        return app
        
    def create_cost_fun (self):                                 
        self.cost = (self.predictions - self.pm25target).norm(L=2) / self.steps
        
    def create_predict_function(self):
        self.pred_fun = theano.function(inputs=[self.gfs,self.pm25in],outputs =self.predictions,allow_input_downcast=True)
                                 
    def create_training_function(self):
        updates, gsums, xsums, lr, max_norm = create_optimization_updates(self.cost, self.params, method="adadelta")#这一步Gradient Decent!!!!
        self.update_fun = theano.function(
            inputs=[self.gfs,self.pm25in, self.pm25target],
            outputs=self.cost,
            updates=updates,
            allow_input_downcast=True)
        
    def __call__(self, gfs,pm25in):
        return self.pred_fun(gfs,pm25in)
        
# construct model & theano functions:
RNNobj = Model(
    input_size=18+2,
    hidden_size=10,
    output_size=1,
    stack_size=1, # make this bigger, but makes compilation slow
    celltype=RNN, # use RNN or LSTM
    steps=2
)


gfs=np.arange(24).reshape(4,6)
#gfs=theano.shared(name='gfss',value=np.arange(24).reshape(4,6).astype(theano.config.floatX))
pm25in=np.arange(4).reshape(4,1)
#pm25in=theano.shared(name='pm25inn',value=np.arange(4).reshape(4,1).astype(theano.config.floatX))
a=RNNobj(gfs,pm25in)

