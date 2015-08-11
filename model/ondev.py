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
    def __init__(self, hidden_size, input_size, output_size, stack_size=1, celltype=RNN):
        # declare model
        self.model = StackedCells(input_size, celltype=celltype, layers =[hidden_size] * stack_size)
        # add a classifier:
        self.model.layers.append(Layer(hidden_size, output_size, activation = T.tanh))
        # inputs are matrices of indices,
        # each row is a sentence, each column a timestep
        self.steps=T.iscalar()
        self.gfs=T.matrix()#输入gfs数据
        self.pm25in=T.matrix()#pm25初始数据部分
        self.pm25target=T.matrix()#输出的目标target
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
        
    '''def create_prediction(self):
        def oneStep(gfs_tm2,gfs_tm1,gfs_t,pm25_in,pm25_tm1,*hidden_states):
            input_x=gfs_tm2+gfs_tm1+gfs_t+pm25_in+pm25_tm1
            new_hiddens=list(hidden_states)
            layers_out = self.model.forward(input_x, prev_hiddens = new_hiddens)
            #这一步更新!!!!,这里input_x和previous_hidden应该是放在outputinfo里进行迭代的
            y_given_x=layers_out[-1]#每一层的结果都有输出，最后一层就是输出层了，这里就是输出了下一帧pm25
            hiddens=layers_out
            return [y_given_x]+hiddens
            
        #按下面三行描述规则排序，预测的那一时刻帧为0
        # in sequence forecasting scenario we take everything
        # up to the before last step, and predict subsequent
        # steps ergo, 0 ... n - 1, hence:
        gfs=self.gfs
        pm25in=self.pm25in
        pm250=self.pm250
        hiddens0=[initial_state_with_taps(layer,1) for layer in self.model.layers]
        #这个函数是自动按照scan的格式，已经把taps=-1加上了，所以之后在scan里就直接写进去了
        
        # pass this to Theano's recurrence relation function:
        
        # choose what gets outputted at each timestep:
        outputs_info = [dict(initial=pm250, taps=[-1])]+hiddens0
        result, _ = theano.scan(fn=oneStep,
                            sequences=[dict(input=gfs, taps=[-2,-1,0]),pm25in],
                            outputs_info=outputs_info,
                            n_steps=self.steps)
                                 

        return result[0]#每一次y_given_x组成的list
        # we reorder the predictions to be:
        # 1. what row / example
        # 2. what timestep
        # 3. softmax dimension'''
        
    def create_prediction(self):
        def oneStep(gfs_tm2,gfs_tm1,gfs_t,pm25_tm2,pm25_tm1,*prev_hiddens):
            input_x=gfs_tm2+gfs_tm1+gfs_t+pm25_tm2+pm25_tm1
            new_states = self.model.forward(input_x, prev_hiddens)
            #错位之后返回
            return [new_states[-1]]+new_states[:-1]
        
        gfs=self.gfs
        initial_predict=self.pm25in
            
        result, updates = theano.scan(oneStep,
                          n_steps=self.steps,
                          sequences=[dict(input=gfs, taps=[-2,-1,-0])],
                          outputs_info=[dict(initial=initial_predict, taps=[-2,-1])] + [dict(initial=layer.initial_hidden_state, taps=[-1]) for layer in self.model.layers if hasattr(layer, 'initial_hidden_state')])
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


##!!!!!数据第一轴是steps方向
gfs=np.arange(24).reshape(4,6)
#gfs=theano.shared(name='gfss',value=np.arange(24).reshape(4,6).astype(theano.config.floatX))
pm25in=np.arange(1,5).reshape(4,1)
#pm25in=theano.shared(name='pm25inn',value=np.arange(4).reshape(4,1).astype(theano.config.floatX))
a=RNNobj.pred_fun(pm25in,gfs,2)
