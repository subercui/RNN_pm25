# -*- coding: utf-8 -*-
import theano, theano.tensor as T
import numpy as np
import theano_lstm
import random
import cPickle, gzip
from datetime import datetime
from theano_lstm import LSTM, RNN, StackedCells, Layer, create_optimization_updates, masked_loss
theano.config.compute_test_value = 'off'
theano.config.floatX = 'float32'
theano.config.mode='FAST_RUN'
#theano.config.profile='True'
theano.config.scan.allow_gc='False'
#theano.config.device = 'gpu'
        
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
        self.create_valid_error()
        self.create_training_function()
        self.create_predict_function()
        self.create_validate_function()
        '''上面几步的意思就是先把公式写好'''
        
        
    @property
    def params(self):
        return self.model.params
        
    def create_prediction(self):#做一次predict的方法
        gfs=self.gfs
        pm25in=self.pm25in
        #初始第一次前传
        self.layerstatus=self.model.forward(T.concatenate([gfs[0],gfs[1],gfs[2],pm25in[0],pm25in[1]],axis=0))
        self.results=T.shape_padright(self.layerstatus[-1])
        if self.steps > 1:
            self.layerstatus=self.model.forward(T.concatenate([gfs[1],gfs[2],gfs[3],pm25in[1],self.results[0]],axis=0),self.layerstatus)
            self.results=T.concatenate([self.results,T.shape_padright(self.layerstatus[-1])],axis=0)      
            #前传之后step-2次
            for i in xrange(2,self.steps):
                self.layerstatus=self.model.forward(T.concatenate([gfs[i],gfs[i+1],gfs[i+2],self.results[i-2],self.results[i-1]],axis=0),self.layerstatus)
                #need T.shape_padright???
                self.results=T.concatenate([self.results,T.shape_padright(self.layerstatus[-1])],axis=0)
        return self.results
        
    def create_cost_fun (self):                                 
        self.cost = (self.predictions - self.pm25target).norm(L=2) / self.steps

    def create_valid_error(self):
        self.valid_error=T.abs_(self.predictions - self.pm25target)
                
    def create_predict_function(self):
        self.pred_fun = theano.function(inputs=[self.gfs,self.pm25in],outputs =self.predictions,allow_input_downcast=True)
                                 
    def create_training_function(self):
        updates, gsums, xsums, lr, max_norm = create_optimization_updates(self.cost, self.params, method="adadelta")#这一步Gradient Decent!!!!
        self.update_fun = theano.function(
            inputs=[self.gfs,self.pm25in, self.pm25target],
            outputs=self.cost,
            updates=updates,
            name='update_fun',
            profile=True,
            allow_input_downcast=True)
            
    def create_validate_function(self):
        self.valid_fun = theano.function(
            inputs=[self.gfs,self.pm25in, self.pm25target],
            outputs=self.valid_error,
            allow_input_downcast=True
        )
        
    def __call__(self, gfs,pm25in):
        return self.pred_fun(gfs,pm25in)
'''        
# construct model & theano functions:
RNNobj = Model(
    input_size=18+2,
    hidden_size=10,
    output_size=1,
    stack_size=1, # make this bigger, but makes compilation slow
    celltype=RNN, # use RNN or LSTM
    steps=2
)
'''

#############
# LOAD DATA #
#############
print '... loading data'
today=datetime.today()
#dataset='/ldata/pm25data/pm25dataset/RNNPm25Dataset'+today.strftime('%Y%m%d')+'_t10p100shuffled.pkl.gz'
dataset='/data/pm25data/dataset/RNNPm25Dataset20150813_t100p100shuffled.pkl.gz'
#dataset='/Users/subercui/RNNPm25Dataset20150813_t100p100shuffled.pkl.gz'
f=gzip.open(dataset,'rb')
data=cPickle.load(f)
data=np.asarray(data,dtype=theano.config.floatX)
f.close()
#风速绝对化，记得加入
#data scale and split
para_min=np.amin(data[:,:,0:data.shape[2]-1],axis=0)#沿着0 dim example方向求最值
para_max=np.amax(data[:,:,0:data.shape[2]-1],axis=0)
data[:,:,0:data.shape[2]-1]=(data[:,:,0:data.shape[2]-1]-para_min)/(para_max-para_min)
data[:,:,-1]=data[:,:,-1]/100.
train_set, valid_set=np.split(data,[int(0.8*len(data))],axis=0)

def construct(data_xy,borrow=True):
    data_gfs,data_pm25=np.split(data_xy,[data_xy.shape[2]-1],axis=2)
    data_pm25in,data_pm25target=np.split(data_pm25,[2],axis=1)
    data_pm25target=data_pm25target.reshape(data_pm25target.shape[0],data_pm25target.shape[1],1)
    #加入shared构造，记得加入,theano禁止调用
    data_gfs=np.asarray(data_gfs,dtype=theano.config.floatX)
    data_pm25in=np.asarray(data_pm25in,dtype=theano.config.floatX)
    data_pm25target=np.asarray(data_pm25target,dtype=theano.config.floatX)
    return data_gfs,data_pm25in,data_pm25target
    
train_gfs,train_pm25in,train_pm25target=construct(train_set)
valid_gfs,valid_pm25in,valid_pm25target=construct(valid_set)
                
######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'

RNNobj = Model(
    input_size=18+2,
    hidden_size=40,
    output_size=1,
    stack_size=2, # make this bigger, but makes compilation slow
    celltype=LSTM, # use RNN or LSTM
    steps=40
)

###############
# TRAIN MODEL #
###############
print '... training'

for k in xrange(100):#run k epochs
    error_addup=0
    for i in xrange(train_set.shape[0]): #an epoch
    #for i in xrange(100): #an epoch
        error_addup=RNNobj.update_fun(train_gfs[i],train_pm25in[i],train_pm25target[i])+error_addup
        if i%(train_set.shape[0]/3) == 0 and i >0:
	    error=error_addup/i
            print ("batch %(batch)d, error=%(error)f" % ({"batch": i, "error": error}))
    error=error_addup/i
    print ("   epoch %(epoch)d, error=%(error)f" % ({"epoch": k+1, "error": error}))
    
    valid_error_addup=0
    for i in xrange(valid_set.shape[0]): #an epoch
    #for i in xrange(100):
        valid_error_addup=RNNobj.valid_fun(valid_gfs[i],valid_pm25in[i],valid_pm25target[i])+valid_error_addup
        if i%(valid_set.shape[0]/3) == 0 and i >0:
            #error=valid_error_addup/i
	    print ("batch %(batch)d, validation error:"%({"batch":i}))
            #print error.transpose()
            #print ("batch %(batch)d, validation error=%(error)f" % ({"batch": i, "error": error}))
    error=valid_error_addup/i
    print ("epoch %(epoch)d, validation error:"%({"epoch":k+1}))
    print error.transpose()
    #print ("   validation epoch %(epoch)d, validation error=%(error)f" % ({"epoch": k, "error": error}))
'''

gfs=np.arange(24).reshape(4,6)
#gfs=theano.shared(name='gfss',value=np.arange(24).reshape(4,6).astype(theano.config.floatX))
pm25in=np.arange(4).reshape(4,1)
#pm25in=theano.shared(name='pm25inn',value=np.arange(4).reshape(4,1).astype(theano.config.floatX))
a=RNNobj(gfs,pm25in)
pm25target=np.arange(2).reshape(2,1)
RNNobj.update_fun(gfs,pm25in,pm25target)
'''
