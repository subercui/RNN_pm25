# -*- coding: utf-8 -*-
import theano, theano.tensor as T
import numpy as np
import theano_lstm
import random
import cPickle, gzip
from datetime import datetime
from scipy.interpolate import interp1d
from theano_lstm import LSTM, RNN, StackedCells, Layer

#construct class
theano.config.compute_test_value = 'off'
theano.config.floatX = 'float32'
theano.config.mode='FAST_RUN'
theano.config.profile='False'
theano.config.scan.allow_gc='False'

class Model(object):
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
        self.gfs=T.tensor3('gfs')#输入gfs数据
        self.pm25in=T.tensor3('pm25in')#pm25初始数据部分
        self.layerstatus=None
        self.results=None
        self.cnt = T.tensor3('cnt')
        # create symbolic variables for prediction:(就是做一次整个序列完整的进行预测，得到结果是prediction)
        self.predictions = self.create_prediction()
        self.create_predict_function()
        self.pm25target=T.matrix('pm25target')#输出的目标target，这一版把target维度改了
        self.create_valid_error()
        self.create_validate_function()
        '''上面几步的意思就是先把公式写好'''
        
        
    @property
    def params(self):
        return self.model.params
        
    def create_prediction(self):#做一次predict的方法
        gfs=self.gfs
        pm25in=self.pm25in
        #初始第一次前传
        self.layerstatus=self.model.forward(T.concatenate([gfs[:,0],gfs[:,1],gfs[:,2],pm25in[:,0],pm25in[:,1],self.cnt[:,:,0]],axis=1))
        #results.shape?40*1
        self.results=self.layerstatus[-1]
        if self.steps > 1:
            self.layerstatus=self.model.forward(T.concatenate([gfs[:,1],gfs[:,2],gfs[:,3],pm25in[:,1],self.results,self.cnt[:,:,1]],axis=1),self.layerstatus)
            self.results=T.concatenate([self.results,self.layerstatus[-1]],axis=1)      
            #前传之后step-2次
            for i in xrange(2,self.steps):
                self.layerstatus=self.model.forward(T.concatenate([gfs[:,i],gfs[:,i+1],gfs[:,i+2],T.shape_padright(self.results[:,i-2]),T.shape_padright(self.results[:,i-1]),self.cnt[:,:,i]],axis=1),self.layerstatus)
                #need T.shape_padright???
                self.results=T.concatenate([self.results,self.layerstatus[-1]],axis=1)
        return self.results
                      
    def create_predict_function(self):
        self.pred_fun = theano.function(inputs=[self.gfs,self.pm25in,self.cnt],outputs =self.predictions,allow_input_downcast=True)
    
    def create_valid_error(self):
        self.valid_error=T.mean(T.abs_(self.predictions - self.pm25target),axis=0)                                   
    
    def create_validate_function(self):
        self.valid_fun = theano.function(
            inputs=[self.gfs,self.pm25in, self.pm25target,self.cnt],
            outputs=self.valid_error,
            allow_input_downcast=True
        )                                                                                                            
                                                                                                                                                                                                                                                                                                                                        
    def __call__(self, gfs,pm25in):
        return self.pred_fun(gfs,pm25in)
        
steps=40
RNNobj = Model(
    input_size=18+2+steps,
    hidden_size=40,
    output_size=1,
    stack_size=2, # make this bigger, but makes compilation slow
    celltype=LSTM, # use RNN or LSTM
    steps=steps
)

#load model
f=gzip.open('/Users/subercui/DetachValidModel20150901.pkl.gz', 'rb')
#for i in range(len(RNNobj.model.layers)):
#    RNNobj.model.layers[i].params=cPickle.load(f)
RNNobj.model.params=cPickle.load(f)
para_min=cPickle.load(f)
para_max=cPickle.load(f)
f.close()

#predict
#############
# LOAD DATA #
#############
print '... loading data'
today=datetime.today()
#dataset='/Users/subercui/RNNPm25Dataset20150813_t100p100shuffled.pkl.gz'
dataset='/Users/subercui/Git/RNN_pm25/test/RNNTrueTest201509010903-0929.pkl.gz'
f=gzip.open(dataset,'rb')
data=cPickle.load(f)
data=np.asarray(data,dtype=theano.config.floatX)
f.close()
#风速绝对化，记得加入
data[:,:,2]=np.sqrt(data[:,:,2]**2+data[:,:,3]**2)
#data scale and split
data[:,:,0:data.shape[2]-1]=(data[:,:,0:data.shape[2]-1]-para_min)/(para_max-para_min)
data[:,:,-1]=data[:,:,-1]/100.
train_set, valid_set=np.split(data,[int(0.99*len(data))],axis=0)

def construct(data_xy,borrow=True):
    data_gfs,data_pm25=np.split(data_xy,[data_xy.shape[2]-1],axis=2)
    data_pm25in,data_pm25target=np.split(data_pm25,[2],axis=1)
    #这里的维度改了
    data_pm25target=data_pm25target.reshape(data_pm25target.shape[0],data_pm25target.shape[1])
    #加入shared构造，记得加入,theano禁止调用
    data_gfs=np.asarray(data_gfs,dtype=theano.config.floatX)
    data_pm25in=np.asarray(data_pm25in,dtype=theano.config.floatX)
    data_pm25target=np.asarray(data_pm25target,dtype=theano.config.floatX)
    return data_gfs,data_pm25in,data_pm25target
    
train_gfs,train_pm25in,train_pm25target=construct(train_set)
valid_gfs,valid_pm25in,valid_pm25target=construct(valid_set)

###########
# Predict #
###########
'''print '... predicting'

batch=1
cnt=np.repeat(np.eye(steps,dtype=theano.config.floatX).reshape(1,steps,steps),batch,axis=0)
#a=RNNobj.pred_fun(train_gfs[0:20],train_pm25in[0:20])
testpm25in=np.array(train_pm25in[0][None,:])
testpm25in[0,0],testpm25in[0,1]=1.,1.
a=RNNobj.pred_fun(train_gfs[0][None,:],testpm25in,cnt)
b=100*a
print testpm25in
print b
#interp
x = np.arange(0,123,3)
y = np.zeros(41)
y[0]=train_pm25in[0][-1,:]
y[1:]=a[0,:]
func = interp1d(x, y,'cubic')
xnew=np.arange(1,121)
output=func(xnew)'''

############
# Validate #
############
print '... predicting'

print train_gfs.shape
batch=train_gfs.shape[0]
cnt=np.repeat(np.eye(steps,dtype=theano.config.floatX).reshape(1,steps,steps),batch,axis=0)
valid_error=RNNobj.valid_fun(train_gfs[0:batch],train_pm25in[0:batch],train_pm25target[0:batch],cnt)
#valid_error=RNNobj.valid_fun(train_gfs,train_pm25in,train_pm25target,cnt)
print 100*valid_error
