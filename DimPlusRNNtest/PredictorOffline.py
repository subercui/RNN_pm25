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
        self.model.layers.append(Layer(hidden_size, output_size, activation = lambda x:x))
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
        gfs_x=T.concatenate([gfs[:,0],gfs[:,1],gfs[:,2]],axis=1)
        pm25in_x=T.concatenate([pm25in[:,0],pm25in[:,1]],axis=1)
        self.layerstatus=self.model.forward(T.concatenate([gfs_x,pm25in_x,self.cnt[:,:,0]],axis=1))
        self.results=self.layerstatus[-1]
        for i in xrange(1,7):#前6次（0-5），输出之前的先做的6个frame，之后第7次是第1个输出
            gfs_x=T.concatenate([gfs_x[:,9:],gfs[:,i+2]],axis=1)
            pm25in_x=T.concatenate([pm25in_x[:,1:],pm25in[:,i+1]],axis=1)
            self.layerstatus=self.model.forward(T.concatenate([gfs_x,pm25in_x,self.cnt[:,:,i]],axis=1),self.layerstatus)
            self.results=T.concatenate([self.results,self.layerstatus[-1]],axis=1)
        if self.steps > 1:
            gfs_x=T.concatenate([gfs_x[:,9:],gfs[:,9]],axis=1)
            pm25in_x=T.concatenate([pm25in_x[:,1:],T.shape_padright(self.results[:,-1])],axis=1)
            self.layerstatus=self.model.forward(T.concatenate([gfs_x,pm25in_x,self.cnt[:,:,7]],axis=1),self.layerstatus)
            self.results=T.concatenate([self.results,self.layerstatus[-1]],axis=1)
            #前传之后step-2次
            for i in xrange(2,self.steps):
                gfs_x=T.concatenate([gfs_x[:,9:],gfs[:,i+8]],axis=1)
                pm25in_x=T.concatenate([pm25in_x[:,1:],T.shape_padright(self.results[:,-1])],axis=1)
                self.layerstatus=self.model.forward(T.concatenate([gfs_x,pm25in_x,self.cnt[:,:,i+6]],axis=1),self.layerstatus)
                #need T.shape_padright???
                self.results=T.concatenate([self.results,self.layerstatus[-1]],axis=1)
        return self.results
                      
    def create_predict_function(self):
        self.pred_fun = theano.function(inputs=[self.gfs,self.pm25in,self.cnt],outputs =self.predictions,allow_input_downcast=True)
    
    def create_valid_error(self):
        self.valid_error=T.mean(T.abs_(self.predictions[:,6:46] - self.pm25target[:,6:46]),axis=0)                                   
    
    def create_validate_function(self):
        self.valid_fun = theano.function(
            inputs=[self.gfs,self.pm25in, self.pm25target,self.cnt],
            outputs=self.valid_error,
            allow_input_downcast=True
        )                                                                                                            
                                                                                                                                                                                                                                                                                                                                        
    def __call__(self, gfs,pm25in):
        return self.pred_fun(gfs,pm25in)
        
steps=40
cntsteps=steps+6
RNNobj = Model(
    input_size=9*3+1*2+cntsteps,
    hidden_size=40,
    output_size=1,
    stack_size=2, # make this bigger, but makes compilation slow
    celltype=LSTM, # use RNN or LSTM
    steps=steps
)

#load model
modeldir='/data/pm25data/model/DimPlusTest20150901.pkl.gz'
f=gzip.open(modeldir, 'rb')
#for i in range(len(RNNobj.model.layers)):
#    RNNobj.model.layers[i].params=cPickle.load(f)
RNNobj.model.params=cPickle.load(f)
for entry in RNNobj.model.params:
    print entry.get_value().shape
para_min=cPickle.load(f)
para_max=cPickle.load(f)
f.close()

#predict
#############
# LOAD DATA #
#############
print '... loading data'
#dataset='/Users/subercui/RNNPm25Dataset20150813_t100p100shuffled.pkl.gz'
#dataset='/data/pm25data/dataset/DimPlusRNNTrueTest201509010903-0929.pkl.gz'
#dataset='/data/pm25data/dataset/BeijingDimPlusTrueTest201509010903-0929.pkl.gz'
dataset='/data/pm25data/dataset/FogyRNNTrueTest201509010903-0929.pkl.gz'
f=gzip.open(dataset,'rb')
data=cPickle.load(f)
#data selection
#data=data[:,6:,(0,1,2,3,4,5,-1)]
data=np.asarray(data,dtype=theano.config.floatX)
f.close()
#风速绝对化，记得加入
data[:,:,2]=np.sqrt(data[:,:,2]**2+data[:,:,3]**2)
#data scale and split
data[:,:,0:6]=(data[:,:,0:6]-para_min)/(para_max-para_min)
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
    return data_gfs,data_pm25,data_pm25target
    
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
print '... predicting for model '+modeldir

print train_gfs.shape,train_pm25in.shape,train_pm25target.shape
batch=train_gfs.shape[0]
cnt=np.repeat(np.eye(cntsteps,dtype=theano.config.floatX).reshape(1,cntsteps,cntsteps),batch,axis=0)
valid_error=RNNobj.valid_fun(train_gfs[0:batch],train_pm25in[0:batch],train_pm25target[0:batch],cnt)
#valid_error=RNNobj.valid_fun(train_gfs,train_pm25in,train_pm25target,cnt)
print 100*valid_error
