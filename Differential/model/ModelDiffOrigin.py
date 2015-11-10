# -*- coding: utf-8 -*-
import theano, theano.tensor as T
import numpy as np
import theano_lstm
import random
import cPickle, gzip
from datetime import datetime
from theano_lstm import LSTM, RNN, StackedCells, Layer, create_optimization_updates
theano.config.compute_test_value = 'off'
theano.config.floatX = 'float32'
theano.config.mode='FAST_RUN'
theano.config.profile='False'
theano.config.scan.allow_gc='False'
#theano.config.device = 'gpu'

today=datetime.today()
today=today.replace(2015,9,1)

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
        self.model.layers.append(Layer(hidden_size, output_size, activation = lambda x:x))
        # inputs are matrices of indices,
        # each row is a sentence, each column a timestep
        self.steps=steps
        self.gfs=T.tensor3('gfs')#输入gfs数据
        self.pm25in=T.tensor3('pm25in')#pm25初始数据部分
        self.pm25target=T.matrix('pm25target')#输出的目标target，这一版把target维度改了
        self.layerstatus=None
        self.results=None
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
        gfs_x=T.concatenate([gfs[:,0],gfs[:,1],gfs[:,2]],axis=1)
        pm25in_x=T.concatenate([pm25in[:,0],pm25in[:,1]],axis=1)
        self.layerstatus=self.model.forward(T.concatenate([gfs_x,pm25in_x],axis=1))
        self.results=self.layerstatus[-1]
        if self.steps > 1:
            for i in xrange(1,self.steps):
                gfs_x=T.concatenate([gfs_x[:,9:],gfs[:,i+2]],axis=1)
                pm25in_x=T.concatenate([pm25in_x[:,2:],T.shape_padright(pm25in_x[:,-2]-self.results[:,i-1]),T.shape_padright(self.results[:,i-1])],axis=1)
                self.layerstatus=self.model.forward(T.concatenate([gfs_x,pm25in_x],axis=1),self.layerstatus)
                self.results=T.concatenate([self.results,self.layerstatus[-1]],axis=1)
                
        return self.results
                
        
    def create_cost_fun (self):                                 
        self.cost = (self.predictions - self.pm25target).norm(L=2)

    def create_valid_error(self):
        self.valid_error=T.mean(T.abs_(self.predictions - self.pm25target),axis=0)
                
    def create_predict_function(self):
        self.pred_fun = theano.function(inputs=[self.gfs,self.pm25in],outputs =self.predictions,allow_input_downcast=True)
                                 
    def create_training_function(self):
        updates, gsums, xsums, lr, max_norm = create_optimization_updates(self.cost, self.params, method="adadelta")#这一步Gradient Decent!!!!
        self.update_fun = theano.function(
            inputs=[self.gfs,self.pm25in, self.pm25target],
            outputs=self.cost,
            updates=updates,
            name='update_fun',
            profile=False,
            allow_input_downcast=True)
            
    def create_validate_function(self):
        self.valid_fun = theano.function(
            inputs=[self.gfs,self.pm25in, self.pm25target],
            outputs=self.valid_error,
            allow_input_downcast=True
        )
        
    def __call__(self, gfs,pm25in):
        return self.pred_fun(gfs,pm25in)
        
        
        
#############
# LOAD DATA #
#############
print '... loading data'
#dataset='/ldata/pm25data/pm25dataset/RNNPm25Dataset'+today.strftime('%Y%m%d')+'_t10p100shuffled.pkl.gz'
dataset='/data/pm25data/dataset/DiffRNNtest'+today.strftime('%Y%m%d')+'_t100p100.pkl.gz'
#dataset='/Users/subercui/48stepsRNNPm25Dataset20150920_t100p100.pkl.gz'
f=gzip.open(dataset,'rb')
data=cPickle.load(f)
print "Dataset Shape"
print data.shape
data=np.asarray(data,dtype=theano.config.floatX)
f.close()
#加入差分数据
data1=data[:,:,:-2]
data2=data[:,:,-1:]#data2 之后就可以不要了
data3=data2-np.roll(data2,1,axis=1)
data3[:,0]=0
data=np.concatenate((data,data3),axis=2)#最后一维就变成差了
#截取
data=data[:80100,-42:]


#风速绝对化，记得加入
data[:,:,2]=np.sqrt(data[:,:,2]**2+data[:,:,3]**2)
#data scale and split
para_min=np.amin(data[:,:,0:6],axis=0)#沿着0 dim example方向求最值
para_max=np.amax(data[:,:,0:6],axis=0)
data[:,:,0:6]=(data[:,:,0:6]-para_min)/(para_max-para_min)
data[:,:,(-1,-2)]=data[:,:,(-1,-2)]/100.
train_set, valid_set=np.split(data,[int(0.8*len(data))],axis=0)
np.random.shuffle(train_set)
np.random.shuffle(valid_set)

def construct(data_xy,borrow=True):#把后两维都作为pm25in
    data_gfs,data_pm25=np.split(data_xy,[data_xy.shape[2]-2],axis=2)
    data_pm25in,data_pm25target=np.split(data_pm25,[2],axis=1)
    #这里的维度改了
    data_pm25target=data_pm25target[:,:,-1].reshape(data_pm25target.shape[0],data_pm25target.shape[1])
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
steps=40
RNNobj = Model(
    input_size=9*3+2*2,
    hidden_size=40,
    output_size=1,
    stack_size=2, # make this bigger, but makes compilation slow
    celltype=LSTM, # use RNN or LSTM
    steps=steps
) 
