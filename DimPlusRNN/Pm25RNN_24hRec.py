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

def create_shared(out_size, in_size=None, name=None):
    """
    Creates a shared matrix or vector
    using the given in_size and out_size.

    Inputs
    ------

    out_size int            : outer dimension of the
                              vector or matrix
    in_size  int (optional) : for a matrix, the inner
                              dimension.

    Outputs
    -------

    theano shared : the shared matrix, with random numbers in it

    """

    if in_size is None:
        return theano.shared(np.zeros((out_size, ),dtype=theano.config.floatX), name=name)
    else:
        return theano.shared(np.zeros((out_size, in_size),dtype=theano.config.floatX), name=name)

        
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
        self.gfs=T.tensor3('gfs')#输入gfs数据
        self.pm25in=T.tensor3('pm25in')#pm25初始数据部分
        self.pm25target=T.matrix('pm25target')#输出的目标target，这一版把target维度改了
        self.layerstatus=None
        self.results=None
        self.cnt = T.tensor3('cnt')
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
        gfs_x=T.concatenate([gfs[:,0],gfs[:,1],gfs[:,2],gfs[:,3],gfs[:,4],gfs[:,5],gfs[:,6],gfs[:,7],gfs[:,8]],axis=1)
        pm25in_x=T.concatenate([pm25in[:,0],pm25in[:,1],pm25in[:,2],pm25in[:,3],pm25in[:,4],pm25in[:,5],pm25in[:,6],pm25in[:,7]],axis=1)
        self.layerstatus=self.model.forward(T.concatenate([gfs_x,pm25in_x,self.cnt[:,:,0]],axis=1))
	#results.shape?40*1
        self.results=self.layerstatus[-1]
        if self.steps > 1:
            gfs_x=T.concatenate([gfs_x[:,6:],gfs[:,9]],axis=1)
            pm25in_x=T.concatenate([pm25in_x[:,1:],self.results],axis=1)
            self.layerstatus=self.model.forward(T.concatenate([gfs_x,pm25in_x,self.cnt[:,:,1]],axis=1),self.layerstatus)
            self.results=T.concatenate([self.results,self.layerstatus[-1]],axis=1)      
            #前传之后step-2次
            for i in xrange(2,self.steps):
                gfs_x=T.concatenate([gfs_x[:,6:],gfs[:,i+8]],axis=1)
                pm25in_x=T.concatenate([pm25in_x[:,1:],T.shape_padright(self.results[:,i-1])],axis=1)
                self.layerstatus=self.model.forward(T.concatenate([gfs_x,pm25in_x,self.cnt[:,:,i]],axis=1),self.layerstatus)
                #need T.shape_padright???
                self.results=T.concatenate([self.results,self.layerstatus[-1]],axis=1)
        return self.results
        
    def create_cost_fun (self):                                 
        self.cost = (self.predictions - self.pm25target).norm(L=2)

    def create_valid_error(self):
        self.valid_error=T.mean(T.abs_(self.predictions - self.pm25target),axis=0)
                
    def create_predict_function(self):
        self.pred_fun = theano.function(inputs=[self.gfs,self.pm25in,self.cnt],outputs =self.predictions,allow_input_downcast=True)
                                 
    def create_training_function(self):
        updates, gsums, xsums, lr, max_norm = create_optimization_updates(self.cost, self.params, method="adadelta")#这一步Gradient Decent!!!!
        self.update_fun = theano.function(
            inputs=[self.gfs,self.pm25in, self.pm25target,self.cnt],
            outputs=self.cost,
            updates=updates,
            name='update_fun',
            profile=False,
            allow_input_downcast=True)
            
    def create_validate_function(self):
        self.valid_fun = theano.function(
            inputs=[self.gfs,self.pm25in, self.pm25target,self.cnt],
            outputs=self.valid_error,
            allow_input_downcast=True
        )
        
    def __call__(self, gfs,pm25in):
        return self.pred_fun(gfs,pm25in)

#############
# LOAD DATA #
#############
print '... loading data'
today=datetime.today()
#dataset='/ldata/pm25data/pm25dataset/RNNPm25Dataset'+today.strftime('%Y%m%d')+'_t10p100shuffled.pkl.gz'
dataset='/data/pm25data/dataset/48stepsRNNPm25Dataset20150920_t100p100shuffled.pkl.gz'
#dataset='/Users/subercui/48stepsRNNPm25Dataset20150920_t100p100.pkl.gz'
f=gzip.open(dataset,'rb')
data=cPickle.load(f)
data=np.asarray(data,dtype=theano.config.floatX)
f.close()
#风速绝对化，记得加入
data[:,:,2]=np.sqrt(data[:,:,2]**2+data[:,:,3]**2)
#data scale and split
para_min=np.amin(data[:,:,0:data.shape[2]-1],axis=0)#沿着0 dim example方向求最值
para_max=np.amax(data[:,:,0:data.shape[2]-1],axis=0)
data[:,:,0:data.shape[2]-1]=(data[:,:,0:data.shape[2]-1]-para_min)/(para_max-para_min)
data[:,:,-1]=data[:,:,-1]/100.
train_set, valid_set=np.split(data,[int(0.8*len(data))],axis=0)

def construct(data_xy,borrow=True):
    data_gfs,data_pm25=np.split(data_xy,[data_xy.shape[2]-1],axis=2)
    data_pm25in,data_pm25target=np.split(data_pm25,[8],axis=1)
    #这里的维度改了
    data_pm25target=data_pm25target.reshape(data_pm25target.shape[0],data_pm25target.shape[1])
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
    input_size=6*9+1*8+steps,
    hidden_size=80,
    output_size=1,
    stack_size=4, # make this bigger, but makes compilation slow
    celltype=LSTM, # use RNN or LSTM
    steps=steps
)

###############
# TRAIN MODEL #
###############
print '... training'

batch=20
train_batches=train_set.shape[0]/batch
valid_batches=valid_set.shape[0]/batch
#cnt = np.zeros((batch,steps),dtype=theano.config.floatX)
#用代数计数法
#cnt=np.zeros((batch,1),dtype=theano.config.floatX)
#用sparse计数法，要加入
cnt=np.repeat(np.eye(steps,dtype=theano.config.floatX).reshape(1,steps,steps),batch,axis=0)
#a=RNNobj.pred_fun(train_gfs[0:20],train_pm25in[0:20])

for k in xrange(100):#run k epochs
    error_addup=0
    for i in xrange(train_batches): #an epoch
    #for i in xrange(100): #an epoch
        error_addup=RNNobj.update_fun(train_gfs[batch*i:batch*(i+1)],train_pm25in[batch*i:batch*(i+1)],train_pm25target[batch*i:batch*(i+1)],cnt)+error_addup
        if i%(train_batches/3) == 0:
	    error=error_addup/(i+1)
            print ("batch %(batch)d, error=%(error)f" % ({"batch": i+1, "error": error}))
    error=error_addup/(i+1)
    print ("   epoch %(epoch)d, error=%(error)f" % ({"epoch": k+1, "error": error}))
    
    valid_error_addup=0
    for i in xrange(valid_batches): #an epoch
    #for i in xrange(100):
        valid_error_addup=RNNobj.valid_fun(valid_gfs[batch*i:batch*(i+1)],valid_pm25in[batch*i:batch*(i+1)],valid_pm25target[batch*i:batch*(i+1)],cnt)+valid_error_addup
        if i%(valid_batches/3) == 0:
            #error=valid_error_addup/(i+1)
	    print ("batch %(batch)d, validation error:"%({"batch":i+1}))
            #print error
            #print ("batch %(batch)d, validation error=%(error)f" % ({"batch": i, "error": error}))
    error=valid_error_addup/(i+1)
    print ("epoch %(epoch)d, validation error:"%({"epoch":k+1}))
    print error
    #print ("   validation epoch %(epoch)d, validation error=%(error)f" % ({"epoch": k, "error": error}))

##############
# SAVE MODEL #
##############
savedir='/data/pm25data/model/24hRecModel0920LSTMs4h80.pkl.gz'
save_file = gzip.open(savedir, 'wb')
cPickle.dump(RNNobj.model.params, save_file, -1)
cPickle.dump(para_min, save_file, -1)#scaling paras
cPickle.dump(para_max, save_file, -1)
save_file.close()

print ('model saved at '+savedir)
