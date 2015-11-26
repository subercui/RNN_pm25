# -*- coding: utf-8 -*-
import numpy as np
import cPickle
import gzip 
from profilehooks import profile
from keras.utils.train_utils import *
import theano, theano.tensor as T
from theano_lstm import LSTM, RNN, StackedCells, Layer

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
#load RNN model
RNN_model_path='RNNModel20151008.pkl.gz'
f=gzip.open(RNN_model_path, 'rb')
RNNobj.model.params=cPickle.load(f)
para_min=cPickle.load(f)
para_max=cPickle.load(f)
f.close()

def RNNpredict(pm25, gfs, pm25_mean, pred_range):
    steps=40
    #风速绝对化，记得加入
    gfs[:,2]=np.sqrt(gfs[:,2]**2+gfs[:,3]**2)
    #data scale and split
    gfs=(gfs-para_min)/(para_max-para_min)
    pm25in=pm25-pm25_mean
    pm25in=pm25in/100.
    #predict
    batch=pm25.shape[0]
    cnt=np.repeat(np.eye(steps,dtype=theano.config.floatX).reshape(1,steps,steps),batch,axis=0)
    a=RNNobj.pred_fun(gfs[None,:],pm25in[None,:],cnt)
        
    #output scaling back
    output=a*100+pm25_mean    
    return output
    
    
##########################################################################################

#def predict_pm25(past_pm25, past_gfs, future_gfs, future_pm25_mean, downsample=1):
#    if downsample == 1:
#        future_pm25 = future_pm25_mean
#    else:
#        n_future_steps
#        assert n_steps % downsample == 0
#        future_pm25 = future_pm25_mean.reshape((n_steps / downsample, downsample)).mean(axis=1)
#    return future_pm25

def pm25_mean_predict(pm25, gfs, date_time, pm25_mean, pred_range, downsample=1):
    return pm25_mean[pred_range[0]:pred_range[1]]

model = None

@profile
def mlp_predict(pm25, gfs, date_time, pm25_mean, pred_range, downsample=1):
    assert pm25.ndim == 1
    assert gfs.ndim == 2
    assert date_time.ndim == 2
    assert pm25_mean.ndim == 1
    pm25 = pm25.reshape((1, pm25.shape[0], 1))
    gfs = gfs.reshape((1, gfs.shape[0], gfs.shape[1]))
    date_time = date_time.reshape((1, date_time.shape[0], date_time.shape[1]))
    pm25_mean = pm25_mean.reshape((1, pm25_mean.shape[0], 1))
    X, y = decompose_sequences(gfs, date_time, pm25_mean, pm25, pred_range)
    n_steps = pred_range[1] - pred_range[0]
    assert X.shape == (n_steps, 24)
    assert y.shape == (n_steps, 1)
    global model
    if model is None:
        print 'loading mlp...'
        model = load_mlp()
        print 'done.'
    print 'predicting...'
    yp = model.predict_on_batch(normalize_batch(X))
    print 'done.'
    assert yp.shape == (n_steps, 1)
    return yp.flatten()
    

@profile
def mlp_predict_batch(pm25, gfs, date_time, pm25_mean, pred_range, downsample=1):
    X, y = decompose_sequences(gfs, date_time, pm25_mean, pm25, pred_range)
    n_steps = pred_range[1] - pred_range[0]
    global model
    if model is None:
        print 'loading mlp...'
        model = load_mlp()
        print 'done.'
    print 'predicting...'
    yp = model.predict_on_batch(normalize_batch(X))
    print 'done.'
    pred_pm25 = yp.reshape((n_steps, pm25.shape[0])).T
    return pred_pm25

def predict_all(data, predict_fn, pred_range=[2, 42]):
    predictions = []
    for i in range(data.shape[0]):
        pm25 = data[i, :, -1]
        gfs = data[i, :, :6]
        date_time = data[i, :, 6:-2]
        pm25_mean = data[i, :, -2]
        pred_pm25 = predict_fn(pm25, gfs, date_time, pm25_mean, pred_range)
        predictions.append(pred_pm25)
    predictions = np.array(predictions)
    return predictions

def predict_all_batch(data, predict_fn, pred_range=[2, 42], batch_size=1024):
    predictions = []
    for i in range(0, data.shape[0], batch_size):
        start = i
        stop = min(data.shape[0], i + batch_size)
        pm25 = data[start:stop, :, -1:]
        gfs = data[start:stop, :, :6]
        date_time = data[start:stop, :, 6:-2]
        pm25_mean = data[start:stop, :, -2:-1]
        pred_pm25 = predict_fn(pm25, gfs, date_time, pm25_mean, pred_range)
        predictions.append(pred_pm25)
    predictions = np.vstack(predictions)
    return predictions

def mean_square_error(predictions, targets):
    return np.square(predictions - targets).mean(axis=0)

def absolute_percent_error(predictions, targets, targets_mean):
    return (np.abs(predictions - targets) / np.abs(targets_mean)).mean(axis=0)
        
def absolute_error(predictions, targets):
    return np.abs(predictions - targets).mean(axis=0)

threshold = 80
    
def misclass_error(predictions, targets):
    return ((predictions >= threshold) != (targets >= threshold)).mean(axis=0)

def downsample(sequences, pool_size):
    assert sequences.ndim == 2
    assert sequences.shape[1] % pool_size == 0
    return sequences.reshape((sequences.shape[0], sequences.shape[1] / pool_size, pool_size)).max(axis=2)

def detection_error(predictions, targets, pool_size=1):
    if pool_size != 1:
        predictions = downsample(predictions, pool_size)
        targets = downsample(targets, pool_size)
    alarm = (predictions >= threshold).mean(axis=0)
    occur = (targets >= threshold).mean(axis=0)
    hit = ((predictions >= threshold) & (targets >= threshold)).mean(axis=0)
    pod = hit / occur
    far = 1. - hit / alarm
    csi = hit / (occur + alarm - hit)
    return pod, far, csi

def seq2point(data, pred_range):
    X = []
    y = []
    for i in range(pred_range[0], pred_range[1]):
        recent_gfs = data[:,i-2:i+1,:6].reshape((data.shape[0], -1))
        current_pm25_mean = data[:,i,6:-1]
        init_pm25 = data[:,pred_range[0]-1,-1:]
        step = np.ones((data.shape[0],1)) * (i - pred_range[0] + 1)
        Xi = np.hstack([recent_gfs, current_pm25_mean, init_pm25, step])
        yi = data[:,i,-1:]
        X.append(Xi)
        y.append(yi)
    X = np.vstack(X)
    y = np.vstack(y)
    return X, y

def decompose_sequences(gfs, date_time, pm25_mean, pm25, pred_range):
    X = []
    y = []
    for i in range(pred_range[0], pred_range[1]):
        recent_gfs = gfs[:,i-2:i+1,:].reshape((gfs.shape[0], -1))
        current_date_time = date_time[:,i,:]
        current_pm25_mean = pm25_mean[:,i,:]
        init_pm25 = pm25[:,pred_range[0]-1,:]
        step = np.ones((pm25.shape[0],1)) * (i - pred_range[0] + 1)
        Xi = np.hstack([recent_gfs, current_date_time, current_pm25_mean, init_pm25, step])
        yi = pm25[:,i,:]
        X.append(Xi)
        y.append(yi)
    X = np.vstack(X)
    y = np.vstack(y)
    return X, y

def normalize(X_train, X_test):
    X_mean = X_train.mean(axis=0)
#    print 'X_mean =', X_mean
    X_train -= X_mean
    X_stdev = np.sqrt(X_train.var(axis=0))
    X_train /= X_stdev
    X_test -= X_mean
    X_test /= X_stdev
    np.save('X_mean.npy', X_mean)
    np.save('X_stdev.npy', X_stdev)
    return X_train.astype('float32'), X_test.astype('float32')
    
def normalize_batch(Xb):
    X_mean = np.load('X_mean.npy')
    X_stdev = np.load('X_stdev.npy')
    Xb -= X_mean
    Xb /= X_stdev
    return Xb
    
def parse_data(data):
    gfs = data[:, :, :6]
    date_time = data[:, :, 6:-2]
    pm25_mean = data[:, :, -2:-1]
    pm25 = data[:, :, -1:]
    return gfs, date_time, pm25_mean, pm25
    
def build_mlp_dataset(data, pred_range=[2,42], valid_pct=1./4):
    train_pct = 1. - valid_pct
    train_data = data[:data.shape[0]*train_pct]
    valid_data = data[data.shape[0]*train_pct:]
    print 'trainset.shape, testset.shape =', train_data.shape, valid_data.shape
#    X_train, y_train = seq2point(trainset, pred_range)
#    X_valid, y_valid = seq2point(validset, pred_range)
    X_train, y_train = decompose_sequences(*(parse_data(train_data) + (pred_range,)))
    X_valid, y_valid = decompose_sequences(*(parse_data(valid_data) + (pred_range,)))
                                           
    X_train, X_valid = normalize(X_train, X_valid)
    print 'X_train.shape, y_train.shape =', X_train.shape, y_train.shape
    return X_train, y_train, X_valid, y_valid

f = gzip.open('/home/xd/data/pm25data/forXiaodaDataset20151022_t100p100.pkl.gz', 'rb')   
data = cPickle.load(f)
data[:,:,-2:] -= 80
data[:,:,2] = np.sqrt(data[:,:,2]**2 + data[:,:,3]**2)
data[:,:,3] = data[:,:,2]
f.close()

#X_train, y_train, X_valid, y_valid = build_mlp_dataset(data)
predictions = predict_all_batch(data[data.shape[0]*3./4:], mlp_predict_batch)  