from __future__ import absolute_import
import numpy as np
import cPickle
import gzip 
np.random.seed(1337)  # for reproducibility

#from keras.datasets import mnist
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, TimeDistributedDense, Dropout, Activation
from keras.layers.recurrent import LSTM, SimpleRNN,GRU,JZS1, JZS2,JZS3
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from keras.initializations import uniform
from keras.regularizers import l2, activity_l2
from keras.callbacks import EarlyStopping, ModelCheckpoint


def build_rnn(in_dim, out_dim, h0_dim, h1_dim=None, layer_type=LSTM, truncate_gradient=-1,return_sequences=False):
    model = Sequential() 
    print in_dim,out_dim,h0_dim,h1_dim 
    model.add(layer_type(h0_dim, input_shape=(None, in_dim),truncate_gradient=truncate_gradient,return_sequences=(h1_dim is not None or return_sequences)))  
    if h1_dim is not None:
        model.add(layer_type(h1_dim, return_sequences=return_sequences))
    if return_sequences:
        model.add(TimeDistributedDense(out_dim,W_regularizer=l2(0.0005)))
    else:
        model.add(Dense(out_dim,W_regularizer=l2(0.0005)))  
    model.add(Activation("linear"))
    #model.add(Dropout(0.2))  
    model.compile(loss="mse", optimizer="rmsprop")  
    return model
    
def train(X_train, y_train, X_test, y_test, model, batch_size=128, nb_epoch=300):
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    checkpointer = ModelCheckpoint(filepath="LSTM_weights.hdf5", verbose=1, save_best_only=True)
    model.fit(X_train, y_train, 
              batch_size=batch_size, 
              nb_epoch=nb_epoch, 
              show_accuracy=False, 
              verbose=2, 
              validation_data=(X_test, y_test), 
              callbacks=[early_stopping, checkpointer])
              
def build_mlp_dataset(data, pred_range=[2,42], valid_pct=1./4):
    train_pct = 1. - valid_pct
    train_data = data[:data.shape[0]*train_pct]
    valid_data = data[data.shape[0]*train_pct:]
    print "trainset.shape, testset.shape =", train_data.shape, valid_data.shape
#    X_train, y_train = seq2point(trainset, pred_range)
#    X_valid, y_valid = seq2point(validset, pred_range)
    X_train, y_train = decompose_sequences(*(parse_data(train_data) + (pred_range,)))
    X_valid, y_valid = decompose_sequences(*(parse_data(valid_data) + (pred_range,)))
                                           
    X_train, X_valid = normalize(X_train, X_valid)
    print 'X_train.shape, y_train.shape =', X_train.shape, y_train.shape
    return X_train, y_train, X_valid, y_valid
    
def decompose_sequences(gfs, date_time, pm25_mean, pm25, pred_range):
    X = []
    y = []
    for i in range(pred_range[0], pred_range[1]):
        recent_gfs = gfs[:,i-2:i+1,:].reshape((gfs.shape[0], -1))
        #recent_gfs = gfs[:,i:i+1,:].reshape((gfs.shape[0], -1))
        current_date_time = date_time[:,i,:]
        current_pm25_mean = pm25_mean[:,i,:]
        init_pm25 = pm25[:,pred_range[0]-1,:]
        step = np.ones((pm25.shape[0],1)) * (i - pred_range[0] + 1)
        Xi = np.hstack([recent_gfs, current_date_time, current_pm25_mean, init_pm25,step])
        yi = pm25[:,i,:]
        X.append(Xi)
        y.append(yi)
    X = np.dstack(X).transpose((0,2,1))
    y = np.dstack(y).transpose((0,2,1))
    print 'decompose',X.shape,y.shape
    return X, y
    
def normalize(X_train, X_test):
    X_mean = X_train.mean(axis=0).mean(axis=0)[None,None,:]
    #print 'X_mean =', X_mean,X_mean.shape
    #print 'X_test_original=',X_test[:1024]
    X_train -= X_mean
    X_stdev = np.sqrt(X_train.reshape((X_train.shape[0]*X_train.shape[1],X_train.shape[2])).var(axis=0))[None,None,:]
    #print 'X_stdev=',X_stdev.shape
    X_train /= X_stdev
    X_test -= X_mean
    X_test /= X_stdev
    #print 'X_test=',X_test[:1024]
    np.save('X_mean.npy', X_mean)
    np.save('X_stdev.npy', X_stdev)
    return X_train.astype('float32'), X_test.astype('float32')
    
def parse_data(data):
    gfs = data[:, :, :6]
    date_time = data[:, :, 6:-2]
    pm25_mean = data[:, :, -2:-1]
    pm25 = data[:, :, -1:]
    return gfs, date_time, pm25_mean, pm25
              
if __name__=='__main__':
    f = gzip.open('/data/pm25data/dataset/forXiaodaDataset20151022_t100p100.pkl.gz', 'rb')   
    data = cPickle.load(f)
    data[:,:,-2:] -= 80
    data[:,:,2] = np.sqrt(data[:,:,2]**2 + data[:,:,3]**2)
    data[:,:,3] = data[:,:,2]
    f.close()
    X_train, y_train, X_valid, y_valid = build_mlp_dataset(data)
    LSTMmodel=build_rnn(X_train.shape[-1] ,y_train.shape[-1], 20, None,layer_type=JZS1,truncate_gradient=-1,return_sequences=True)
    #print X_train[:1024],y_train[:1024],X_valid[:1024],y_valid[:1024]
    #print X_train[:1024].mean(axis=0)
    train(X_train, y_train, X_valid, y_valid, LSTMmodel, batch_size=128)
