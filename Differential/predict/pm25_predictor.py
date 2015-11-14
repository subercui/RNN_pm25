# -*- coding: utf-8 -*-
#预测模型，presict模式
#这一版本应用了差分输出的模型,预测下一小时比上一小时多出的
'''
[gfs-3h， gfs0h, gfs+3h, gfs+6h, gfs+9h,..., gfs+120h]; [pm25-3h,pm250h]+[pm25+3h,pm25+6h,...,pm25+120h]
'''
import numpy as np
import gzip
import os.path as path
import cPickle, requests, time, datetime
from pyproj import Proj
from scipy.interpolate import interp1d
import theano, theano.tensor as T
from theano_lstm import LSTM, RNN, StackedCells, Layer

import caiyun.platform.base as base
import caiyun.platform.config as cfg

cfg = cfg.load_config("pm25.cfg")

now = time.time()
yesterday = datetime.datetime.fromtimestamp(now - now % 3600 - 3600 * 24).strftime('%Y%m%d')

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
    def __init__(self, hidden_size, input_size, output_size, stack_size, celltype=RNN,steps=40):
        # declare model
        self.model = StackedCells(input_size, celltype=celltype, layers =[hidden_size]*stack_size)
        # add a classifier:
        self.model.layers.append(Layer(hidden_size, output_size, activation = lambda x:x))
        # inputs are matrices of indices,
        # each row is a sentence, each column a timestep
        self.steps=steps
        self.gfs=T.tensor3('gfs')#输入gfs数据
        self.pm25in=T.tensor3('pm25in')#pm25初始数据部分
        self.layerstatus=None
        self.results=None
        # create symbolic variables for prediction:(就是做一次整个序列完整的进行预测，得到结果是prediction)
        self.predictions = self.create_prediction()
        self.create_predict_function()        
        
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
        pm25next=pm25in[:,1]-self.results
        if self.steps > 1:
            for i in xrange(1,self.steps):
                gfs_x=T.concatenate([gfs_x[:,9:],gfs[:,i+2]],axis=1)
                pm25in_x=T.concatenate([pm25in_x[:,1:],pm25next],axis=1)
                self.layerstatus=self.model.forward(T.concatenate([gfs_x,pm25in_x],axis=1),self.layerstatus)
                self.results=T.concatenate([self.results,self.layerstatus[-1]],axis=1)
                pm25next=pm25next-self.layerstatus[-1]                
        return self.results

    def create_predict_function(self):
        self.pred_fun = theano.function(inputs=[self.gfs,self.pm25in],outputs =self.predictions,allow_input_downcast=True)
                                                              
    def __call__(self, gfs,pm25in):
        return self.pred_fun(gfs,pm25in)
        
steps=40
RNNobj = Model(
    input_size=9*3+1*2,
    hidden_size=40,
    output_size=1,
    stack_size=2, # make this bigger, but makes compilation slow
    celltype=LSTM, # use RNN or LSTM
    steps=steps
)
#load RNN model
RNN_model_path=cfg.get('pm25', 'RNN', raw=True) % {'yesterday': yesterday}
f=gzip.open(RNN_model_path, 'rb')
RNNobj.model.params=cPickle.load(f)
para_min=cPickle.load(f)
para_max=cPickle.load(f)
f.close()
        
def RNNpredict(gfs,pm25in,steps):
    #风速绝对化，记得加入
    gfs[:,2]=np.sqrt(gfs[:,2]**2+gfs[:,3]**2)
    #data scale and split
    gfs[:,:6]=(gfs[:,:6]-para_min)/(para_max[:,:6]-para_min)
    pm25in=pm25in/100.
    #predict
    a=RNNobj.pred_fun(gfs[None,:],pm25in[None,:])
    #interp
    x = np.arange(0,123,3)
    y = np.zeros(41)
    y[0]=pm25in[-1,:]
    for i in range(1,y.shape[0]):
        y[i]=y[i-1]+a[:,i-1]
    func = interp1d(x, y,'cubic')
    xnew=np.arange(1,121)
    output=func(xnew)
    
    #output scaling back
    output=output*100    
    return output

def lonlat2mercator(lon=116.3883,lat=39.3289):
    p = Proj('+proj=merc')

    radius=[17,72,54,135]
    res=10000
    longitude,latitude = p(lon,lat)
    latlng = np.array([latitude,longitude])
    y,x = np.round(np.array(p(radius[1],radius[0]))/res)
    y1,x1 = np.round(np.array(p(radius[3],radius[2]))/res)
    latlng = np.abs(np.round(latlng/res)-np.array([x1,y]))
    return latlng

def aqi_online(provider):
    return provider.pm25predict + np.array(provider.tmp[24:144])
    #return provider.pm25predict + np.array(provider.tmp[24:72])
    
def predict_onlineRNN(provider,t_predict=120):
    if 17 < provider.lat < 72 and 54 < provider.lng < 135:
        pos = lonlat2mercator(provider.lng, provider.lat) #在中国地图中的坐标

        #generate inputs
        steps=t_predict/3
        gfs = np.zeros((steps+2,6+3))#(42,6)
        pm25in=np.zeros((2,1))
        #把gfs到底是哪个时间弄清楚,应该要取145个数了
	gfs[:-1,0]=np.array(provider.tmp[21:24+t_predict:3]) + 273.0 #绝对温度
        gfs[:-1,1]=np.array(provider.rh[21:24+t_predict:3])
        gfs[:-1,2]=np.array(provider.ugrd[21:24+t_predict:3])
        gfs[:-1,3]=np.array(provider.vgrd[21:24+t_predict:3])
        gfs[:-1,4]=np.array(provider.prate[21:24+t_predict:3])
        gfs[:-1,5]=np.array(provider.tcdc[21:24+t_predict:3])
	gfs[-1,0]=provider.tmp[24+t_predict-1] + 273.0 #绝对温度
	gfs[-1,1]=provider.rh[24+t_predict-1]
	gfs[-1,2]=provider.ugrd[24+t_predict-1]
	gfs[-1,3]=provider.vgrd[24+t_predict-1]
	gfs[-1,4]=provider.prate[24+t_predict-1]
	gfs[-1,5]=provider.tcdc[24+t_predict-1]
	
	#第7，8，9维三个时间维度
	timesteps=np.arange(-3,t_predict+3,3)
	now=datetime.datetime.today()
	for i in xrange(0,steps+2):
	    point=now+datetime.timedelta(hours=timesteps[i])
	    gfs[i,6]=point.hour/24
	    gfs[i,7]=point.weekday()/7
	    gfs[i,8]=(point.month+point.day/30)/12
	
        #第10维是pm25,这一句
        pm25in[:,0]=np.array(provider.pm25data[::-1])[(-4,-1),]#真实绝对pm25

        predict = RNNpredict(gfs,pm25in,steps)#直接返回未来的pm25真实值

        #输出裁剪
        predict = np.concatenate(([provider.pm25data[0]], predict))[:t_predict]
	
        return predict * (predict > 0)
    else:
        return []
