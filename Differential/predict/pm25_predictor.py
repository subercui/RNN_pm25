# -*- coding: utf-8 -*-
#预测模型，presict模式
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

pm25mean_path = cfg.get('pm25', 'mean', raw=True) % {'yesterday': yesterday}

with open(path.join(base.find_path(), pm25mean_path, 'meanfor0.pkl'), 'rb') as f:
    tmpPm25mean = cPickle.load(f)
    pm25mean = np.zeros((24,tmpPm25mean.shape[0],tmpPm25mean.shape[1]))
    del tmpPm25mean

for h in range(24): #取出各个小时的pm25mean备用
    with open(path.join(base.find_path(), pm25mean_path, 'meanfor%d.pkl' % h), 'rb') as f:
        pm25mean[h,:,:] = cPickle.load(f)


#load mlp120 model
mlp120_model_path = cfg.get('pm25', 'mlp120', raw=True) % {'yesterday': yesterday}

with gzip.open(path.join(base.find_path(), mlp120_model_path), 'rb') as f:
    hidden_w120= cPickle.load(f).get_value()
    hidden_b120= cPickle.load(f).get_value()
    out_w120= cPickle.load(f).get_value()
    out_b120= cPickle.load(f).get_value()
    para_min120= cPickle.load(f)
    para_max120= cPickle.load(f)

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
RNN_model_path=cfg.get('pm25', 'RNN', raw=True) % {'yesterday': yesterday}
f=gzip.open(RNN_model_path, 'rb')
RNNobj.model.params=cPickle.load(f)
para_min=cPickle.load(f)
para_max=cPickle.load(f)
f.close()
        
def mlp_predict120(inputs,t_predict):
    #inputs scaling
    #风速绝对化，实验试试看是否提高预测精度
    for i in range(8*(t_predict/24+1)):
        wind=np.sqrt(inputs[(i*6+2)]**2+inputs[(i*6+3)]**2)
        drct=inputs[(i*6+2)]/inputs[(i*6+3)]
        inputs[(i*6+2)]=wind
        #inputs[(i*6+3)]=drct
    #风速绝对值化
    inputs[0:2*t_predict+48]=np.abs(inputs[0:2*t_predict+48])
    inputs[0:2*t_predict+48]=(inputs[0:2*t_predict+48]-para_min120[0:2*t_predict+48])/(para_max120[0:2*t_predict+48]-para_min120[0:2*t_predict+48])
    inputs[2*t_predict+48:]=inputs[2*t_predict+48:]/100

    #predict
    hidden_out120=np.tanh(np.dot(inputs,hidden_w120)+hidden_b120)
    output=np.dot(hidden_out120,out_w120)+out_b120

    #output scaling back
    output=output*100
    return output

def RNNpredict(gfs,pm25in,steps):
    #风速绝对化，记得加入
    gfs[:,2]=np.sqrt(gfs[:,2]**2+gfs[:,3]**2)
    #data scale and split
    gfs=(gfs-para_min)/(para_max-para_min)
    pm25in=pm25in/100.
    #predict
    batch=1
    cnt=np.repeat(np.eye(steps,dtype=theano.config.floatX).reshape(1,steps,steps),batch,axis=0)
    a=RNNobj.pred_fun(gfs[None,:],pm25in[None,:],cnt)
    #interp
    x = np.arange(0,123,3)
    y = np.zeros(41)
    y[0]=pm25in[-1,:]
    y[1:]=a[0,:]
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


def predict_online120h(provider,t_predict=120):
    if 17 < provider.lat < 72 and 54 < provider.lng < 135:
        pos = lonlat2mercator(provider.lng, provider.lat) #在中国地图中的坐标

        #generate inputs
        inputs = np.zeros(t_predict*2+48+24)
        #第i小时,注意根据现在的输入格式，每隔3小时取一个tmp值
        inputs[0:t_predict*2+48:6]=np.array(provider.tmp[0:24+t_predict:3]) + 273.0 #绝对温度
        inputs[1:t_predict*2+48:6]=np.array(provider.rh[0:24+t_predict:3])
        inputs[2:t_predict*2+48:6]=np.array(provider.ugrd[0:24+t_predict:3])
        inputs[3:t_predict*2+48:6]=np.array(provider.vgrd[0:24+t_predict:3])
        inputs[4:t_predict*2+48:6]=np.array(provider.prate[0:24+t_predict:3])
        inputs[5:t_predict*2+48:6]=np.array(provider.tcdc[0:24+t_predict:3])

        hour = time.localtime().tm_hour #当前小时
        #生成后24维的pm25数据，时间从之前24小时到当前
        inputs[t_predict*2+48:]=np.array(provider.pm25data[:24][::-1])+80.0-np.roll(pm25mean[:,pos[0],pos[1]],-((hour-23)%24))

        predict = mlp_predict120(inputs,t_predict)
        #减80是因为，原始数据来自中国地图pm25数据，是在真实值上增加了80的
        predict = predict + np.roll(np.tile(pm25mean[:,pos[0], pos[1]],np.ceil(t_predict/24)),-((1+hour)%24)) - 80.0

        #输出裁剪
        predict = np.concatenate(([provider.pm25data[0]], predict))[:t_predict]

        return predict * (predict > 0)
    else:
        return []

def aqi_online(provider):
    return provider.pm25predict + np.array(provider.tmp[24:144])
    #return provider.pm25predict + np.array(provider.tmp[24:72])
    
def predict_onlineRNN(provider,t_predict=120):
    if 17 < provider.lat < 72 and 54 < provider.lng < 135:
        pos = lonlat2mercator(provider.lng, provider.lat) #在中国地图中的坐标

        #generate inputs
        steps=t_predict/3
        gfs = np.zeros((steps+2,6))
        pm25in=np.zeros((2,1))
        #把gfs到底是哪个时间弄清楚,应该要取145个数了
	gfs[:-1,0]=np.array(provider.tmp[21:24+t_predict:3]) + 273.0 #绝对温度
        gfs[:-1,1]=np.array(provider.rh[21:24+t_predict:3])
        gfs[:-1,2]=np.array(provider.ugrd[21:24+t_predict:3])
        gfs[:-1,3]=np.array(provider.vgrd[21:24+t_predict:3])
        gfs[:-1,4]=np.array(provider.prate[21:24+t_predict:3])
        gfs[:-1,5]=np.array(provider.tcdc[21:24+t_predict:3])
	gfs[-1,0]=provider.tmp[24+t_predict-1]
	gfs[-1,1]=provider.rh[24+t_predict-1]
	gfs[-1,2]=provider.ugrd[24+t_predict-1]
	gfs[-1,3]=provider.vgrd[24+t_predict-1]
	gfs[-1,4]=provider.prate[24+t_predict-1]
	gfs[-1,5]=provider.tcdc[24+t_predict-1]

        hour = time.localtime().tm_hour #当前小时
        #第7维是pm25,这一句
        pm25in[:,0]=np.array(provider.pm25data[::-1])[(-4,-1),]+80.0-np.roll(pm25mean[:,pos[0],pos[1]],-((hour-23)%24))[(-4,-1),]

        predict = RNNpredict(gfs,pm25in,steps)
        #减80是因为，原始数据来自中国地图pm25数据，是在真实值上增加了80的
        predict = predict + np.roll(np.tile(pm25mean[:,pos[0], pos[1]],np.ceil(t_predict/24)),-((1+hour)%24)) - 80.0

        #输出裁剪
        predict = np.concatenate(([provider.pm25data[0]], predict))[:t_predict]
	
        return predict * (predict > 0)
    else:
        return []

def predict_combine(provider,t_predict=120):
    if 17 < provider.lat < 72 and 54 < provider.lng < 135:
        MLPpredict=predict_online120h(provider,t_predict)
        RNNpredict=predict_onlineRNN(provider,t_predict)
        output=np.concatenate((MLPpredict[0:15], MLPpredict[15:25]*np.arange(1,0,-0.1)+RNNpredict[15:25]*np.arange(0,1,0.1),RNNpredict[25:]))
        return output
    else:
        return []
