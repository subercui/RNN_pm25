# -*- coding: utf-8 -*-
#大雾霾真实测试集
'''
[gfs-3h， gfs0h, gfs+3h, gfs+6h, gfs+9h,..., gfs+120h]; [pm25-3h,pm250h]+[pm25+3h,pm25+6h,...,pm25+120h]
'''
#并注意到全球gfs数据是格林尼治时间，偏移8小时到北京时间之后之后对准
#gfs分辨率为0.25°，经度从0开始向东经为正，维度从北纬90°开始向南为正。第一维是维度，第二维是经度（721*1440）。

from __future__ import division
import os
import cPickle, gzip
import numpy as np
import datetime 
from pyproj import Proj

p = Proj('+proj=merc')

today=datetime.datetime.today()
today=today.replace(2015,9,1)
savedir='/data/pm25data/dataset/'
loaddir='/data/pm25data/dataset/'

t_predict=120
#n_predict=8
def savefile(m,path):
    save_file = gzip.open(path, 'wb')  # this will overwrite current contents
    cPickle.dump(m, save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
    save_file.close()

f=gzip.open(loaddir+'DimPlusRNNTrueTest'+today.strftime('%Y%m%d')+'0903-0929.pkl.gz','rb')
inputs=cPickle.load(f)
print 'enhance heavy data'
repeat=10
morerows=0
moreindexlist=[]
for i in range(inputs.shape[0]):
    test = inputs[i,:,inputs.shape[2]-1]#寻找有用的典型例子
    if np.max(test)>80 and np.min(test)<80:
        morerows=morerows+repeat
        moreindexlist.append(i)
    more=np.zeros((morerows,inputs.shape[1],inputs.shape[2]))
for i in range(len(moreindexlist)):
    more[i*repeat:(i+1)*repeat,:,:]=inputs[moreindexlist[i],:,:]
    dataset=more

  

if __name__ == '__main__':
    print __file__
    print dataset.shape
    savefile(dataset,savedir+'FogyRNNTrueTest'+today.strftime('%Y%m%d')+'0903-0929.pkl.gz')
    print "saved at "+savedir+'FogyRNNTrueTest'+today.strftime('%Y%m%d')+'0903-0929.pkl.gz'
