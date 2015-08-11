# -*- coding: utf-8 -*-
#这一版的pm25_maker直接从全球经纬度数据中取值，并且做插值
#并注意到全球gfs数据是格林尼治时间，偏移8小时到北京时间之后之后对准
#gfs分辨率为0.25°，经度从0开始向东经为正，维度从北纬90°开始向南为正。第一维是维度，第二维是经度（721*1440）。

#现在要预测48小时pm25，输入前144维是72小时内的每隔3小时gfs，之后24维pm25，之后48维未来pm25
#imputs包括之前一天24小时的pm25,6*8=48维的gfs，之后预测时间段内t_predict/3*6维的gfs数据；
from __future__ import division
import os
import cPickle, gzip
import numpy as np
import datetime 
from pyproj import Proj

p = Proj('+proj=merc')

gfsdir='/ldata/pm25data/gfs/'
pm25dir='/mnt/storm/nowcasting/pm25/'
today=datetime.datetime.today()
pm25meandir='/ldata/pm25data/pm25mean/mean'+today.strftime('%Y%m%d')+'/'
savedir='/ldata/pm25data/pm25dataset/'

t_predict = 48 #设置预测的时间

def interp(data,lat_x,lon_y):
    
    x1=int(lat_x)#整数部分
    dcm_lat_x=lat_x-x1#小数部分
    y1=int(lon_y)
    dcm_lon_y=lon_y-y1
    x2=x1+1
    y2=y1+1
    
    R1=(1-dcm_lat_x)*data[x1,y1]+dcm_lat_x*data[x2,y1]
    R2=(1-dcm_lat_x)*data[x1,y2]+dcm_lat_x*data[x2,y2]
    result=(1-dcm_lon_y)*R1+dcm_lon_y*R2
    return result

def lonlat2mercator(lon=[116.3883],lat=[39.3289]):
    #p = Proj('+proj=merc')
    radius=[17,72,54,135]
    r=10000
    
    cord_x=[]
    cord_y=[]
    y,x = np.round(np.array(p(radius[1],radius[0]))/r)
    y1,x1 = np.round(np.array(p(radius[3],radius[2]))/r)
    for i in range(len(lon)):
        longitude,latitude = p(lon[i],lat[i])
        latlng = np.array([latitude,longitude])
    
        if lat[i]<radius[0] or lat[i]>radius[2] or lon[i]<radius[1] or lon[i]>radius[3]:
            raise Exception,'Out of range  '+'lon:'+str(radius[1])+'-'+str(radius[3])+' lat:'+str(radius[0])+'-'+str(radius[2])
        latlng = np.abs(np.round(latlng/r)-np.array([x1,y]))
        cord_x.append(latlng[0])
        cord_y.append(latlng[1])
    return cord_x,cord_y
    
def savefile(m,path):
    save_file = gzip.open(path, 'wb')  # this will overwrite current contents
    cPickle.dump(m, save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
    save_file.close()

class Pm25Dataset(object):
    '''
    给出预测点的位置和数据集的时间范围，利用gfs和pm25原始
    数据文件，构建这一时间和地点内的pm25 dataset
    '''
    def __init__(self,lon=np.hstack((np.array([116.3883,117.20,121.48,106.54,118.78,113.66]),110+10*np.random.rand(94))),lat=np.hstack((np.array([39.3289,39.13,31.22,29.59,32.04,34.76]),32+10*np.random.rand(94))),start='2015040108',stop='2015051008',d_in=24+6*8+t_predict*2,d_out=t_predict):
        '''Initialize the parameters
        
        :lon:longitude of prediction points, scalar or vector like
        :lat:latitude of prediction points , scalar or vector like
        :start: Beijing time the dataset forecast point starts, e.g.'2015022800'
                better be 3 hours counted(08,11,14), since the gfs data
                are divided into 3 hours.
        :stop: Beijing time the dataset forecast point stops, e.g.'2015022818' 
               better be 3 hours counted(08,11,14), since the gfs data
               are divided into 3 hours.
        :d_in: dimention of input
        :d_out: dimention of output.
        '''
        self.cord_x,self.cord_y=lonlat2mercator(lon,lat)#中国地图mercator中坐标
        self.lat_x=(90.0-lat)/0.25#全球gfs图上坐标
        self.lon_y=lon/0.25
        
        self.d_in=d_in
        self.d_out=d_out
        self.starttime=datetime.datetime(int(start[0:4]),int(start[4:6]),int(start[6:8]),int(start[8:10]))
        self.stoptime=datetime.datetime(int(stop[0:4]),int(stop[4:6]),int(stop[6:8]),int(stop[8:10]))
        self.n_points=len(lon)#从地图上取了n_points个试验点
        self.n_perpoint=int((self.stoptime-self.starttime).days*8+(self.stoptime-self.starttime).seconds/10800+1)
        self.n_exp=self.n_points * self.n_perpoint#总样本数
        #n_exp按照每隔3小时一组training example来计算
        
        self.input_data=self.generateinput()#这里新版本input包括了之后的target t_predict维
        
    def generateinput(self):
        '''generate input vector'''
        inputs=np.zeros((self.n_exp,self.d_in+self.d_out))
        '''gfs,6*8+t_predict*2 dimentions'''
        for h in range(-24,t_predict,3):#在starttime前后两天的时间点,这个循环填上第一个example的前6*8+t_predict*2个数据
            p_time=self.starttime+datetime.timedelta(hours=h)-datetime.timedelta(hours=8)#做时区变换，转回GMT
            #p_time就代表这一帧的时间，尝试打开对应文件
            if p_time.hour==3 or p_time.hour==9 or p_time.hour==15 or p_time.hour==21:#去开003的数据
                name=p_time-datetime.timedelta(hours=3)
                filename=name.strftime('%Y%m%d')+'_'+name.strftime('%H')+'_003.pkl.gz'
            elif p_time.hour==6 or p_time.hour==12 or p_time.hour==18 or p_time.hour==00:#去开006的数据
                name=p_time-datetime.timedelta(hours=6)
                filename=name.strftime('%Y%m%d')+'_'+name.strftime('%H')+'_006.pkl.gz'
            else:
                raise Exception,'gfs filename error at time:'+p_time.strftime('%Y%m%d%H')
                
            if os.path.exists(gfsdir+filename) and os.path.getsize(gfsdir+filename)>0:#判断文件是否存在
                f = gzip.open(gfsdir+filename)
                print('current file:'+gfsdir+filename)
                cnt=0
                for entry in ['tmp','rh','ugrd','vgrd','prate','tcdc']:#填6个数据
                    temp=cPickle.load(f)
                    for k in range(self.n_points):
                        inputs[(0+k*self.n_perpoint,(h+24)*2+cnt)]=interp(temp.reshape((180*4+1,360*4)),self.lat_x[k],self.lon_y[k])
                        #(h+24)*2+cnt是用来找对应的格位置，（h+24）*2是对应小时的6格开始位置，再加上cnt作为偏置
                    cnt=cnt+1
                f.close()
            else:#用三小时之前的替换
                cnt=0
                print('no such file:'+gfsdir+filename)
                for entry in ['tmp','rh','ugrd','vgrd','prate','tcdc']:#填6个数据
                    for k in range(self.n_points):
                        inputs[(0+k*self.n_perpoint,(h+24)*2+cnt)]=inputs[(0+k*self.n_perpoint,(h+24)*2+cnt-6)]
                        #(h+24)*2+cnt是用来找对应的格位置，（h+24）*2是对应小时的6格开始位置，再加上cnt作为偏置
                cnt=cnt+1
        
   
        for i in range(1,self.n_perpoint):#填上矩阵中，剩余trainning example的数据
            current=self.starttime+datetime.timedelta(hours=3*i)-datetime.timedelta(hours=8)#做时区变换，转回GMT
            #对于当前current所指的这一行，前90维的数据可以从上一行获得
            for k in range(self.n_points):
                inputs[i+k*self.n_perpoint,0:6*8+t_predict*2-6]=inputs[i+k*self.n_perpoint-1,6:6*8+t_predict*2]
            #对于当前current所指的这一行，最后小时的六个数据需要以下重新读文件获得
            p_time=current+datetime.timedelta(hours=t_predict-3)
            #p_time就代表这一帧的时间，尝试打开对应文件
            if p_time.hour==3 or p_time.hour==9 or p_time.hour==15 or p_time.hour==21:#去开003的数据
                name=p_time-datetime.timedelta(hours=3)
                filename=name.strftime('%Y%m%d')+'_'+name.strftime('%H')+'_003.pkl.gz'
            elif p_time.hour==6 or p_time.hour==12 or p_time.hour==18 or p_time.hour==00:#去开006的数据
                name=p_time-datetime.timedelta(hours=6)
                filename=name.strftime('%Y%m%d')+'_'+name.strftime('%H')+'_006.pkl.gz'
            else:
                raise Exception,'gfs filename error at time:'+p_time.strftime('%Y%m%d%H')
                
            if os.path.exists(gfsdir+filename) and os.path.getsize(gfsdir+filename)>0:#判断文件是否存在
                f = gzip.open(gfsdir+filename)
                print('current file:'+gfsdir+filename)
                cnt=0
                for entry in ['tmp','rh','ugrd','vgrd','prate','tcdc']:#填6个数据
                    temp=cPickle.load(f)
                    for k in range(self.n_points):
                        inputs[(i+k*self.n_perpoint,6*8+t_predict*2-6+cnt)]=interp(temp.reshape((180*4+1,360*4)),self.lat_x[k],self.lon_y[k])
                        #(h+24)*2+cnt是用来找对应的格位置，（h+24）*2是对应小时的6格开始位置，再加上cnt作为偏置
                    cnt=cnt+1
                f.close()
            else:#用三小时之前的替换
                cnt=0
                print('no such file:'+gfsdir+filename)
                for entry in ['tmp','rh','ugrd','vgrd','prate','tcdc']:#填6个数据
                    for k in range(self.n_points):
                        inputs[(i+k*self.n_perpoint,6*8+t_predict*2-6+cnt)]=inputs[(i+k*self.n_perpoint,6*8+t_predict*2-12+cnt)]
                        #(h+24)*2+cnt是用来找对应的格位置，（h+24）*2是对应小时的6格开始位置，再加上cnt作为偏置
                cnt=cnt+1
                
        '''pm25,24 dimentions'''
        pm25mean=[None]*24
        for h in range(24):#取出各个小时的pm25mean备用
            f = open(pm25meandir+'meanfor'+str(h)+'.pkl', 'rb')
            pm25mean[h]=cPickle.load(f)
            f.close()
        #生成第一行
        for h in range(24+t_predict):#在starttime前后两天的时间点,这个循环填上第一个example的后24+t_predict个数据
            name=(self.starttime+datetime.timedelta(hours=h-23)).strftime('%Y%m%d%H')
            cnt=0
            if int(name) > 2015061324:
                if os.path.exists(pm25dir+name[0:8]+'/'+name+'.pkl.gz') and os.path.getsize(pm25dir+name[0:8]+'/'+name+'.pkl.gz')>0:#判断文件是否存在
                    f = gzip.open(pm25dir+name[0:8]+'/'+name+'.pkl.gz', 'rb')
                    temp=cPickle.load(f)
                    f.close()
                    for k in range(self.n_points):
                        inputs[(0+k*self.n_perpoint,6*8+t_predict*2+h)]=temp[(self.cord_x[k],self.cord_y[k])]-pm25mean[int(name[8:10])][self.cord_x[k],self.cord_y[k]]
                    #(h+24)*2+cnt是用来找对应的格位置，（h+24）*2是对应小时的6格开始位置，再加上cnt作为偏置
                else:#用1小时之前的替换
                    for k in range(self.n_points):
                        inputs[(0+k*self.n_perpoint,6*8+t_predict*2+h)]=inputs[(0+k*self.n_perpoint,6*8+t_predict*2+h-1)]                
                cnt=cnt+1
            else:
                if os.path.exists(pm25dir+name+'.pkl.gz') and os.path.getsize(pm25dir+name+'.pkl.gz')>0:#判断文件是否存在
                    f = gzip.open(pm25dir+name+'.pkl.gz', 'rb')
                    temp=cPickle.load(f)
                    f.close()
                    for k in range(self.n_points):
                        inputs[(0+k*self.n_perpoint,6*8+t_predict*2+h)]=temp[(self.cord_x[k],self.cord_y[k])]-pm25mean[int(name[8:10])][self.cord_x[k],self.cord_y[k]]
                    #(h+24)*2+cnt是用来找对应的格位置，（h+24）*2是对应小时的6格开始位置，再加上cnt作为偏置
                else:#用1小时之前的替换
                    for k in range(self.n_points):
                        inputs[(0+k*self.n_perpoint,6*8+t_predict*2+h)]=inputs[(0+k*self.n_perpoint,6*8+t_predict*2+h-1)]                
                cnt=cnt+1
            
        for i in range(1,self.n_perpoint):#for 全部行,生成后24+t_predict维数据
            current=self.starttime+datetime.timedelta(hours=3*i)
            #对于当前current所指的这一行，前24+t_predict-3维的数据可以从上一行获得
            for k in range(self.n_points):
                inputs[i+k*self.n_perpoint,6*8+t_predict*2:(6*8+t_predict*2)+(24+t_predict)-3]=inputs[i+k*self.n_perpoint-1,(6*8+t_predict*2)+3:(6*8+t_predict*2)+(24+t_predict)]
            for h in range(3):#最后新的3位数据要读文件
                #每一行从左向右的时间顺序,这里直接生成48维数据，前24维作为input（包括当前时刻）
                #后24维作为output,从当前时刻之后的一小时开始
                #name=(current-datetime.timedelta(hours=h)).strftime('%Y%m%d%H')
                name=(current+datetime.timedelta(hours=h+t_predict-2)).strftime('%Y%m%d%H')#未来t_predict-2，t_predict-1，t_predict小时
                if int(name) > 2015061324:
                    if os.path.exists(pm25dir+name[0:8]+'/'+name+'.pkl.gz') and os.path.getsize(pm25dir+name[0:8]+'/'+name+'.pkl.gz')>0:#判断文件是否存在
                        f = gzip.open(pm25dir+name[0:8]+'/'+name+'.pkl.gz', 'rb')
                        print(name+'.pkl.gz')
                        temp=cPickle.load(f)
                        f.close()
                        for k in range(self.n_points):
                            inputs[i+k*self.n_perpoint,(6*8+t_predict*2)+(24+t_predict)-3+h]=temp[self.cord_x[k],self.cord_y[k]]-pm25mean[int(name[8:10])][self.cord_x[k],self.cord_y[k]]
                    else:#用右边后一个小时的数据填充（暂定可能的补偿方法）
                        for k in range(self.n_points):
                            inputs[i+k*self.n_perpoint,(6*8+t_predict*2)+(24+t_predict)-3+h]=inputs[i+k*self.n_perpoint,(6*8+t_predict*2)+(24+t_predict)-3+h-1]
                else:
                    if os.path.exists(pm25dir+name+'.pkl.gz') and os.path.getsize(pm25dir+name+'.pkl.gz')>0:#判断文件是否存在
                        f = gzip.open(pm25dir+name+'.pkl.gz', 'rb')
                        print(name+'.pkl.gz')
                        temp=cPickle.load(f)
                        f.close()
                        for k in range(self.n_points):
                            inputs[i+k*self.n_perpoint,(6*8+t_predict*2)+(24+t_predict)-3+h]=temp[self.cord_x[k],self.cord_y[k]]-pm25mean[int(name[8:10])][self.cord_x[k],self.cord_y[k]]
                    else:#用右边后一个小时的数据填充（暂定可能的补偿方法）
                        for k in range(self.n_points):
                            inputs[i+k*self.n_perpoint,(6*8+t_predict*2)+(24+t_predict)-3+h]=inputs[i+k*self.n_perpoint,(6*8+t_predict*2)+(24+t_predict)-3+h-1]

        
        '''send out'''
        return inputs
                                 
    def save():
        '''save files'''
        
if __name__ == '__main__':
    #a,b=lonlat2mercator()
    start=(today-datetime.timedelta(days=103)).strftime('%Y%m%d')+'08'
    stop=(today-datetime.timedelta(days=3)).strftime('%Y%m%d')+'08'
    obj=Pm25Dataset(start=start,stop=stop)
    #obj=Pm25Dataset(lon=np.array([116.3883,117.20,121.48,106.54,118.78,113.66]),lat=np.array([39.3289,39.13,31.22,29.59,32.04,34.76]))
    savefile(obj.input_data,savedir+'Pm25Dataset'+today.strftime('%Y%m%d')+'_t45p100.pkl.gz')
    np.savetxt(savedir+"Pm25Dataset"+today.strftime('%Y%m%d')+"_t45p100.txt", obj.input_data, fmt='%.2f')
    np.random.shuffle(obj.input_data)
    savefile(obj.input_data,savedir+'Pm25Dataset'+today.strftime('%Y%m%d')+'_t45p100shuffled.pkl.gz')
