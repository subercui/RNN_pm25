# -*- coding: utf-8 -*-
#几个bash命令，以及调用的.py文件内的bash命令，访问顺序是什么样的
#用于每日调用，更新model以及依赖文件

import sys
import os
from datetime import datetime

today=datetime.today()
print "begin at "+today.strftime('%Y-%m-%d %H:%M')

#scp dataset
if not os.path.exists('/data/pm25data/dataset/RNNPm25Dataset'+today.strftime('%Y%m%d')+'_t100p100shuffled.pkl.gz'):
    os.system('scp suber@swarma.net:/ldata/pm25data/pm25dataset/RNNPm25Dataset'+today.strftime('%Y%m%d')+'_t100p100shuffled.pkl.gz /data/pm25data/dataset/')
if not os.path.exists('/data/pm25data/dataset/RNNPm25Dataset'+today.strftime('%Y%m%d')+'_t100p100shuffled.pkl.gz'):
    os.system('echo "Pm25 RNN Dataset file generating error!" | mail -s "caiyun pm25 alarm from lighting" "subercui@sina.com"')
    sys.exit(-1)
print "dataset downloaded"

#update model
os.system('python /home/suber/Git/RNN_pm25/model/Pm25RNN_MINIBATCH.py')
print "RNN model trained"
if not os.path.exists('/data/pm25data/model/RNNModel'+today.strftime('%Y%m%d')+'.pkl.gz'):
    os.system('echo "Pm25 RNN model generating error!" | mail -s "caiyun pm25 alarm from lighting" "subercui@sina.com"')
    sys.exit(-1)

#scp
os.system('scp /data/pm25data/model/RNNModel'+today.strftime('%Y%m%d')+'.pkl.gz caiyun@api.dev2.caiyunapp.com:/ldata/pm25data/pm25model/')
