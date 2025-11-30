import sys
import os
import pandas as pd
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import OrderedDict
from .common import dict_round

class Logger():
    def __init__(self,save_name):
        self.log = None
        self.summary = None
        self.name = save_name
        self.time_now = time.strftime('_%Y-%m-%d-%H-%M', time.localtime())

    def update(self,epoch,train_log,val_log):
        item = OrderedDict({'epoch':epoch})
        item.update(train_log)
        item.update(val_log)
        item = dict_round(item, 6)
        print(item)
        self.update_csv(item)
        self.update_tensorboard(item)

    def update_csv(self,item):
        tmp = pd.DataFrame(item,index=[0])
        if self.log is not None:
            self.log = pd.concat([self.log, tmp], ignore_index=True)
        else:
            self.log = tmp
        self.log.to_csv('%s/log%s.csv' %(self.name,self.time_now), index=False)

    def update_tensorboard(self,item):
        if self.summary is None:
            self.summary = SummaryWriter('%s/' % self.name)
        epoch = item['epoch']
        for key,value in item.items():
            if key != 'epoch': self.summary.add_scalar(key, value, epoch)
    def save_graph(self,model,input):
        if self.summary is None:
            self.summary = SummaryWriter('%s/' % self.name)
        self.summary.add_graph(model, (input,))
        print("Architecture of Model have saved in Tensorboard!")

class Print_Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        try:
            self.terminal.write(message)
            self.log.write(message)
        except UnicodeEncodeError:
            safe_message = message.encode('ascii', 'ignore').decode('ascii')
            self.terminal.write(safe_message)
            self.log.write(message)
        except Exception:
            safe_message = message.encode('ascii', 'ignore').decode('ascii')
            self.terminal.write(safe_message)
            self.log.write(safe_message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        if hasattr(self, 'log') and not self.log.closed:
            self.log.close()

