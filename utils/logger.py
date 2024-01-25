import datetime ,time   #用于获取当前时间和日期的
import os  #操作系统的功能，比如说文件目录 当前目录 增删改查文件夹
import numpy as np
import torch
import torchvision.utils as vutils
from scipy.io import savemat
from . import utils
from utils import eval_utils, time_utils


class Logger(object):
    def __init__(self,args):   #list存放参数到   args
        self.times={'init': time.time()}
        self._checkPath(args) #检查路径？
        self.args = args
        self.printArgs()

    def printArgs(self):  #输出参数
        strs = '------------------Options-------------\n'
        strs += '{}'.format(utils.dictToString(vars(self.args))) #format设置格式，调用把 vars()转换出来的字典格式转到string
        strs +='---------------- End------------------\n'
        print(strs)

    def _checkPath(self,args):
        if hasattr(args,'run_model') and args.run_model: #args是否有 run_model属性
            #dirname()从指定路径获取目录名称   join() 和'run_model‘连接后生成新的字符串
            log_root = os.path.join(os.path.dirname(args.retrain), 'run_model')
            utils.makeFiles([os.path.join(log_root,'test')])
        else:
            #.isfile 需要的是绝对路径，一般都用 .join 进行路径拼接 , 判断某一对象是否是文件
            if args.resume and os.path.isfile(args.resume):
                log_root = os.path.join(os.path.dirname(os.path.dirname(args.resume)),'resume')
            else:
                log_root = os.path.join(args.save_root,args.item)
            for sub_dir in ['train','val']:  #子目录
                utils.makeFiles([os.path.join(log_root,sub_dir)])
            args.cp_dir = os.path.join(log_root,'train') #这个是什么路径
        args.log_dir = log_root

    def getTimeInfo(self,epoch,iters,batch):
        time_elapsed = (time.time() - self.times['init']) / 3600.0 #目前的时间-开始的时间 = 多少小时
        total_iters = (self.args.epochs - self.args.start_epoch +1) *batch
        cur_iters =(epoch - self.args.start_epoch)*batch + iters
        time_total = time_elapsed*(float(total_iters)/cur_iters) #训练了这多么 tiers用了elapsed时间，那么总的iters大概需要的时间
        return time_elapsed,time_total

    def printItersSummary(self ,opt):
        epoch,iters,batch = opt['epoch'],opt['iters'],opt['batch']
        strs = ' |{}'.format(str.upper(opt['split']))  #,upper大写，
        strs += ' Iter [{}/{}] Epoch [{}/{}]' .format(iters,batch,epoch, self.args.epochs)
        if opt['split'] == 'train':
            time_elapsed , time_total = self.getTimeInfo(epoch,iters,batch)
            strs += ' Clock [{:.2f}h/{:.2f}h]'.format(time_elapsed,time_total)
            strs += ' LR [{}]'.format(opt['recorder'].records[opt['split']]['lr'][epoch][0])#没懂 还没写
            print(strs)
        if 'timer' in opt.keys():  #又是干什么的
            print(opt['timer'].timeToString())   # 还有个 time.utils.py 没写 在那里面
        if 'recorder' in opt.keys():
            print(opt['recorder'].iterRecToString(opt['split'],epoch))  #这个在 别的地方还没写

    def printEpochSummary(self,opt): #输出训练过程的
        split = opt['split']
        epoch = opt['epoch']
        print('--------------  {} Epoch {} Summary  ----------'.format(str.upper(split),epoch))
        print(opt['recorder'].epochRecToString(split,epoch))

    def saveNormalResults(self,results,split,epoch,iters):   #保存结果图片的
        save_dir = os.path.join(self.args.log_dir,split)
        save_name = '%d_%d.png' % (epoch,iters)
        vutils.save_image(results,os.path.join(save_dir,save_name))

    def saveErrorMap(self,error_map,split,epoch,iters):  #是瞎吗这么久才看出来。
        save_dir = os.path.join(self.args.log_dir, split)
        save_name = 'errormap%d_%d.png' % (epoch, iters)
        vutils.save_image(error_map, os.path.join(save_dir, save_name))
        
    def saveNpyMap(self,results,split,epoch,iters):   #保存结果图片的  我自己写的
        save_dir = os.path.join(self.args.log_dir,split)
        save_name = '%d_%d.npy' % (epoch,iters)
        ########results = results.data.cpu().numpy()
        #print("#####results.shape:",results.shape)
        #########results_trans=results.permute(2,3,1,0)
        results_trans=results.squeeze()
        results_trans=results_trans.permute(1,2,0)
        print("#####results_trans.shape:",results_trans.shape)  # [512,612,3]
        results_trans=results_trans.data.cpu().numpy()
        np.save(os.path.join(save_dir,save_name),results_trans)
        print("save .npy done")
    def saveMatMap(self,results,split,epoch,iters):   #保存结果图片的  我自己写的
        save_dir = os.path.join(self.args.log_dir,split)
        save_name = '%d_%d.mat' % (epoch,iters)
        results = results.cpu().detach().numpy()
        results = np.array(results)
        savemat(os.path.join(save_dir,save_name),{'normal':results})
        print("save .mat done")