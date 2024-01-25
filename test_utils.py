import os
import torch
import torchvision.utils as vutils
import numpy as np
from models import model_utils
from utils import eval_utils,time_utils

def get_itervals(args,split): # 保存相应的法向图？
    args_var = vars(args)
    disp_intv =  args_var[split+'_disp']
    save_intv =  args_var[split+'_save']
    return disp_intv,save_intv

def test(args,split,loader,model,log,epoch,recorder):
    model.eval() #测试过程中要保证BN层的均值和方差不变,否则有数据输入就算不训练也会改变权值
    print('------  Start %s Epoch %d:%d batches  -----' % (split,epoch,len(loader)))
    timer = time_utils.Timer(args.time_sync)

    disp_intv, save_intv = get_itervals(args,split)
    with torch.no_grad(): #禁止使用梯度计算，加快计算速度
        for i , sample in enumerate(loader):
            data = model_utils.parseData(args,sample,timer,split)
            input = model_utils.getInput(args,data)

            out_var = model(input)
            
            timer.updateTime('Forward')
            # 新增1，获得误差图
            #print("###########out_var.data.shape:::",out_var.data.shape)   ####[1,3,512,612]
            acc, error_map = eval_utils.calNormalAcc(data['tar'].data, out_var.data, data['m'].data)
            #acc = eval_utils.calNormalAcc(data['tar'].data, out_var.data, data['m'].data)
            recorder.updateIter(split,acc.keys(),acc.values())

            iters = i+1
            if iters % disp_intv == 0:
                opt = {'split': split, 'epoch': epoch, 'iters': iters, 'batch': len(loader),
                       'timer': timer, 'recorder': recorder}
                log.printItersSummary(opt)

            if iters % save_intv == 0:
                pred =(out_var.data+1)/2
                masked_pred = pred * data['m'].data.expand_as(out_var.data)
                #log.saveNormalResults(masked_pred,split,epoch,iters)
                log.saveNormalResults(out_var.data,split,epoch,iters)
                # 罗开新增，我尝试保存把结果保存成npy文件
                #print('##############masked_pred_type',masked_pred.type)
                #print('##############masked_pred_shape',masked_pred.shape) [1,3,512,612]
                log.saveNpyMap(out_var.data,split,epoch,iters)
                # 新增2，保存误差图
                log.saveErrorMap(error_map, split, epoch, iters)

    opt = {'split':split,'epoch':epoch,'recorder':recorder}
    log.printEpochSummary(opt)