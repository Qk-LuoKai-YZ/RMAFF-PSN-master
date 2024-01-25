import torch
import torch.nn as nn
import os

class Criterion(object):
    def __init__(self,args):
        self.setupNormalCrit(args)

    def setupNormalCrit(self,args):
        print('=> using {} for criterion normal' .format(args.normal_loss))
        self.normal_loss = args.normal_loss
        self.normal_w    = args.normal_w
        if args.normal_loss == 'mse':
            self.n_crit = torch.nn.MSELoss()
        elif args.normal_loss =='cos':
            self.n_crit = torch.nn.CosineEmbeddingLoss()
        else:
            raise  Exception("=> unknow criterion '{}'".format(args.normal_loss))
        if args.cuda:
            self.n_crit = self.n_crit.cuda()

    def forward(self,output,target):
        if self.normal_loss == 'cos':
            num = target.nelement() // target.shape[1]
            if not hasattr(self,'flag') or num != self.flag.nelement():  #判断对象是否有flag属性
                self.flag = torch.autograd.Variable(target.data.new().resize_(num).fill_(1)) #当我们进行了一系列计算，并想获取一些变量间的梯度信息
            self.out_reshape = output.permute(0,2,3,1).contiguous().view(-1,3)
            self.gt_reshape  = target.permute(0,2,3,1).contiguous().view(-1,3)
            self.loss = self.n_crit(self.out_reshape,self.gt_reshape,self.flag)
        elif self.normal_loss == 'mse':
            self.loss = self.normal_w * self.n_crit(output,target)
        out_loss = {'N_loss': self.loss.item()}
        return out_loss

    def backward(self):
        self.loss.backward()

def getOptimizer(args,params):
    print('=> using %s solver for optimization ' % (args.solver)) # solver是优化器
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(params,args.init_lr,betas=(args.beta_1,args.beta_2)) #学习率 lr 权重衰减
    elif args.solver =='sgd':
        optimizer = torch.optim.SGD(params,args.init_lr,momentum=args.momentum)
    else:
        raise Exception('=> unknow optimizer %s' % (args.solver))
    return optimizer

#按设定的间隔调整学习率
#milestones(list)- 一个list，每一个元素代表何时调整学习率，list元素必须是递增的。如 milestones=[30,80,120]
#gamma(float)- 学习率调整倍数，默认为0.1倍，即下降10倍。
#last_epoch(int)- 上一个epoch数，这个变量用来指示学习率是否需要调整。当last_epoch符合设定的间隔时，就会对学习率进行调整。当为-1时，学习率设置为初始值。

def getLrScheduler(args,optimizer):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.milestones,gamma=args.lr_decay, last_epoch=args.start_epoch-2)
    return scheduler

def loadRecords(path,model,optimizer): #是干什么的
    records = None
    if os.path.isfile(path):
        records = torch.load(path[:-8] + '_rec' +path[-8:])
        optimizer.load_state_dict(records['optimizer'])
        start_epoch = records['epoch'] +1
        records = records['records']
        print("=> loaded records")
    else:
        raise Exception("=> no checkpoint found at '{}'" .format(path))
    return records,start_epoch


def configOptimizer(args,model):
    records = None
    optimizer = getOptimizer(args,model.parameters())   #中断去写  getoptimizer
    if args.resume:
        print("=> resume loading checkpoint '{}'" .format(args.resume))
        records,start_epoch = loadRecords(args.resume,model,optimizer)
        args.start_epoch = start_epoch
    scheduler = getLrScheduler(args,optimizer)
    return optimizer,scheduler,records
