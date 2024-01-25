import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def getInput(args,data):
    input_list = [data['input']]
    if args.in_light:
        input_list.append(data['l'])
    return input_list

def parseData(args,sample,timer=None,split='train'):  #把 data变成一个数据库
    input,target,mask = sample['img'] ,sample['N'],sample['mask']
    if timer:
        timer.updateTime('ToCPU')
    if args.cuda:
        input = input.cuda()
        target = target.cuda()
        mask = mask.cuda()
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)
    mask_var   = torch.autograd.Variable(mask, requires_grad=False)
    #mask_var = torch.autograd.Variable(mask)
    if timer:
        timer.updateTime('ToGPU')
        data = {'input':input_var,'tar':target_var,'m':mask_var}
    if args.in_light:
        light = sample['light'].expand_as(input)
        if args.cuda:
            light = light.cuda()
        light_var = torch.autograd.Variable(light)
        data['l'] = light_var
    return data

def getInputChanel(args):
    print('[Network Input] Color image as input')
    c_in =3
    if args.in_light:
        print('[Network Input] Adding Light direction as input')
        c_in +=3            #在rgb通道的基础上又加上了光照通道
    print('[Network Input] Input channel : {}'.format(c_in))
    return c_in

def get_n_params(model): #返回模型的参数量
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn=nn*s
        pp += nn
    return pp

def loadCheckpoint(path,model,cuda=True): #加载训练好的模型
    if cuda:
        checkpoint = torch.load(path)
    else:  #模型是GPU，预加载的训练参数却是CPU
        checkpoint = torch.load(path,map_location=lambda storage,loc:storage)
    model.load_state_dict(checkpoint['state_dict']) #参数对不上？

def saveCheckpoint(save_path,epoch=-1,model=None,optimizer=None,records=None,args=None):  #保存模型
    state = {'state_dict':model.state_dict(),'model': args.model}
    records = {'epoch': epoch, 'optimizer':optimizer.state_dict(), 'records': records,
            'args': args}
    torch.save(state, os.path.join(save_path, 'check_%d.pth.tar' % (epoch)),_use_new_zipfile_serialization=False) #使用旧版的模型板寸方法，不是zip
    torch.save(records, os.path.join(save_path, 'check_%d_rec.pth.tar' % (epoch)),_use_new_zipfile_serialization=False)

def conv(batchNorm,cin,cout,k=3,stride=1,pad=-1):
    pad= (k-1) // 2 if pad<0 else pad
    print('Conv pad = %d' % (pad))
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
            nn.Conv2d(cin,cout,kernel_size=k,stride=stride,padding=pad,bias=False),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        print('=> convolutional layer without bachnorm')
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
def deconv(cin,cout):
    return nn.Sequential(
        nn.ConvTranspose2d(cin,cout,kernel_size=4,stride=2,padding=1,bias=False),
        nn.LeakyReLU(0.1, inplace=True)
    )
def conv1_1(batchNorm, cin, cout, k=1, stride=1, pad=-1):
    pad = (k - 1) // 2 if pad < 0 else pad
    print('Conv pad = %d' % (pad))
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.1, inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
                )
def conv3_3(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = (k - 1) // 2 if pad < 0 else pad
    print('Conv pad = %d' % (pad))
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.1, inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
                )
class ResNet_basic_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = 3,  #3*3卷积
                              padding = 1,  #通过padding操作使x维度不变
                              bias = False)
        self.bn1 = nn.BatchNorm2d(num_features = out_channels)
        self.conv2 = nn.Conv2d(in_channels = out_channels,
                              out_channels = out_channels,
                              kernel_size = 3,   #3*3卷积
                              padding = 1,   #通过padding操作使x维度不变
                              bias = False)
        self.bn2 = nn.BatchNorm2d(num_features = out_channels)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(self.bn1(out), inplace = True)
        out = self.conv2(out)
        out = self.bn2(out)
        #print("#########out", out.shape)   #[1,64,32,32]
        #print("#########x", x.shape)        #[1,64,32,32]
        out_res=F.relu(self.bn1(out+x), inplace = True)
        #print("#########out_res", out_res.shape) #[1,64,32,32]
        return out_res




















