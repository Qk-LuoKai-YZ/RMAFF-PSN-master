import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_   #少写一个  _
from . import model_utils
from .RMFE import RF2B
from collections import OrderedDict

class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3,
                                           stride=1, padding=1, bias=False))

    # ����forward����
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i+1),
                            _DenseLayer(in_channels+growth_rate*i,
                                        growth_rate, bn_size))


class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1,
                                          stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=1, stride=1))


class DenseNet_BC(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6,12,24,16),
                 bn_size=4, theta=0.5, num_classes=10):
        super(DenseNet_BC, self).__init__()

        # ��ʼ�ľ��Ϊfilter:2����growth_rate
        num_init_feature = 2 * growth_rate

        # ��ʾcifar-10
        if num_classes == 10:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_feature,
                                    kernel_size=7, stride=1,
                                    padding=3, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_feature,
                                    kernel_size=3, stride=1,
                                    padding=1, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_feature)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ]))



        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock%d' % (i+1),
                                     _DenseBlock(num_layers, num_feature,
                                                 bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config)-1:
                self.features.add_module('transition%d' % (i + 1),
                                         _Transition(num_feature,
                                                     int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        # self.features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))

        # self.classifier = nn.Linear(num_feature, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # features = self.features(x)
        # out = features.view(features.size(0), -1)
        # out = self.classifier(out)
        out = self.features(x)
        return out


# DenseNet_BC for ImageNet
def DenseNet121():
    #return DenseNet_BC(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=1000)
    return DenseNet_BC(growth_rate=32, block_config=(1, 2, 4, 3), num_classes=1000)
def DenseNet169():
    return DenseNet_BC(growth_rate=32, block_config=(6, 12, 32, 32), num_classes=1000)

def DenseNet201():
    return DenseNet_BC(growth_rate=32, block_config=(6, 12, 48, 32), num_classes=1000)

def DenseNet161():
    return DenseNet_BC(growth_rate=48, block_config=(6, 12, 36, 24), num_classes=1000,)

# DenseNet_BC for cifar
def densenet_BC_100():
    return DenseNet_BC(growth_rate=12, block_config=(16, 16, 16))



class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64, k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64, 128, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 128, 256, k=3, stride=2, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 256, 256, k=3, stride=1, pad=1)

        self.deconv11 = model_utils.deconv(256, 128)
        self.RME1 = RF2B(128, 64)

        self.conv6 = model_utils.deconv(256, 128)
        self.conv7 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

        self.RME2 = RF2B(128, 64)

    def forward(self, x):
        # print("##########F_x::",x.shape)   [1,6,512,612]
        out = self.conv1(x)
        # print("##########F_c1::", out.shape)  [1,64,512,612]
        out = self.conv2(out)
        # print("##########F_c2::", out.shape)  [1,128,256,306]
        out = self.conv3(out)
        # print("##########F_c3::", out.shape)   [1,128,256,306]
        out = self.conv4(out)
        # print("##########F_c4::", out.shape)  [1,256,128,153]
        out = self.conv5(out)
        # print("##########F_c5::", out.shape)  [1,256,128,153]
        out51 = self.deconv11(out)
        #print("##########F_51::", out51.shape)
        out52 = self.RME1(out51)
        #print("##########F_52::", out52.shape)
        out = self.conv6(out)
        # print("##########F_c6::", out.shape)   [1,128,256,306]
        out = self.conv7(out)
        # print("##########F_c7::", out.shape)   [1,128,256,306]
        out71 = self.RME2(out)
        #print("##########F_71::", out71.shape)
        out_feat = torch.cat((out52, out71), 1)
        #print("##########F_cat::", out_feat.shape)
        n, c, h, w = out_feat.data.shape
        out_feat = out_feat.view(-1)
        # print("##########F_outfeat::", out_feat.shape)  [10027008]
        # print("##########F_nchw::", [n,c,h,w])   [1,128,256,306]
        return out_feat, [n, c, h, w]


class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}):
        super(Regressor, self).__init__()
        self.other   = other

        self.deconv00 = model_utils.deconv(128, 64)
        self.deconv01 = model_utils.conv(batchNorm, 64, 128, k=3, stride=1, pad=1)

        self.deconv1 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv2 = model_utils.deconv(128, 64)
        self.deconv3= self._make_output(64, 3, k=3, stride=1, pad=1)
        self.deconv4=DenseNet121()          #你知道你没有把densenet写入这个文件吗？！
        self.deconv5 = model_utils.deconv(188, 64)
        self.deconv6 = model_utils.conv(batchNorm, 64,64,  k=3, stride=2, pad=1)
        self.est_normal=self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other   = other


    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
               nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        x      = x.view(shape[0], shape[1], shape[2], shape[3])
        #print("##########R_x::", x.shape)                            #[1,128,256,306]
        out    = self.deconv00(x)   #[1,64,512,612]
        #print("##########R_d00::", out.shape)
        out    = self.deconv01(out) #[1,128,512,612]
        #print("##########R_d01::", out.shape)
        out    = self.deconv1(out)  #[1,128,512,612]
        #print("##########R_d1::", out.shape)
        out    = self.deconv2(out)  #[1,64,1024,1224]
        #print("##########R_d2::", out.shape)
        out    = self.deconv3(out)  #[1,64,1024,1224]
        #print("##########R_d3::", out.shape)
        out    = self.deconv4(out)  #[1,188,512,612]
        #print("##########R_d4::", out.shape)
        out    = self.deconv5(out)  #[1,64,512,612]
        #print("##########R_d5::", out.shape)
        out    = self.deconv6(out)  #[1,64,512,612]
        #print("##########R_d6::", out.shape)
        normal = self.est_normal(out)  #[1,3,512,612]
        #print("##########R_estNormal::", normal.shape)
        normal = torch.nn.functional.normalize(normal, 2, 1)  #[1,3,512,612]
        #print("##########R_normal::", normal.shape)
        return normal

class PS_FCN(nn.Module):
    def __init__(self,fuse_type='max',batchNorm=False,c_in=3,other={}):
        super(PS_FCN,self).__init__()
        self.extractor = FeatExtractor(batchNorm,c_in,other)
        self.regressor = Regressor(batchNorm,other)
        self.c_in = c_in
        self.fuse_type =fuse_type
        self.other = other
        for m in self.modules():  #不懂是干什么的
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        img = x[0]
        img_split = torch.split(img,3,1)  # 3 一个张量是多大 相同大小，如果是[1,4]就是一个1，一个4，分了两个张量 ||   1 划分张量依据的维度
        if len(x) >1: #img+light >1 == have lighting
            light = x[1]
            light_split = torch.split(light,3,1) #同理
        feats = []
        for i in range(len(img_split)):
            net_in = img_split[i] if len(x) ==1 else torch.cat([img_split[i],light_split[i]],1) #不是的话拼在一起作为输入
            feat,shape = self.extractor(net_in)
            feats.append(feat)
        if self.fuse_type =='mean':
            feat_fused = torch.stack(feats,1).mean(1)#把多个2维的张量凑成一个3维的张量；多个3维的凑成一个4维的张量…以此类推，也就是在增加新的维度进行堆叠。
        elif self.fuse_type =='max':
            feat_fused,_ = torch.stack(feats, 1).max(1) #这么多特征都变成了 1维的了 接着按照第1维进行取最大特征
        normal = self.regressor(feat_fused,shape)
        return normal





























