from __future__ import division
import os
import numpy as np
from imageio import imread
import scipy.io as sio  #读取数据的

import torch
import torch.utils.data as data

from datasets import pms_transforms
from . import util
np.random.seed(0)

from PIL import Image

class DiLiGenT_main(data.Dataset):
    def __init__(self,args,split='train'):
        self.root   = os.path.join(args.bm_dir)  # bm_dir是什么
        self.split = split
        self.args  = args
        self.objs  = util.readList(os.path.join(self.root,'objects.txt'),sort= False)
        self.names = util.readList(os.path.join(self.root,'filenames.txt'),sort=False)
        self.l_dir = util.light_source_directions()
        print('[%s Data] \t%d objs %d lights. Root: %s' %
              (split,len(self.objs),len(self.names),self.root))
        self.intens = {}
        intens_name = 'light_intensities.txt'
        print('Files for intensity: %s' %(intens_name))
        for obj in self.objs:
            self.intens[obj] = np.genfromtxt(os.path.join(self.root,obj,intens_name))

    def _getMask(self,obj):
        mask = imread(os.path.join(self.root,obj,'mask.png'))
        if mask.ndim >2:
            mask=mask[:,:,0]
        mask = mask.reshape(mask.shape[0],mask.shape[1],1)
        return mask /255.0

    def __getitem__(self, index):  #返回第 index 样本的数据
        np.random.seed(index)
        obj = self.objs[index]

        #print('##############len_names', len(self.names))  #6
        #print('##############in_img_num', self.args.in_img_num) # 6
        select_idx = np.random.permutation(len(self.names))[:self.args.in_img_num]
        #print('##############select_idx',select_idx)   #[5 2 1 3 0 4]
        img_list = [os.path.join(self.root,obj,self.names[i]) for i in select_idx]
        #print('##############img_list', img_list)
        intens   = [np.diag(1 / self.intens[obj][i]) for i in select_idx]
        #print('##############intens', intens)

        normal_path = os.path.join(self.root,obj,'Normal_gt.mat')
        print('##############normal_path', normal_path)  # 100
        normal = sio.loadmat(normal_path)  #读取 .mat 文件夹内容
        normal = normal['Normal_gt']

        imgs =[]
        for idx,img_name in enumerate(img_list):
            img = imread(img_name).astype(np.float32) /255.0   #图像数据归一化 ，感觉不像是一个人写的代码这个文件

            #print('##############img', img.shape)  [512,612,4]
            #print('##############intens[idx]', intens[idx].shape)  [3,3]
            #print('##############img_name', img_name)  [/fcPNG/006.png]
            #print('##############idx', idx)    0 1 2 3 4 5
            if img.shape[2] == 4:   #如果图像是4通道的话，然鹅diligent是3通道，当然也可以直接修改原图为3通道的
                img = Image.open(img_name).convert("RGB")
            #print('##############img', img)
            #print('##############intens[idx]',intens[idx].shape)

            img = np.dot(img,intens[idx])
            imgs.append(img)
        if self.args.normalize:
            imgs = pms_transforms.normalize(imgs)
        img = np.concatenate(imgs,2)  #数组进行拼接
        if self.args.normalize:
            img = img * np.sqrt(len(imgs) / self.args.train_img_num)

        mask = self._getMask(obj)
        down = 4
        if mask.shape[0] % down != 0 or mask.shape[1] %down != 0  : # h w 不能被4除尽
            pad_h = down -mask.shape[0] %down  #填充
            pad_w = down -mask.shape[1] %down
            img = np.pad(img,((0,pad_h),(0,pad_w),(0,0)),'constant',constant_values=((0,0),(0,0),(0,0)))
            # **********************************新增 本来没有下面这一行
            mask = np.pad(mask, ((0,pad_h), (0,pad_w), (0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            normal = np.pad(normal,((0,pad_h),(0,pad_w),(0,0)),'constant',constant_values=((0,0),(0,0),(0,0)))
        img = img*mask.repeat(img.shape[2],2)
        item = {'N':normal,'img':img,'mask':mask}

        for k in item.keys():
            item[k] = pms_transforms.arrayToTensor(item[k])

        if self.args.in_light:
            item['light']=torch.from_numpy(self.l_dir[select_idx]).view(-1,1,1).float()
        item['obj']=obj
        return item

    def __len__(self):
        return len(self.objs)




































