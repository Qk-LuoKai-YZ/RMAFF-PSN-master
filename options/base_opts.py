import argparse #直接在命令行中就可以向程序中传入参数并让程序运行

class Base0pts(object):    #object 继承自父类，包含很多类的高级特性   你这里 0 应该写成 字母O 的写错了但是不影响
    def __init__(self):#默认构造方法
        #创建解析器对象 ArgumentParser , formatter_calss 自定义帮助信息的格式
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    def initialize(self):
        #设备设置

        # --代表可选参数， 默认是真，如果有这个参数了，那么就变成FALSE了
        self.parser.add_argument('--cuda',default=True , action='store_false')
        self.parser.add_argument('--time_sync',default=False,action='store_true')

        self.parser.add_argument('--workers',default=8,type=int)
        #self.parser.add_argument('--workers',default=0,type=int)
        self.parser.add_argument('--seed',default=0,type=int)

        #模型设置
        self.parser.add_argument('--fuse_type',default='max')
        self.parser.add_argument('--normalize',default=False,action='store_true')
        self.parser.add_argument('--in_light',default=True,action='store_false')#输入是否有光照信息
        self.parser.add_argument('--use_BN',default=False,action='store_true')
        self.parser.add_argument('--train_img_num',default=32,type=int) # for data normalization不知道什么意思
        self.parser.add_argument('--in_img_num',default=32,type=int) #应该数输入数量
        self.parser.add_argument('--start_epoch',default=1,type=int) #开始的轮数
        self.parser.add_argument('--epochs',default=1,type=int)#默认的轮数
        self.parser.add_argument('--resume' ,default=None)
        self.parser.add_argument('--retrain',default=None)

        #日志参数，保存的路径
        self.parser.add_argument('--save_root',default='data/Training/')
        self.parser.add_argument('--item',default='calib') #上一个路径下的项目路径，保存了模型，还有局部法向图

    def parse(self):
        self.args = self.parser.parse_args()
        return self.args

