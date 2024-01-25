from .base_opts import Base0pts   #从相同目录下的该文件中导入函数
class Train0pts(Base0pts):   #继承自Base0opts
    def __init__(self):
        super(Train0pts,self).__init__()  # 子类继承父类的属性和方法
        self.initialize()

    def initialize(self):  #重写父类的方法
        Base0pts.initialize(self)

        #数据设置
        self.parser.add_argument('--dataset',default='PS_Synth_Dataset')
        self.parser.add_argument('--data_dir',default='../../data/datasets/PS_Blobby_Dataset')
        self.parser.add_argument('--data_dir2', default='../../data/datasets/PS_Sculpture_Dataset')
        self.parser.add_argument('--concat_data',default=False,action='store_true')
        self.parser.add_argument('--rescale',default=True,action='store_false')
        self.parser.add_argument('--crop',default=True,action='store_false')
        self.parser.add_argument('--crop_h',default=32,type=int)
        self.parser.add_argument('--crop_w',default=32,type=int)
        self.parser.add_argument('--noise_aug',default=True,action='store_false')
        self.parser.add_argument('--noise',default=0.05,type=float)
        self.parser.add_argument('--color_aug',default=True,action='store_false')
        #训练设置
        self.parser.add_argument('--model',default='RMAFF_PSN')
        self.parser.add_argument('--solver',default='adam',help='adam|sgd')
        self.parser.add_argument('--milestones',default=[5,10,15,20,25],nargs='+',type=int)#nargs允许多个参数
        self.parser.add_argument('--init_lr',default=1e-3,type=float)
        self.parser.add_argument('--lr_decay', default=0.5, type=float)
        self.parser.add_argument('--beta_1',default=0.9,type=float,help='adam')
        self.parser.add_argument('--beta_2',default=0.999,type=float,help='adam')
        self.parser.add_argument('--momentum',default=0.9,type=float,help='sgd')
        self.parser.add_argument('--batch',default=32,type=int)
        self.parser.add_argument('--val_batch',default=8,type=int)
        #显示设置
        self.parser.add_argument('--train_disp',default=20,type=int)
        self.parser.add_argument('--train_save',default=200,type=int)
        self.parser.add_argument('--val_intv',default=1,type=int)
        self.parser.add_argument('--val_disp', default=1, type=int)
        self.parser.add_argument('--val_save',default=1,type=int)
        #检查点参数？
        self.parser.add_argument('--save_intv',default=1,type=int)
        #损失函数参数
        self.parser.add_argument('--normal_loss',default='cos',help='cos|mse')
        self.parser.add_argument('--normal_w',default=1)

    def parse(self):
        Base0pts.parse(self)
        self.args.train_img_num=self.args.in_img_num #数据归一化
        return self.args