from  .base_opts import Base0pts
class RunModelOpts(Base0pts):
    def __init__(self):
        super(RunModelOpts,self).__init__()
        self.initialize()

    def initialize(self):
        Base0pts.initialize(self)
        #模型和数据设置
        self.parser.add_argument('--run_model',default=True,action='store_false')
        #self.parser.add_argument('--benchmark',default='DiLiGenT_main')
        #self.parser.add_argument('--bm_dir',default='../../data/datasets/PSOUC')

        self.parser.add_argument('--benchmark', default='DiLiGenT_main')
        self.parser.add_argument('--bm_dir', default='../../data/datasets/DiLiGenT/pmsData')

        self.parser.add_argument('--model', default='RMAFF_PSN_run')
        self.parser.add_argument('--test_batch',default=1,type=int)

        #显示和保存
        self.parser.add_argument('--test_intv',default=1,type=int)
        self.parser.add_argument('--test_disp',default=1,type=int)
        self.parser.add_argument('--test_save',default=1,type=int)

    def parse(self):
        Base0pts.parse(self)
        return self.args
