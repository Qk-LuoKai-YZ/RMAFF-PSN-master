from . import model_utils

def buildModel(args):
    print('creating model %s' % (args.model))
    in_c = model_utils.getInputChanel(args)   #获取输入通道
    other = {'img_num' : args.in_img_num, 'in_light' : args.in_light}
    if args.model == 'RMAFF_PSN':
        from models.RMAFF_PSN import RMAFF_PSN
        model = RMAFF_PSN(args.fuse_type,args.use_BN,in_c,other)
    elif args.model == 'RMAFF_PSN_run':
        from models.RMAFF_PSN_run import RMAFF_PSN
        model = RMAFF_PSN(args.fuse_type, args.use_BN, in_c, other)
    else:
        raise Exception("=>unknow model '{}'".format(args.model))

    if args.cuda:
        model = model.cuda()

    if args.retrain:
        print("=>using pre-trained model %s" %(args.retrain))
        model_utils.loadCheckpoint(args.retrain,model,cuda=args.cuda) #转去写  load函数

    if args.resume:
        print("=>resume loading checkpoint %s" %(args.resume))
        model_utils.loadCheckpoint(args.resume,model,cuda=args.cuda)
    print(model)
    print("=> model parameters: %d " %(model_utils.get_n_params(model)))
    return model