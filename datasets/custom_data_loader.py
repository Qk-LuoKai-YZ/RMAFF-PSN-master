import torch.utils.data

def customDataloader(args):
    print("=> fetching img pairs in %s" % (args.data_dir))
    if args.dataset == 'PS_Synth_Dataset':
        from datasets.PS_Synth_Dataset import PS_Synth_Dataset   #中断去写 PS_SYNTH_DATASET文件
        train_set  = PS_Synth_Dataset(args,args.data_dir,'train')
        val_set    = PS_Synth_Dataset(args,args.data_dir,'val')
    else:
        raise Exception('Unknown dataset: %s' % (args.dataset))  #抛出异常

    if args.concat_data:
        print('******* using concat data ******')
        print("=> fetching img pairs in %s" % (args.data_dir2))
        #更快的训练出一个epoch 只使用一个数据集，正式的时候去掉注释
        train_set2  = PS_Synth_Dataset(args,args.data_dir2,'train')
        val_set2    = PS_Synth_Dataset(args,args.data_dir2,'val')
        train_set   = torch.utils.data.ConcatDataset([train_set,train_set2]) #使用两个数据集进行训练，上面是只使用一个第一个数据集
        val_set     = torch.utils.data.ConcatDataset([val_set,val_set2])

    print('\t found data: %d train and %d val' % (len(train_set),len(val_set))) #训练和测试的数据集数量
    print('\t train batch %d ,val batch: %d'  %(args.batch,args.val_batch)) #batch的大小

    train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch,num_workers=args.workers,pin_memory=args.cuda,shuffle=True)
    test_loader  = torch.utils.data.DataLoader(val_set , batch_size=args.val_batch,num_workers=args.workers, pin_memory=args.cuda, shuffle=False)
    return train_loader, test_loader

def benchmarkLoader(args):
    print("=> fetching img pairs in data/%s" % (args.benchmark))
    if args.benchmark == 'DiLiGenT_main':
        from datasets.DiLiGenT_main import DiLiGenT_main
        test_set = DiLiGenT_main(args,'test')
    else:
        raise Exception('Unknown benchmark')

    print('\t Found Benchmark Data : %d samples' % (len(test_set)))
    print('\t Test Batch %d' % (args.test_batch))

    test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.test_batch,num_workers=args.workers,pin_memory=args.cuda,shuffle=False)
    return test_loader



































