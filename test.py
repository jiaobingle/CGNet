###########################################################################
# Created by: Tianyi Wu
# Email: wutianyi@ict.ac.cn 
# Copyright (c) 2018
###########################################################################
import os
import torch
import pickle
import random
import model.CGNet as net
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from  dataset.cityscapes_datasets import CityscapesTestDataSet
import time
from argparse import ArgumentParser
from utils.convert_state import convert_state_dict
from utils.colorize_mask import cityscapes_colorize_mask
from utils.loss import CrossEntropyLoss2d
from utils.metric import get_iou
from PIL import Image
import numpy as np
import torch.nn as nn

def test(args, test_loader, model):
    '''
    Args:
      test_loader: loaded for test dataset
      model: model
    '''
    #switch to evaluation mode
    model.eval()
    total_batches = len(test_loader)
    
    data_list=[]
    for i, (input , size, name) in enumerate(test_loader):
        start_time = time.time()
        #input_var = Variable(input, volatile=True).cuda()  torch version==0.3
        with torch.no_grad():
            input_var = Variable(input).cuda()
        # run the mdoel
        output = model(input_var)
        time_taken = time.time() - start_time
        print('[%d/%d]  time: %.2f' % (i, total_batches, time_taken))
        
        # save seg image
        output= output.cpu().data[0].numpy()  # 1xCxHxW ---> CxHxW
        output= output.transpose(1,2,0) # CxHxW --> HxWxC
        output= np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        output_color = cityscapes_colorize_mask(output)
        output = Image.fromarray(output)
        output_color.save('%s/%s.png' % (args.save_seg_dir, name[0]))

def test_func(args):
    '''
     Main function for testing
     param args: global arguments
     return: None
    '''
    print(args)
    global network_type

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    
    args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed) 
    
    
    print( 'checking if processed cached_data_file exists or not')
    if not os.path.isfile(args.cached_data_file):
        dataCollect = StatisticalInformDataset(args.data_dir, args.classes, args.cached_data_file)
        data= dataCollect.collectDataAndSave()
        if data is  None:
            print("error while pickling data, please check")
            exit(-1)
    else:
        data = pickle.load(open(args.cached_data_file, "rb"))

    M = args.M
    N = args.N
    # load the model
    print('====> Building network')
    model = net.Context_Guided_Network(classes= args.classes, M=M, N=N)
    network_type="CGNet"
    print("Arch:  CGNet")

    if args.cuda:
        #model = torch.nn.DataParallel(model).cuda()  # multi-card testing
        model = model.cuda()  # single-card testing
    print('Dataset statistics')
    print('mean and std: ', data['mean'], data['std'])
    print('classWeights: ', data['classWeights'])

    if args.save_seg_dir:
        if not os.path.exists(args.save_seg_dir):
            os.makedirs(args.save_seg_dir)


    testLoader = torch.utils.data.DataLoader(
            CityscapesTestDataSet(args.data_dir,args.test_data_list, mean= data['mean']),
        batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.cuda:
        cudnn.benchmark = True

    print("=====> load pretrained model")
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(convert_state_dict(checkpoint['model']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    # evaluate on test set
    print("=====> beginning test")
    print("test set length: ",len(testLoader))
    test(args, testLoader, model)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model', default="CGNet", help='model name')
    parser.add_argument('--data_dir', default="/home/wty/AllDataSet/CityScapes", help='Data directory')
    parser.add_argument('--test_data_list', default="./dataset/list/Cityscapes/cityscapes_test_list.txt", help='test set file')
    parser.add_argument('--scaleIn', type=int, default=1, help='rescale for input image')  
    parser.add_argument('--num_workers', type=int, default= 1, help='Numbers of parallel threads') 
    parser.add_argument('--batch_size', type=int, default=1)  
    parser.add_argument('--resume', type=str, default='./checkpoint/CGNet_M3N21/model_350.pth', help='the checkpoint for testing')
    parser.add_argument('--classes', type=int, default=19, help='No of classes in the dataset. 19 for cityscapes')
    parser.add_argument('--cached_data_file', default='cityscapes_inform.pkl', help='Cached file name')
    parser.add_argument('--cuda', default=True, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--M', default=3, type=int, help='the number of blocks in stage 2')
    parser.add_argument('--N', default=21, type=int, help='the number of blocks in stage 3')
    parser.add_argument('--save_seg_dir', type=str, default='./result/TestSet_350_1/', help='the directory of saving segmenation results')
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    
    test_func(parser.parse_args())

