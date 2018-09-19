###########################################################################
# Created by: Tianyi Wu
# Email: wutianyi@ict.ac.cn 
# Copyright (c) 2018
###########################################################################
import numpy as np
import os
import torch
import pickle
import random
import model.CGNet as CGNet
from torch.autograd import Variable
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.nn as nn
from utils.loss import CrossEntropyLoss2d
from utils.metric import get_iou
from  dataset.cityscapes_datasets import CityscapesDataSet,CityscapesValDataSet, CityscapesTrainInform
import torchvision.transforms as transforms
import time
from argparse import ArgumentParser
import timeit

def val(args, val_loader, model, criterion):
    '''
    Args:
      val_loader: loaded for validation dataset
      model: model
      criterion: loss function
    return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    #evaluation mode
    model.eval()
    total_batches = len(val_loader)
   
    interp = nn.Upsample(size=(1024, 2048), mode='bilinear')
    
    data_list=[]
    for i, (input, label, size, name) in enumerate(val_loader):
        start_time = time.time()
        input_var = Variable(input, volatile=True).cuda()
        # run the mdoel
        output = model(input_var)
        time_taken = time.time() - start_time
        print('[%d/%d]  time: %.2f' % (i, total_batches, time_taken))
        
        output= interp(output)
        output= output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype = np.uint8)
        output= output.transpose(1,2,0)
        output= np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        data_list.append( [gt.flatten(), output.flatten()])
        

    meanIoU, per_class_iu= get_iou(data_list, args.classes)
    return meanIoU, per_class_iu

def adjust_learning_rate( args, cur_epoch, max_epoch, curEpoch_iter, perEpoch_iter, baselr):
    """
    poly learning stategyt
    lr = baselr*(1-iter/max_iter)^power
    """
    cur_iter = cur_epoch*perEpoch_iter + curEpoch_iter
    max_iter=max_epoch*perEpoch_iter
    lr = baselr*pow( (1 - 1.0*cur_iter/max_iter), 0.9)

    return lr


def train(args, train_loader, model, criterion, optimizer, epoch):
    '''
    Args:
       train_loader: loaded for training dataset
       model: model
       criterion: loss function
       optimizer: optimization algo, such as ADAM or SGD
       epoch: epoch number
    return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    #switch to train mode
    model.train()
    epoch_loss = []

    data_list=[]
    total_batches = len(train_loader)
    print("iteration numbers of per epoch: ", total_batches)
    for iteration, batch in enumerate(train_loader,0):
        
        lr= adjust_learning_rate(args, cur_epoch= epoch, max_epoch= args.max_epochs, curEpoch_iter= iteration, 
                                    perEpoch_iter= total_batches, baselr=args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr']=lr;
        start_time = time.time()
        images, labels, _, _ = batch
        images = Variable(images).cuda()
        labels = Variable(labels.long()).cuda()
        #run the mdoel
        output = model(images)
        print("model output size: ", output.size())
        loss = criterion(output, labels)
        #set the grad to zero
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        time_taken = time.time() - start_time

        #compute the confusion matrix
        gt = np.asarray(labels.cpu().data[0].numpy(), dtype = np.uint8)
        output = output.cpu().data[0].numpy()
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        data_list.append( [gt.flatten(), output.flatten()])

        print('===> Epoch[%d] (%d/%d) \tcur_lr: %.6f loss: %.3f time:%.2f' % (epoch, iteration, total_batches, lr,loss.item(), time_taken))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    meanIoU, per_class_iu= get_iou(data_list, args.classes)

    return average_epoch_loss_train, per_class_iu, meanIoU, lr


def netParams(model):
    '''
    Computing total network parameters
    Args:
       model: model
    return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters

def train_model(args):
    """
    Main function for training 
    Args:
       args: global arguments
    """
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print("=====> Check if processed data file exists or not")
    if not os.path.isfile(args.inform_data_file):
        print("%s is not found" %(args.inform_data_file))
        dataCollect = CityscapesTrainInform(args.data_dir, args.classes, train_set_file= args.dataset_list, 
                                            inform_data_file = args.inform_data_file) #collect mean std, weigth_class information
        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        datas = pickle.load(open(args.inform_data_file, "rb"))
    
    print(args)
    global network_type
     
    if args.cuda:
        print("=====> Use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    
    args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed) 
    
    cudnn.enabled = True
    M = args.M
    N = args.N
    # load the model
    print('=====> Building network')
    model = CGNet.Context_Guided_Network(classes= args.classes, M= M, N= N)
    network_type="CGNet"
    print("current architeture:  CGNet")
    args.savedir = args.savedir + network_type +"_M"+ str(M) + 'N' +str(N) + '/'

    # create the directory of checkpoint if not exist
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    
    print('=====> Computing network parameters')
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    print("data['classWeights']: ", datas['classWeights'])
    weight = torch.from_numpy(datas['classWeights'])
    criteria = CrossEntropyLoss2d(weight)
    criteria = criteria.cuda()
    print('=====> Dataset statistics')
    print('mean and std: ', datas['mean'], datas['std'])


    if args.cuda:
        if torch.cuda.device_count()>1:
            print("torch.cuda.device_count()=",torch.cuda.device_count())
            model = torch.nn.DataParallel(model).cuda()  #multi-card data parallel
        else:
            print("single GPU for training")
            model = model.cuda()  #single card 
    
    start_epoch = 0
    
    #DataLoader
    trainLoader = data.DataLoader(CityscapesDataSet(args.data_dir, args.train_data_list, crop_size = input_size, scale=args.random_scale, 
                                                    mirror=args.random_mirror, mean= datas['mean']),
                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    valLoader = data.DataLoader(CityscapesValDataSet(args.data_dir, args.val_data_list,f_scale=1,  mean= datas['mean']),
                                  batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)


    if args.resume:
        if os.path.isfile(args.resume):
            print("=====> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            print("=====> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=====> no checkpoint found at '{}'".format(args.resume))
    
    model.train()
    cudnn.benchmark= True
    
    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t" % ('Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val)'))
    logger.flush()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
 
    print('=====> beginning training')
    for epoch in range(start_epoch, args.max_epochs):
        lossTr, per_class_iu_tr, mIOU_tr, lr = train(args, trainLoader, model, criteria, optimizer, epoch)
        # evaluate on validation set
        if epoch % 50 ==0:
            mIOU_val, per_class_iu = val(args, valLoader, model, criteria)
            logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f" % (epoch, lossTr, mIOU_tr, mIOU_val, lr))
            logger.flush()
            print("Epoch : " + str(epoch) + ' Details')
            print("\nEpoch No.: %d\tTrain Loss = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f\t lr= %.6f" % (epoch, lossTr, mIOU_tr, mIOU_val, lr))
        else:
            logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.7f" % (epoch, lossTr, mIOU_tr, lr))
            logger.flush()
            print("Epoch : " + str(epoch) + ' Details')
            print("\nEpoch No.: %d\tTrain Loss = %.4f\t mIOU(tr) = %.4f\t lr= %.6f" % (epoch, lossTr, mIOU_tr, lr))
        #save the model
        model_file_name = args.savedir +'/model_' + str(epoch + 1) + '.pth'
        state = {"epoch": epoch+1, "model": model.state_dict()}
        torch.save(state, model_file_name)
    logger.close()

if __name__ == '__main__':
    start = timeit.default_timer()
    parser = ArgumentParser()
    parser.add_argument('--model', default="CGNet", help='Model name: Context Guided Network')
    parser.add_argument('--data_dir', default="/home/wty/AllDataSet/CityScapes", help='Data directory')
    parser.add_argument('--dataset_list', default="cityscapes_trainval_list.txt",help='train and val data, for computing the ratio of all kinds, mean and std')
    parser.add_argument('--train_data_list', default="./dataset/list/Cityscapes/cityscapes_trainval_list.txt", help='Data directory')
    parser.add_argument('--val_data_list', default="./dataset/list/Cityscapes/cityscapes_val_list.txt", help='Data directory')
    parser.add_argument('--scaleIn', type=int, default=1, help='For rescaling input image, default is 1, keep fixed size')  
    parser.add_argument('--max_epochs', type=int, default=350, help='Max. number of epochs')
    parser.add_argument('--input_size', type =str, default = '680,680', help='input size') 
    parser.add_argument('--random_mirror', type = bool, default = True, help='input image random mirror') 
    parser.add_argument('--random_scale', type = bool, default = True, help='input image resize 0.5 to 2') 
    parser.add_argument('--num_workers', type=int, default= 0, help='No. of parallel threads') 
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')

    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--savedir', default='./checkpoint/', help='directory to save the results')
    parser.add_argument('--resume', type=str, default='./checkpoint/CGNet_M3N21/model_28.pth', help='Use this flag to load last checkpoint for training')
    parser.add_argument('--classes', type=int, default=19, help='No of classes in the dataset. 19 for cityscapes')
    parser.add_argument('--inform_data_file', default='cityscapes_inform.pkl', help='store statistic information of trainset')
    parser.add_argument('--M', default=3, type=int, help='The number of block in stage 2')
    parser.add_argument('--N', default=21, type=int, help='The number of block in stage 3')

    parser.add_argument('--logFile', default='log.txt', help='File that stores the training and validation logs')
    parser.add_argument('--cuda', default=True, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument("--gpus", default="0,1", type=str, help="gpu ids (default: 0,1)")
    train_model(parser.parse_args())
    end = timeit.default_timer()
    print("training time:", 1.0*(end-start)/3600)

