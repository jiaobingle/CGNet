###########################################################################
# Created by: Tianyi Wu
# Email: wutianyi@ict.ac.cn 
# Copyright (c) 2018
###########################################################################
from utils.summary import summary
import model.CGNet as net
model = net.Context_Guided_Network(19, M=3, N=21)
model.cuda()
summary(model,(3,640, 640))
