import argparse
import os
import numpy as np
import time
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

from dataset import *
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
# please google how to use argparse
# a short intro:
# to train: python main.py
# to test:  python main.py --test


class_num = 4  # cat dog person background

num_epochs = 100
batch_size = 16  # origin=32, change to 16 due to the cuda mem limitation


boxs_default = default_box_generator([10, 5, 3, 1], [0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7])


# Create network
network = SSD(class_num)
network.cuda()
cudnn.benchmark = True


if not args.test:
    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train=True, image_size=320)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train=False, image_size=320)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    # feel free to try other optimizers and parameters.
    
    start_time = time.time()
    print('train!')
    for epoch in range(num_epochs):
        # TRAINING
        network.train()

        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_, _, _, _= data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1

        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        
        # visualize
        if epoch % 5 == 0:
            pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
            pred_box_ = pred_box[0].detach().cpu().numpy()

            visualize_pred(epoch, "train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)

            after_nms_pred_conf, after_nms_pred_box = non_maximum_suppression(pred_confidence_, pred_box_, boxs_default, overlap=0.2, threshold=0.6)
            visualize_pred(epoch, 'nms_train', after_nms_pred_conf, after_nms_pred_box, ann_confidence_[0].numpy(),
                           ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        # VALIDATION
        network.eval()
        
        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_, _, _, _ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            pred_confidence, pred_box = network(images)
            
            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()
            
            # optional: implement a function to accumulate precision and recall to compute mAP or F1.
            # update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)
        
        # visualize
        if epoch % 5 == 0:
            pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
            pred_box_ = pred_box[0].detach().cpu().numpy()
            visualize_pred(epoch, "val", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)

            after_nms_pred_conf, after_nms_pred_box = non_maximum_suppression(pred_confidence_, pred_box_, boxs_default, overlap=0.2, threshold=0.6)
            visualize_pred(epoch, 'nms_val', after_nms_pred_conf, after_nms_pred_box, ann_confidence_[0].numpy(),
                           ann_box_[0].numpy(), images_[0].numpy(), boxs_default)

        # optional: compute F1
        # F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        # print(F1score)
        
        # save weights
        if epoch % 10 == 9:
            # save last network
            print('saving net...')
            torch.save(network.state_dict(), 'checkpoints/network{0}.pth'.format(epoch))

    print('save final pth!')
    torch.save(network.state_dict(), 'checkpoints/network_final.pth')

else:
    # TEST
    print('test!')
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train=False, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('checkpoints/network_final.pth'))
    network.eval()
    count = 0
    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_, cur_name, height, width = data
        num = cur_name[0]
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()

        nms_pred_confidence_, nms_pred_box_ = non_maximum_suppression(pred_confidence_, pred_box_, boxs_default,
                                                                          overlap=0.2, threshold=0.6)
        a = np.where(nms_pred_confidence_ != 0)[0]
        a = np.unique(a)
        if len(a) == 0:
            count += 1

        getText(height, width, 'predicted_boxes/train/', num, images_, nms_pred_box_, nms_pred_confidence_, boxs_default)

        visualize_pred(num, "nms_test", nms_pred_confidence_, nms_pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)

        if i % 100 == 0:
            print('{} image...'.format(i))

    print(count)

