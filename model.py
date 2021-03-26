import os
import random
import numpy as np

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


def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    # input:
    # pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    # pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    # ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    # output:
    # loss -- a single number for the value of the loss function, [1]
    
    # TODO: write a loss function for SSD
    #
    # For confidence (class labels), use cross entropy (F.cross_entropy)
    # You can try F.binary_cross_entropy and see which loss is better
    # For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    # Note that you need to consider cells carrying objects and empty cells separately.
    # I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    # and reshape box to [batch_size*num_of_boxes, 4].
    # Then you need to figure out how you can get the indices of all cells carrying objects,
    # and use confidence[indices], box[indices] to select those cells.
    pred_confidence = pred_confidence.reshape((-1, 4))
    ann_confidence = ann_confidence.reshape((-1, 4))
    pred_box = pred_box.reshape((-1, 4))
    ann_box = ann_box.reshape((-1, 4))

    obj = torch.where(ann_confidence[:, 3] == 0)
    no_obj = torch.where(ann_confidence[:, 3] == 1)
    idx = obj[0]
    no_idx = no_obj[0]
    target = torch.where(ann_confidence[idx] == 1)[1]  # class id
    no_target = torch.where(ann_confidence[no_idx] == 1)[1]

    loss_conf = F.cross_entropy(pred_confidence[idx], target) + 3 * F.cross_entropy(
        pred_confidence[no_idx], no_target)
    loss_box = F.smooth_l1_loss(pred_box[obj], ann_box[obj])
    loss = loss_conf + loss_box

    return loss


class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num  # num_of_classes, in this assignment, 4: cat, dog, person, background
        
        # TODO: define layers
        # add bias term in the last few conv layers (left1234 & right1234)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.left1 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)  # reshape [N, 16, 100]
        self.right1 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)  # reshape [N, 16, 100]

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.left2 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)  # reshape [N, 16, 25]
        self.right2 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.left3 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)  # reshape [N, 16, 9]
        self.right3 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.left4 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, stride=1)  # reshape [N, 16, 1]
        self.right4 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, stride=1)

    def forward(self, x):
        # input:
        # x -- images, [batch_size, 3, 320, 320]
        
        x = x/255.0  # normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        # TODO: define forward
        #print('x', x.shape)
        x1 = self.conv1(x)
        #print('x1', x1.shape)

        x2 = self.conv2(x1)
        #print('x2', x2.shape)
        x_left1 = self.left1(x1)
        #print('x_left1', x_left1.shape)
        x_left1 = x_left1.reshape((-1, 16, 100))
        x_right1 = self.right1(x1)
        #print('x_right1', x_right1.shape)
        x_right1 = x_right1.reshape((-1, 16, 100))

        x3 = self.conv3(x2)
        #print('x3', x3.shape)
        x_left2 = self.left2(x2)
        #print('x_left2', x_left2.shape)
        x_left2 = x_left2.reshape((-1, 16, 25))
        x_right2 = self.right2(x2)
        #print('x_right2', x_right2.shape)
        x_right2 = x_right2.reshape((-1, 16, 25))

        x4 = self.conv4(x3)
        #print('x4', x4.shape)
        x_left3 = self.left3(x3)
        #print('x_left3', x_left3.shape)
        x_left3 = x_left3.reshape((-1, 16, 9))
        x_right3 = self.right3(x3)
        #print('x_right3', x_right3.shape)
        x_right3 = x_right3.reshape((-1, 16, 9))

        x_left4 = self.left4(x4)
        #print('x_left4', x_left4.shape)
        x_left4 = x_left4.reshape((-1, 16, 1))
        x_right4 = self.right4(x4)
        #print('x_right4', x_right4.shape)
        x_right4 = x_right4.reshape((-1, 16, 1))

        x_bbox = torch.cat((x_left1, x_left2, x_left3, x_left4), dim=2)  # [N, 16, 135]
        x_bbox = x_bbox.permute((0, 2, 1))  # [N, 135, 16]
        bboxes = x_bbox.reshape((-1, 540, 4))

        x_conf = torch.cat((x_right1, x_right2, x_right3, x_right4), dim=2)  # [N, 16, 135]
        x_conf = x_conf.permute((0, 2, 1))  # [N, 135, 16]
        x_conf = x_conf.reshape((-1, 540, 4))
        confidence = torch.softmax(x_conf, dim=2)

        # should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        # sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        # confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        # bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        
        return confidence, bboxes

'''
if __name__ == '__main__':
    ssd = SSD

'''







