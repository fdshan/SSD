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
import numpy as np
import os
import cv2


# generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    # input:
    # layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    # large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    # small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    # output:
    # boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    # TODO:
    # create an numpy array "boxes" to store default bounding boxes
    # you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    # the first dimension means number of cells, 10*10+5*5+3*3+1*1
    # the second dimension 4 means each cell has 4 default bounding boxes.
    # their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    # where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    # for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    # the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]

    boxes = np.zeros((135, 4, 8))
    idx, num = 0, 0
    for cell_size in layers:  # cell_size [10, 5, 3, 1]
        lsize = large_scale[idx]
        ssize = small_scale[idx]
        cell_step = 1/cell_size
        box1 = [ssize, ssize]
        box2 = [lsize, lsize]
        box3 = [lsize * np.sqrt(2), lsize / np.sqrt(2)]
        box4 = [lsize / np.sqrt(2), lsize * np.sqrt(2)]

        # each cell
        # [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
        for x in range(cell_size):
            for y in range(cell_size):
                center_x = x*cell_step+cell_step/2
                center_y = y*cell_step+cell_step/2

                # first box
                x_min1 = center_x-box1[0]/2
                y_min1 = center_y-box1[1]/2
                x_max1 = center_x+box1[0]/2
                y_max1 = center_y+box1[1]/2
                # clipping
                if x_min1 < 0: x_min1 = 0
                if y_min1 < 0: y_min1 = 0
                if x_max1 > 1: x_max1 = 1
                if y_max1 > 1: y_max1 = 1
                if box1[0] > 1: box1[0] = 1
                if box1[1] > 1: box1[1] = 1
                boxes[num, 0, :] = [center_x, center_y, box1[0], box1[1], x_min1, y_min1, x_max1, y_max1]

                # second box
                x_min2 = center_x - box2[0] / 2
                y_min2 = center_y - box2[1] / 2
                x_max2 = center_x + box2[0] / 2
                y_max2 = center_y + box2[1] / 2
                if x_min2 < 0: x_min2 = 0
                if y_min2 < 0: y_min2 = 0
                if x_max2 > 1: x_max2 = 1
                if y_max2 > 1: y_max2 = 1
                if box2[0] > 1: box2[0] = 1
                if box2[1] > 1: box2[1] = 1
                boxes[num, 1, :] = [center_x, center_y, box2[0], box2[1], x_min2, y_min2, x_max2, y_max2]

                # third box
                x_min3 = center_x - box3[0] / 2
                y_min3 = center_y - box3[1] / 2
                x_max3 = center_x + box3[0] / 2
                y_max3 = center_y + box3[1] / 2
                if x_min3 < 0: x_min3 = 0
                if y_min3 < 0: y_min3 = 0
                if x_max3 > 1: x_max3 = 1
                if y_max3 > 1: y_max3 = 1
                if box3[0] > 1: box3[0] = 1
                if box3[1] > 1: box3[1] = 1
                boxes[num, 2, :] = [center_x, center_y, box3[0], box3[1], x_min3, y_min3, x_max3, y_max3]

                # forth box
                x_min4 = center_x - box4[0] / 2
                y_min4 = center_y - box4[1] / 2
                x_max4 = center_x + box4[0] / 2
                y_max4 = center_y + box4[1] / 2
                if x_min4 < 0: x_min4 = 0
                if y_min4 < 0: y_min4 = 0
                if x_max4 > 1: x_max4 = 1
                if y_max4 > 1: y_max4 = 1
                if box4[0] > 1: box4[0] = 1
                if box4[1] > 1: box4[1] = 1
                boxes[num, 3, :] = [center_x, center_y, box4[0], box4[1], x_min4, y_min4, x_max4, y_max4]
                num += 1
        idx += 1
    boxes = boxes.reshape((540, 8))  # [box_num,8]
    return boxes


# this is an example implementation of IOU.
# It is different from the one used in YOLO, please pay attention.
# you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min, y_min, x_max, y_max):
    # input:
    # boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    # x_min,y_min,x_max,y_max -- another box (box_r)
    
    # output:
    # ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:, 6], x_max)-np.maximum(boxs_default[:, 4], x_min), 0)*np.maximum(np.minimum(boxs_default[:, 7], y_max)-np.maximum(boxs_default[:, 5], y_min), 0)
    area_a = (boxs_default[:, 6]-boxs_default[:, 4])*(boxs_default[:, 7]-boxs_default[:, 5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union, 1e-8)


def match(ann_box, ann_confidence, boxs_default, threshold, cat_id, x_min, y_min, x_max, y_max):
    # input:
    # ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    # ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    # boxs_default            -- [num_of_boxes,8], default bounding boxes
    # threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    # cat_id                  -- class id, 0-cat, 1-dog, 2-person
    # x_min,y_min,x_max,y_max -- bounding box
    
    # compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min, y_min, x_max, y_max)
    # print(np.unique(ious))
    ious_true = ious > threshold
    gw = x_max - x_min
    gh = y_max - y_min
    gx = x_min + gw / 2
    gy = y_min + gh / 2
    unique = np.unique(ious_true)
    # TODO:
    # update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    # if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    # this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    if True not in ious_true:
        ious_true = np.argmax(ious)  # make sure at least one default bounding box is used
        # print('here')
        # idx = np.where(ious_true == True)
        a = boxs_default[ious_true]
        # obj_cell = idx[0]

        px, py, pw, ph = boxs_default[ious_true][0:4]
        tx = (gx - px) / pw
        ty = (gy - py) / ph
        tw = np.log(gw / pw)
        th = np.log(gh / ph)
        if tx < 0: tx = 0  # clipping
        if tx > 1: tx = 1
        if ty < 0: ty = 0
        if ty > 1: ty = 1
        if tw < 0: tw = 0
        if tw > 1: tw = 1
        if th < 0: th = 0
        if th > 1: th = 1
    else:  # multiple bounding boxes
        a = boxs_default[ious_true]
        # idx = np.where(ious_true == True)
        px = boxs_default[ious_true][:, 0]
        py = boxs_default[ious_true][:, 1]
        pw = boxs_default[ious_true][:, 2]
        ph = boxs_default[ious_true][:, 3]
        tx = (gx - px) / pw
        ty = (gy - py) / ph
        tw = np.log(gw / pw)
        th = np.log(gh / ph)
        tx[tx < 0] = 0  # clipping
        tx[tx > 1] = 1
        ty[ty < 0] = 0
        ty[ty > 1] = 1
        tw[tw < 0] = 0
        tw[tw > 1] = 1
        th[th < 0] = 0
        th[th > 1] = 1
    b = [tx, ty, tw, th]
    ann_box[ious_true, :] = np.asarray([tx, ty, tw, th]).T
    # ann_box[ious_true, :] = [tx, ty, tw, th]
    ann_confidence[ious_true, cat_id] = 1
    ann_confidence[:, -1] = 0  # not background
    return ann_box, ann_confidence

class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train=True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        # overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        
        # notice:
        # you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num, 4], np.float32)  # bounding boxes
        ann_confidence = np.zeros([self.box_num, self.class_num], np.float32)  # one-hot vectors
        final_box = np.zeros_like(ann_box)  # for multiple objects
        final_confidence = np.zeros_like(ann_confidence)  # for multiple objects
        # one-hot vectors with four classes
        # [1,0,0,0] -> cat
        # [0,1,0,0] -> dog
        # [0,0,1,0] -> person
        # [0,0,0,1] -> background
        
        ann_confidence[:, -1] = 1  # the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        
        # TODO:
        # 1. prepare the image [3,320,320], by reading image "img_name" first.
        # 2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        # 3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        # 4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        
        # to use function "match":
        # match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        # where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        # note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        # For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        image = cv2.imread(img_name)  # [320, 320, 3]
        height, width, _ = image.shape
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.transpose(2, 0, 1)  # [3, 320, 320]
        if self.train:
            ann_info = open(ann_name)
            info = ann_info.readlines()
            for object_info in info:
                obj = object_info.split()

                class_id, x_min, y_min, w, h = obj
                class_id = int(class_id)
                x_min = (float(x_min))
                y_min = (float(y_min))
                w = (float(w))
                h = (float(h))

                x_max = ((x_min + w) / width)
                y_max = ((y_min + h) / height)
                x_min = (x_min / width)
                y_min = (y_min / height)

                ann_box, ann_confidence = match(ann_box, ann_confidence, self.boxs_default, self.threshold, class_id, x_min, y_min, x_max, y_max)

                final_box = final_box + ann_box
                final_confidence = final_confidence + ann_confidence
        return image, final_box, final_confidence
