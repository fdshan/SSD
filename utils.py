import numpy as np
import cv2
from dataset import iou


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def visualize_pred(epoch, windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)

    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    height, width, _ = image.shape
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    # image1: draw ground truth bounding boxes on image1
    # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    # image3: draw network-predicted bounding boxes on image3
    # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i, j] > 0.5:  # if the network/ground_truth has high confidence on cell[i] with class[j]
                # TODO:
                # image1: draw ground truth bounding boxes on image1
                px = boxs_default[i, 0]
                py = boxs_default[i, 1]
                pw = boxs_default[i, 2]
                ph = boxs_default[i, 3]
                dx = ann_box[i, 0]
                dy = ann_box[i, 1]
                dw = ann_box[i, 2]
                dh = ann_box[i, 3]
                gx = pw*dx+px
                gy = ph*dy+py
                gw = pw*np.exp(dw)
                gh = ph*np.exp(dh)
                x_min = int((gx-gw/2)*width)
                y_min = int((gy-gh/2)*height)
                x_max = int((gx+gw/2)*width)
                y_max = int((gy+gh/2)*height)
                start_point = (x_min, y_min) #top left corner, x1<x2, y1<y2
                end_point = (x_max, y_max) #bottom right corner
                color = colors[j] #use red green blue to represent different classes
                thickness = 2
                image1 = cv2.rectangle(image1, start_point, end_point, color, thickness)
                # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                cell_x_min = int(boxs_default[i, 4]*width)
                cell_y_min = int(boxs_default[i, 5]*height)
                cell_x_max = int(boxs_default[i, 6]*width)
                cell_y_max = int(boxs_default[i, 7]*height)
                cell_start = (cell_x_min, cell_y_min)
                cell_end = (cell_x_max, cell_y_max)
                image2 = cv2.rectangle(image2, cell_start, cell_end, color, thickness)
    # pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i, j] > 0.5:
                # TODO:
                # image3: draw network-predicted bounding boxes on image3
                px = boxs_default[i, 0]
                py = boxs_default[i, 1]
                pw = boxs_default[i, 2]
                ph = boxs_default[i, 3]
                dx = pred_box[i, 0]
                dy = pred_box[i, 1]
                dw = pred_box[i, 2]
                dh = pred_box[i, 3]
                pred_x = pw * dx + px
                pred_y = ph * dy + py
                pred_w = pw * np.exp(dw)
                pred_h = ph * np.exp(dh)
                x_min = int((pred_x - pred_w / 2) * width)
                y_min = int((pred_y - pred_h / 2) * height)
                x_max = int((pred_x + pred_w / 2) * width)
                y_max = int((pred_y + pred_h / 2) * height)
                start_point = (x_min, y_min)
                end_point = (x_max, y_max)
                color = colors[j]  # use red green blue to represent different classes
                thickness = 2
                image3 = cv2.rectangle(image3, start_point, end_point, color, thickness)
                # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                cell_x_min = int(boxs_default[i, 4] * width)
                cell_y_min = int(boxs_default[i, 5] * height)
                cell_x_max = int(boxs_default[i, 6] * width)
                cell_y_max = int(boxs_default[i, 7] * height)
                cell_start = (cell_x_min, cell_y_min)
                cell_end = (cell_x_max, cell_y_max)
                image4 = cv2.rectangle(image4, cell_start, cell_end, color, thickness)
    # combine four images into one
    h, w, _ = image1.shape
    image = np.zeros([h*2, w*2, 3], np.uint8)
    image[:h, :w] = image1
    image[:h, w:] = image2
    image[h:, :w] = image3
    image[h:, w:] = image4
    # cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    # cv2.waitKey(1)
    # if you are using a server, you may not be able to display the image.
    # in that case, please save the image using cv2.imwrite and check the saved image for visualization.
    # cv2.imwrite("results/{0}_example_{1}.jpg".format(windowname, epoch), image)
    cv2.imwrite("results/{0}/example_{1}.jpg".format(windowname, epoch), image)

def getBox(pred_box, boxes_default):
    # pred_box [num_of_boxes, 4] - [dx, dy, dw, dh]
    # boxes_default [num_of_boxes, 8] - [px, py, pw, ph, x_min, y_min, x_max, y_max]
    # output: box with g hat [num_of_boxes, 4] - [gx, gy, gw, gh]
    # box with g hat [num_of_boxes, 8] - [0, 0, 0, 0, x_min, y_min, x_max, y_max]
    box4 = np.zeros_like(pred_box)
    box8 = np.zeros_like(boxes_default)

    dx = pred_box[:, 0]
    dy = pred_box[:, 1]
    dw = pred_box[:, 2]
    dh = pred_box[:, 3]
    px = boxes_default[:, 0]
    py = boxes_default[:, 1]
    pw = boxes_default[:, 2]
    ph = boxes_default[:, 3]
    gx = pw*dx+px
    gy = ph*dy+py
    gw = pw*np.exp(dw)
    gh = ph*np.exp(dh)
    box4[:, 0] = gx
    box4[:, 1] = gy
    box4[:, 2] = gw
    box4[:, 3] = gh
    box8[:, 4] = gx-gw/2
    box8[:, 5] = gy-gh/2
    box8[:, 6] = gx+gw/2
    box8[:, 7] = gy+gh/2
    return box4, box8


def non_maximum_suppression(confidence_, box_, boxs_default, overlap, threshold):
    # input:
    # confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    # boxs_default -- default bounding boxes, [num_of_boxes, 8]
    # overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    # threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    # output:
    # depends on your implementation.
    # if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    # you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    # visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default)
    
    #TODO: non maximum suppression
    result_box = np.zeros_like(box_)
    result_confidence = np.zeros_like(confidence_)
    before_box = box_
    before_confidence = confidence_
    pred_box4, pred_box8 = getBox(box_, boxs_default)
    for class_num in range(0, 3):  # class cat, dog or person
        # Select the bounding box in A with the highest probability in class cat, dog or person
        cur_class = confidence_[:, class_num]  # [540]
        idx = np.where(cur_class)[0]
        # print(len(idx))
        max_idx = np.argmax(cur_class)
        max_prob = cur_class[max_idx]
        max_box = box_[max_idx]  # [tx, ty, tw, th]
        # print(max_idx)
        # print(max_prob)

        # If that highest probability is greater than a threshold (threshold=0.5), proceed;
        # otherwise, the NMS is done
        while max_prob > threshold:
            # idx = np.setdiff1d(idx, max_idx)  # remove the current max from A
            # print('idx', idx)
            result_box[max_idx, :] = max_box
            result_confidence[max_idx, :] = confidence_[max_idx]
            confidence_[max_idx, :] = 0
            box_[max_idx, :] = 0
            gx, gy, gw, gh = pred_box4[max_idx]
            x_min = gx - gw / 2
            y_min = gy - gh / 2
            x_max = gx + gw / 2
            y_max = gx + gh / 2
            # iou(boxs_default, x_min, y_min, x_max, y_max):
            ious = iou(pred_box8, x_min, y_min, x_max, y_max)  # shape = [540]
            # b = np.unique(ious)
            # c = np.unique(confidence_)
            # For all boxes in A, if a box has IOU greater than an overlap threshold (overlap=0.5) with x,
            # remove that box from A
            ious_true = ious[ious > overlap]
            del_idx = np.where(ious > overlap)[0]  # index to delete
            confidence_[del_idx, :] = 0
            box_[del_idx, :] = 0
            # print('before', len(idx))
            # idx = np.delete(idx, del_idx)  # remove boxes from A
            # if len(idx) == 0:  # A is empty, break
            #    break
            # print('idx after', idx)
            # print(len(idx))
            # a = len(idx)
            # cur_class = confidence_[idx, class_num]
            # max_prob = np.max(cur_class)
            # max_idx = idx[confidence_[idx, class_num] == max_prob][0]  # make sure max is picked
            # max_box = pred_box4[max_idx]
            max_idx = np.argmax(cur_class)
            max_prob = cur_class[max_idx]
            max_box = box_[max_idx]  # [gx, gy, gw, gh]
    # convert to np array
    # result_box = np.asarray(result_box)
    # result_confidence = np.asarray(result_confidence)

    return result_confidence, result_box


'''
if __name__ == '__main__':
    a = 0
'''











