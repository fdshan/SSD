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

    # draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i, j] > 0.5:  # if the network/ground_truth has high confidence on cell[i] with class[j]
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
    
    result_box = np.zeros_like(box_)
    result_confidence = np.zeros_like(confidence_)
    before_box = box_
    before_confidence = confidence_
    pred_box4, pred_box8 = getBox(box_, boxs_default)
    for class_num in range(0, 3):  # class cat, dog or person
        # Select the bounding box in A with the highest probability in class cat, dog or person
        cur_class = confidence_[:, class_num]  # [540]
        max_idx = np.argmax(cur_class)
        max_prob = cur_class[max_idx]
        max_box = box_[max_idx]  # [tx, ty, tw, th]

        # If that highest probability is greater than a threshold (threshold=0.5), proceed;
        # otherwise, the NMS is done
        while max_prob > threshold:
            # idx = np.setdiff1d(idx, max_idx)  # remove the current max from A
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

            # For all boxes in A, if a box has IOU greater than an overlap threshold (overlap=0.5) with x,
            # remove that box from A
            del_idx = np.where(ious > overlap)[0]  # index to delete
            confidence_[del_idx, :] = 0
            box_[del_idx, :] = 0
            max_idx = np.argmax(cur_class)
            max_prob = cur_class[max_idx]
            max_box = box_[max_idx]  # [gx, gy, gw, gh]

    return result_confidence, result_box


def getText(cur_height, cur_width, save_path, img_name, image_, pred_box, pred_confidence, boxs_default):
    # save_path - predicted_boxes/test, predicted_boxes/train
    # cur_width, cur_height - original image h & w

    image = image_.reshape((3, 320, 320))
    _, height, width = image.shape
    image = np.transpose(image, (1, 2, 0))
    g_hat, _ = getBox(pred_box, boxs_default)
    a = np.where(pred_confidence != 0)[0]
    a = np.unique(a)
    box_num = a
    content = []
    gx = g_hat[:, 0]  # center_x
    gy = g_hat[:, 1]  # center_y
    gw = g_hat[:, 2]  # width
    gh = g_hat[:, 3]  # height
    for i in box_num:
        class_id = np.argmax(pred_confidence[i, :])
        if class_id != 3:
            x_min = (gx[i] - gw[i] / 2) * cur_width
            y_min = (gy[i] - gh[i] / 2) * cur_height
            w = gw[i] * cur_width
            h = gh[i] * cur_height
            each_line = [class_id, x_min, y_min, w, h]
            content.append(each_line)

    result = np.asarray(content)
    paths = save_path + img_name + '.txt'
    np.savetxt(paths, result)

'''
if __name__ == '__main__':
    a = 0
'''











