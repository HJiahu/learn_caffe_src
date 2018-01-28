#coding:utf8
import sys
sys.path.append('../12net')
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU
anno_file = "wider_face_train.txt"
im_dir = "WIDER_train/images"
pos_save_dir = "12/positive"
part_save_dir = "12/part"
neg_save_dir = '12/negative'
save_dir = "./12"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
# pos 和 neg 用于训练分类器
# pos 和 part 用于边框回归
f1 = open(os.path.join(save_dir, 'pos_12.txt'), 'w') # 与ground bbx相交大于0.65的框
f2 = open(os.path.join(save_dir, 'neg_12.txt'), 'w') # 与ground bbx相交小于0.3的框
f3 = open(os.path.join(save_dir, 'part_12.txt'), 'w') # 与ground bbx相交面积在0.4与0.65之间的框
# 读取所有的标记
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print("%d pics in total" % num)
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0
for annotation in annotations:
    # annotation 示例：
    # 0--Parade/0_Parade_marchingband_1_242 21.79 401.00 35.95 417.69 273.33 439.83 296.93 462.34 326.69 363.97 336.13 378.85 193.84 506.98 215.98 531.30 323.40 236.71 362.28 296.46 440.22 400.73 455.46 417.78 393.83 364.57 402.93 377.99 491.62 355.47 502.53 369.57 544.60 351.83 558.24 368.89 420.46 548.68 450.90 586.23 340.58 565.68 371.64 601.68 239.56 331.83 246.18 340.47 781.56 392.01 794.33 408.93 797.21 421.38 813.81 442.45 796.25 440.85 812.85 467.03 839.35 354.98 855.95 374.13 926.90 354.06 942.14 368.91 970.12 353.09 983.62 369.49 57.71 358.03 66.33 370.24
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    bbox = list(map(float, annotation[1:]))
    # boxes 的每一行都为四元组，标志一个bbx
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
    idx += 1 # 处理计数
    if idx % 100 == 0:
        print(idx, "images done")

    height, width, channel = img.shape

    neg_num = 0
    # 以 bbx 为基础随机裁剪50张子图，如果IOU小于0.3则作为负样本
    while neg_num < 50:
        size = npr.randint(40, min(width, height) / 2)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny : ny + size, nx : nx + size, :]
        resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write("12/negative/%s"%n_idx + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1
    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # 因为小的bbx原本就不是很清晰所以就不再裁剪
        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue

        # generate positive examples and part faces
        for i in range(20):
            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)

            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)

            cropped_im = img[ny1 : ny2, nx1 : nx2, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                f1.write("12/positive/%s"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                f3.write("12/part/%s"%d_idx + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        box_idx += 1
        print("%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx))

f1.close()
f2.close()
f3.close()
