#coding:utf8
import sys
from timeit import timeit
sys.path.append('.')
sys.path.append(r'D:\Programs\caffe_deps\caffe_VS2015x64RGPU\python')
import tools_matrix as tools
import caffe
import cv2
import numpy as np

use_gpu = False
test_time = False

deploy = '12net.prototxt'
caffemodel = '12net.caffemodel'
net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = '24net.prototxt'
caffemodel = '24net.caffemodel'
net_24 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = '48net.prototxt'
caffemodel = '48net.caffemodel'
net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)



def detectFace(img_path,threshold):
    img = cv2.imread(img_path)
    caffe_img = (img.copy()-127.5)/128
    origin_h,origin_w,ch = caffe_img.shape
    scales = tools.calculateScales(img)
    out = []
    for scale in scales:
        hs = int(origin_h*scale)
        ws = int(origin_w*scale)
        scale_img = cv2.resize(caffe_img,(ws,hs))
        # opencv以BGR顺序保存图片，这里转化为RGB
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_12.blobs['data'].reshape(1,3,ws,hs) #全卷积网络的特点
        net_12.blobs['data'].data[...] = scale_img
        if use_gpu:
            caffe.set_device(0)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        # 因为12net是全卷积网络，所以输出feature map的尺度与输入有关
        out_ = net_12.forward()
        out.append(out_)

    print("12net out len: ",len(out))

    image_num = len(scales)
    rectangles = []
    for i in range(image_num):    
        cls_prob = out[i]['prob1'][0][1]
        roi      = out[i]['conv4-2'][0]
        out_h,out_w = cls_prob.shape
        out_side = max(out_h,out_w)
        rectangle = tools.detect_face_12net(cls_prob,roi,out_side,1/scales[i],origin_w,origin_h,threshold[0])
        rectangles.extend(rectangle)
    rectangles = tools.NMS(rectangles,0.7,'iou')

    if len(rectangles)==0:
        return rectangles
    net_24.blobs['data'].reshape(len(rectangles),3,24,24)
    crop_number = 0
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(24,24))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_24.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_24.forward()
    cls_prob = out['prob1']
    roi_prob = out['conv5-2']
    rectangles = tools.filter_face_24net(cls_prob,roi_prob,rectangles,origin_w,origin_h,threshold[1])
    
    if len(rectangles)==0:
        return rectangles
    net_48.blobs['data'].reshape(len(rectangles),3,48,48)
    crop_number = 0
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(48,48))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_48.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_48.forward()
    cls_prob = out['prob1']
    roi_prob = out['conv6-2']
    pts_prob = out['conv6-3']
    rectangles = tools.filter_face_48net(cls_prob,roi_prob,pts_prob,rectangles,origin_w,origin_h,threshold[2])

    return rectangles

threshold = [0.6,0.6,0.7]
imgpath = r"./TVHID1130_4.jpg"

def test_find_time():
    return detectFace(imgpath, threshold)


rectangles = detectFace(imgpath,threshold)
img = cv2.imread(imgpath)
draw = img.copy()
for rectangle in rectangles:
    cv2.putText(draw,str(rectangle[4]),(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
    for i in range(5,15,2):
        cv2.circle(draw,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
cv2.imshow("test",draw)
cv2.waitKey()
cv2.imwrite('TVHID1130_4_result.jpg',draw)
if test_time:
    print('test time for detectFace...')
    num_test = 1000
    if use_gpu:
        print("use GPU.")
    else:
        print("use CPU.")
    t = timeit('test_find_time()','from __main__ import test_find_time',number = num_test)
    print(t/num_test)

