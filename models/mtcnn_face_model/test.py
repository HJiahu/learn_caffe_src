#coding:utf8
import sys, os
import time
sys.path.append('.')
sys.path.append(r'D:\Programs\CaffeGPUBin\python')
os.environ["GLOG_minloglevel"] = "1" # do not show msg when load caffe model
import tools_matrix as tools
import caffe
import cv2
import numpy as np


use_gpu = False
test_time = False

print("loading caffe model...")
deploy = '12net.prototxt'
caffemodel = '12net.caffemodel'
net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = '24net.prototxt'
caffemodel = '24net.caffemodel'
net_24 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = '48net.prototxt'
caffemodel = '48net.caffemodel'
net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)


# threshold中有三个变量，如：threshold = [0.6,0.6,0.7]，用于三个不同的网络
def detectFace(img_path,threshold):
    img = cv2.imread(img_path)
    caffe_img = (img.copy()-127.5)/128
    origin_h,origin_w,ch = caffe_img.shape
    scales = tools.calculateScales(img)
    out = []
    # 使用12net处理图像金字塔中的所有子图并将对应的结果保存在数组out中
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
        cls_prob = out[i]['prob1'][0][1] # 这个给出的是每个点的分类精度
        roi      = out[i]['conv4-2'][0]  # 这个给出了每个预测方框的回归参数（修正方框）
        # 网络的输入被resize后输出也会对应的发生一些变化
        out_h,out_w = cls_prob.shape
        out_side = max(out_h,out_w)
        # 参数分别为：网络的类别输出、网络的边框回归、网络类别输出的最大边长、缩放因子的倒数、原图的宽、原图高、阈值
        rectangle = tools.detect_face_12net(cls_prob,roi,out_side,1/scales[i],origin_w,origin_h,threshold[0])
        rectangles.extend(rectangle)
    # detect_face_12net中是有NMS的，不过其阈值为0.5
    rectangles = tools.NMS(rectangles,0.7,'iou')

    if len(rectangles)==0:
        return rectangles
    # 这里一次处理所有的子图，不使用循环，即直接设定caffe网络中的num
    net_24.blobs['data'].reshape(len(rectangles),3,24,24)
    crop_number = 0
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(24,24))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_24.blobs['data'].data[crop_number] = scale_img
        crop_number += 1
    out = net_24.forward()
    cls_prob = out['prob1']
    roi_prob = out['conv5-2']
    rectangles = tools.filter_face_24net(cls_prob,roi_prob,rectangles,origin_w,origin_h,threshold[1])
    
    if len(rectangles)==0:
        return rectangles
    # 同样一次处理完所有子图，避免使用循环
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

    return rectangles # 前4个元素为bbx的坐标、下一个元素是置信值、再下2*5 = 10个元素为5个landmark




# use above function to detect face and show detecting result

threshold = [0.6,0.6,0.7]
imgpath = r"./TVHID1130_4.jpg"

rectangles = detectFace(imgpath, threshold)
img = cv2.imread(imgpath)
draw = img.copy()
for rectangle in rectangles:
    cv2.putText(draw,str(rectangle[4]),(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
    for i in range(5,15,2):
        cv2.circle(draw,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
cv2.imshow("test",draw)
cv2.waitKey()
# cv2.imwrite('TVHID1130_4_result.jpg',draw)

# added by HJiahu@20180127
if test_time:
    print('test time for detectFace...')
    num_test = 1000
    if use_gpu:
        print("use GPU.")
    else:
        print("use CPU.")
    start = time.time()
    detectFace(imgpath, threshold)
    end = time.time()
    print(end - start)
