#coding=utf-8

import os.path as osp
import sys
import copy
import os
import numpy as np
import google.protobuf as pb

CAFFE_ROOT = r'D:\Programs\CaffeGPUBin'
sys.path.append(osp.join(CAFFE_ROOT,'python'))

import caffe
import caffe.proto.caffe_pb2 as cp

caffe.set_mode_cpu()
layer_type = ['Convolution', 'InnerProduct']
bnn_type = ['BatchNorm', 'Scale']
temp_file = './temp.prototxt'

class ConvertBnn:
    def __init__(self, model, weights, dest_model_dir, dest_weight_dir):
        self.net_model = caffe.Net(model, weights, caffe.TEST)
        # 下面这条语句只能获得网络的结构，网络中的训练参数还没有载入
        self.net_param = self.get_netparameter(model)
        self.dest_model = None
        self.dest_param = self.get_netparameter(model)
        self.remove_ele = []
        self.bnn_layer_location = []
        self.dest_dir = dest_model_dir
        self.dest_weight_dir = dest_weight_dir
        self.pre_process()

    def pre_process(self):
        net_param = self.dest_param
        # 获得所有层的结构与信息，例如输入与输出的维度、层的名称类型等
        layer_params = net_param.layer
        length = len(layer_params)
        i = 0
        while i < length:
            if layer_params[i].type in layer_type: # 当前文件只对两种类型的层（卷积和全连接）合并BN
                # 确定后两层分别是bn和scale
                if (i + 2 < length) and layer_params[i + 1].type == bnn_type[0] and \
                                layer_params[i + 2].type == bnn_type[1]:
                    params = layer_params[i].param
                    if len(params) == 0:
                        params.add()
                        params[0].lr_mult = 1
                        params[0].decay_mult = 1
                    if len(params) < 2:
                        params.add()
                        params[1].lr_mult = 2
                        params[1].decay_mult = 0
                        if layer_params[i].type == 'Convolution':
                            layer_params[i].convolution_param.bias_term = True
                            layer_params[i].convolution_param.bias_filler.type = 'constant'
                            layer_params[i].convolution_param.bias_filler.value = 0
                        elif layer_params[i].type == 'InnerProduct':
                            layer_params[i].inner_product_param.bias_term = True
                            layer_params[i].inner_product_param.bias_filler.type = 'constant'
                            layer_params[i].inner_product_param.bias_filler.value = 0
                    # 修改配置params
                    self.bnn_layer_location.extend([i])
                    self.remove_ele.extend([layer_params[i + 1], layer_params[i + 2]])
                    i = i + 3
                else:
                    i = i + 1
            elif layer_params[i].type == 'Scale' and layer_params[i - 1].type == bnn_type[0]:
                self.bnn_layer_location.extend([i])
                self.remove_ele.extend([layer_params[i - 1]])
                i += 1
            else:
                i += 1
        # for ele in remove_ele:
        #    layer_params.remove(ele)
        with open(temp_file, 'w') as f:
            f.write(str(net_param))
        print('asdf')
        self.dest_model = caffe.Net(temp_file, caffe.TEST)
        model_layers = self.net_model.layers
        for i, layer in enumerate(model_layers):
            if layer.type == 'Convolution' or layer.type == 'InnerProduct' or layer.type == 'Scale':
                self.dest_model.layers[i].blobs[0] = layer.blobs[0]
                if len(layer.blobs) > 1:
                    self.dest_model.layers[i].blobs[1] = layer.blobs[1]
        print('asdf end')

    # 这个函数主要读取网络的结构并返回
    def get_netparameter(self, model):
        with open(model) as f:
            net = cp.NetParameter()
            pb.text_format.Parse(f.read(), net)
            return net

    def convert(self):
        # layer param 需要修改 BIAS 参数 添加bias param 还有设置为 true
        out_params = self.dest_param.layer
        model_layers = self.net_model.layers
        out_model_layers = self.dest_model.layers

        length = len(self.bnn_layer_location)
        param_layers = self.dest_param.layer
        print(len(model_layers))
        print(len(self.net_param.layer))
        for layer in param_layers:
            print(layer.name)

        print('*******************************')

        for layer in self.net_model.layers:
            print(layer.type)

        param_layer_type_list = [layer.type for layer in param_layers]
        model_layer_type_list = [layer.type for layer in model_layers]

        i = j = 0
        dict_layer_id_param_to_model = {}
        while i < len(param_layer_type_list):
            if param_layer_type_list[i] == model_layer_type_list[j]:
                dict_layer_id_param_to_model[i] = j
                i = i + 1
                j = j + 1
            else:
                j = j + 1
        print(dict_layer_id_param_to_model)

        l = 0
        while l < length:
            i = self.bnn_layer_location[l]
            print(param_layers[i].name, param_layers[i].type)
            if param_layers[i].type in layer_type:
                # i = self.net_model.params.keys().index(param_layers[l].name);

                channels = self.net_model.params[param_layers[i].name][0].num
                # count = model_layers[i].blobs[0].count / channels
                scale = self.net_model.params[param_layers[i + 1].name][2].data[0]
                # print scale
                mean = self.net_model.params[param_layers[i + 1].name][0].data / scale
                # print mean
                std = np.sqrt(self.net_model.params[param_layers[i + 1].name][1].data / scale)
                a = self.net_model.params[param_layers[i + 2].name][0].data
                b = self.net_model.params[param_layers[i + 2].name][1].data
                for k in range(channels):
                    self.dest_model.params[param_layers[i].name][0].data[k] = \
                    self.net_model.params[param_layers[i].name][0].data[k] * a[k] / std[k]
                    self.dest_model.params[param_layers[i].name][1].data[k] = \
                    self.dest_model.params[param_layers[i].name][1].data[k] * a[k] / std[k] - a[k] * mean[k] / std[k] + \
                    b[k]
            elif param_layers[i].type == 'Scale':
                channels = self.net_model.params[param_layers[i - 1].name][0].num
                scale = self.net_model.params[param_layers[i - 1].name][2].data[0]
                mean = self.net_model.params[param_layers[i - 1].name][0].data / scale
                std = np.sqrt(self.net_model.params[param_layers[i - 1].name][1].data / scale)
                a = copy.deepcopy(self.net_model.params[param_layers[i].name][0].data)
                b = copy.deepcopy(self.net_model.params[param_layers[i].name][1].data)
                for k in range(channels):
                    self.dest_model.params[param_layers[i].name][0].data[k] = a[k] / std[k]
                    self.dest_model.params[param_layers[i].name][1].data[k] = a[k] / std[k] - a[k] * mean[k] / std[k] + \
                                                                              b[k]
            l += 1
        self.dest_model.save(self.dest_weight_dir)
        for ele in self.remove_ele:
            out_params.remove(ele)
        with open(self.dest_dir, 'w') as f:
            f.write(str(self.dest_param))
        os.remove(temp_file)
        print('MERGED SUCCEED!')

if __name__ == '__main__':
    cb = ConvertBnn('./res20_cifar_deploy.prototxt', './ResNet_20.caffemodel', './result.prototxt', './result.caffemodel')
    cb.convert()
