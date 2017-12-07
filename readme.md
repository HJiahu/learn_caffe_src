读caffe源码（CPU版）
==========================================
### 说明
*	include和src中分别保存着caffe的头文件和cpp文件，其中包含了我写的中文注释 :)。在VS中配置caffe的方式请参考[使用vs2013调试caffe源码][0]
*	我读caffe源码的过程记录在当前目录下的文件`learn_caffe.md`中
*	`proto_example`是protobuf的一个简单例子
*	`lenet_model`中是lenet的网络结构与一个已经训练好的模型，我在调试caffe源码时使用了lenet例程
*	`lenet_model`文件夹下的`digits_10000`中有10000张jpg格式的训练集与测试集，已经按3:1进行了分割并生成了对应的LMDB文件。此文件夹下亦有训练好的模型，分别迭代了500次与1000次。

### 其他
*	如果在windows下无法运行sh文件，请下载并安装[cmder][1]，然后按照cmder的说明文档将cmder注册到系统中【[参考][2]】。
*	[百度云分享][3]中有完整版的mnist数据集，与官方不同这个数据集中都是jpg格式的图片。命名规则：[标签]-[同标签计数]-[总图片计数].jpg
*	[百度云分享][4]中也有从cifar数据集中提取的图片，命名规则①cifar10：[标签]-[总图片计数].jpg ②cifar100：[coarse_label]-[fine_label]-[总图片计数].jpg





















[0]:http://www.cnblogs.com/jiahu-Blog/p/6423962.html
[1]:http://cmder.net/
[2]:https://segmentfault.com/a/1190000004408436
[3]:http://pan.baidu.com/s/1boR8seb
[4]:https://pan.baidu.com/s/1c2tXlss