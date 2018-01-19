读caffe源码（CPU版）
==========================================
### 说明

caffe的源码都在`read_caffe_vs2015`文件夹中  

#### 基本配置
*	当前目录下的`read_caffe_vs2015`中保存着对应的VS2015项目文件，为了正常使用项目，需要做下面的准备：
	> 在C盘中创建文件夹`caffe_deps`  
	> 解压下载的caffe[第三方依赖库][5]，把其中的libraries放入上面建立的文件夹   
	> 默认情况下我会在C盘执行：   
	> `git clone https://github.com/HJiahu/learn_caffe_src.git  read_caffe_src`  
	> 在编译时常会提示boost的lib找不到，其实对应的目录下有对应的文件，只要改个名字即可【[参考][0]】   
	> 调试程序前先依据具体情况修改`my_configs.h`中的根目录变量`root_path_g`、`model_root_path_g`  
	> 通过注释`tools_config.h`中的中的宏，选择需要编译与执行的文件  

#### 文件说明
*	`proto_example`是protobuf的一个简单例子
*	`models`文件夹下有几种不同网络的模型文件，如lenet、cifar10、squeezenet、shufflenet等
	*	`models/lenet_model`中是lenet的网络结构与一个已经训练好的模型，我在调试caffe源码时使用了lenet例程
	*	`models/lenet_model` 文件夹下的 `digits_10000` 中有10000张jpg格式的训练集与测试集，已经按3:1进行了分割并生成了对应的LMDB文件。此文件夹下亦有训练好的模型，分别迭代了500次与1000次。

### 其他
*	如果在windows下无法运行sh文件，请下载并安装[cmder][1]，然后按照cmder的说明文档将cmder注册到系统中【[参考][2]】。
*	[百度云分享][3]中有完整版的mnist数据集，与官方不同这个数据集中都是jpg格式的图片。命名规则：[标签]-[同标签计数]-[总图片计数].jpg
*	[百度云分享][4]中也有从cifar数据集中提取的图片，命名规则①cifar10：[标签]-[总图片计数].jpg ②cifar100：[coarse_label]-[fine_label]-[总图片计数].jpg
*	MACROs used in vs2015 debug/release x64 
	*	debug

			_DEBUG
			_CONSOLE
			CPU_ONLY
			_SCL_SECURE_NO_WARNINGS
			_CRT_SECURE_NO_DEPRECATE
			_CRT_NONSTDC_NO_DEPRECATE
			USE_LMDB
			USE_OPENCV

	*	release

			NDEBUG
			_CONSOLE
			CPU_ONLY
			_SCL_SECURE_NO_WARNINGS
			_CRT_SECURE_NO_DEPRECATE
			_CRT_NONSTDC_NO_DEPRECATE
			USE_LMDB
			USE_OPENCV

*	3rdparty libs use in vs2015 debug/release x64
	
	> 说明：如果不想自己编译opencv，可以使用官方提供的opencv310，此时请替换opencv_world为官方提供的opencv libs
	*	debug

			opencv_world310d.lib
			boost_chrono-vc140-mt-gd-1_61.lib
			boost_date_time-vc140-mt-gd-1_61.lib
			boost_filesystem-vc140-mt-gd-1_61.lib
			boost_system-vc140-mt-gd-1_61.lib
			boost_thread-vc140-mt-gd-1_61.lib
			boost_timer-vc140-mt-gd-1_61.lib
			libopenblas.dll.a
			caffehdf5_D.lib
			caffehdf5_cpp_D.lib
			caffehdf5_hl_D.lib
			caffehdf5_hl_cpp_D.lib
			caffezlibd.lib
			caffezlibstaticd.lib
			gflagsd.lib
			glogd.lib
			leveldbd.lib
			libboost_chrono-vc140-mt-gd-1_61.lib
			libboost_system-vc140-mt-gd-1_61.lib
			libboost_timer-vc140-mt-gd-1_61.lib
			libcaffehdf5_D.lib
			libcaffehdf5_cpp_D.lib
			libcaffehdf5_hl_D.lib
			libcaffehdf5_hl_cpp_D.lib
			libprotobufd.lib
			libprotocd.lib
			lmdbd.lib
			snappy_staticd.lib
			snappyd.lib
			ntdll.lib
			
			# 官方提供的opencv，如果使用自己编译的opencv_world310d.lib，请不要添加下面内容
			opencv_calib3d310d.lib
			opencv_core310d.lib
			opencv_features2d310d.lib
			opencv_flann310d.lib
			opencv_highgui310d.lib
			opencv_imgcodecs310d.lib
			opencv_imgproc310d.lib
			opencv_ml310d.lib
			opencv_objdetect310d.lib
			opencv_photo310d.lib
			opencv_shape310d.lib
			opencv_stitching310d.lib
			opencv_superres310d.lib
			opencv_ts310d.lib
			opencv_video310d.lib
			opencv_videoio310d.lib
			opencv_videostab310d.lib

	*	release	

			opencv_world310.lib
			boost_chrono-vc140-mt-1_61.lib
			boost_date_time-vc140-mt-1_61.lib
			boost_filesystem-vc140-mt-1_61.lib
			boost_system-vc140-mt-1_61.lib
			boost_timer-vc140-mt-1_61.lib
			libopenblas.dll.a
			caffehdf5.lib
			caffehdf5_cpp.lib
			caffehdf5_hl.lib
			caffehdf5_hl_cpp.lib
			caffezlib.lib
			caffezlibstatic.lib
			gflags.lib
			glog.lib
			leveldb.lib
			libboost_chrono-vc140-mt-1_61.lib
			libboost_system-vc140-mt-1_61.lib
			libboost_timer-vc140-mt-1_61.lib
			libcaffehdf5.lib
			libcaffehdf5_cpp.lib
			libcaffehdf5_hl.lib
			libcaffehdf5_hl_cpp.lib
			libprotobuf.lib
			libprotoc.lib
			lmdb.lib
			snappy_static.lib
			snappy.lib
			ntdll.lib

			# 官方提供的opencv，如果使用自己编译的opencv_world310.lib，请不要添加下面内容
			opencv_calib3d310.lib
			opencv_core310.lib
			opencv_features2d310.lib
			opencv_flann310.lib
			opencv_highgui310.lib
			opencv_imgcodecs310.lib
			opencv_imgproc310.lib
			opencv_ml310.lib
			opencv_objdetect310.lib
			opencv_photo310.lib
			opencv_shape310.lib
			opencv_stitching310.lib
			opencv_superres310.lib
			opencv_ts310.lib
			opencv_video310.lib
			opencv_videoio310.lib
			opencv_videostab310.lib
	

















[0]:http://www.cnblogs.com/jiahu-Blog/p/6423962.html
[1]:http://cmder.net/
[2]:https://segmentfault.com/a/1190000004408436
[3]:http://pan.baidu.com/s/1boR8seb
[4]:https://pan.baidu.com/s/1c2tXlss
[5]:https://pan.baidu.com/s/1ht0t3Li