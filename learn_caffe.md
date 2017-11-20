caffe源码阅读杂记
=====================================================
### 准备
*	一些参考网页
	*	[Neural Networks and Deep Learning][7]
	*	[TUTORIAL ON DEEP LEARNING FOR VISION][6]
	*	[Deep Learning Tutorial][10]
	*	[知乎-深度学习caffe的代码怎么读][9]
	*	[Caffe源码解析][15]
	*	[caffe源码结构][16]
	*	[官方代码结构doxygen][17]
*	以C++源码形式配置debug&CPU版的caffe，便于阅读源码与单步调试【[参考][8]】
*	参考官方的[文档][5]，先了解某个模块的作用
*	为了简化学习，先不考虑任何与GPU相关的问题
### 我的阅读过程
#### Caffe中几个主要的类
*	Solver: 网络求解类，一般使用的方法是梯度下降法
*	Net: 描述网络的结构
*	Layer: 数据处理方式的封装，不同layer对data进行不同的运算，如卷积、池化、全连接等
*	Blob: 数据的保存与传递都使用的类

#### 读caffe.proto
*	caffe.proto中定义了很多数据结构（这些结构都会转化为对应C++类），具体内容参考caffe.proto中的定义，文件中描述的很详细
##### [google/protobuf][12] 简介
*	protobuf是一个数据[持久化与序列化][4]的工具，具体教程参考[1][1]、[2][2]、[官方C++版教程][3]（[C++国免翻墙制版][14]）。
*	**数据存储和交换**（包括通过网络交换数据） 是 Protobuf 最有效的应用领域。caffe中主要用protobuf实现网络模型的结构定义、存储和读取（网络结构映射到内存中就是对应对象之间的结构，对象初始化时需要参数。caffe使用protobuf定义内存中对象的结构并保存相应的数据，protobuf将管理这些参数在文件与内存中的映射关系）。
*	使用流程如下：
	*	首先我们需要编写一个 proto 文件，定义我们程序中需要处理的结构化数据，在 protobuf 的术语中，结构化数据被称为 Message；
	*	写好 proto 文件之后就可以用 Protobuf 编译器将该文件编译成目标语言了，对于C++而言就是生成一个hpp和cpp文件；
	*	在生成的头文件中，有一个 C++ 类，使用这个类可以对消息进行操作。诸如对消息的成员进行赋值，将消息序列化、反序列化等等。
*	[proto格式说明](#proto)(见本文下方)

### blob、layer、net、solver的大致结构
#### blob.hpp
*	blob类用来保存学习到的参数以及网络传输过程中产生的数据，bolb是caffe在内存中数据交换的核心
*	一层一层往下分BLOB的基本元素分别为：图像->通道->行->列（这里已经是最小的元素了，其类型就是DType，常见的如float、double）。`( (n * channels() + c) * height() + h) * width() + w;`表示的是元素(n,c,h,w)在blob中的索引。
	*	`n*channels()+c `表示的是当前元素所处的通道的索引
	*	`((n * channels() + c) * height() + h)`  当前元素所处的行所在的索引
	*	最后求的就是当前元素的索引了
*	以RGB图像为例描述BLOB在内存中的格式，假设当前BLOB中有N张图，图的高为H，宽为W像素：
	
		|------------------------img0-----------------------|...img1~imgN-1...|
		|<------channel R-------> <-channel G-><-channel B->|..................
        |<row0--row2-...row(H-1)>...........................|..................
        |对应的R、或G、或B的值......
*	blob类的成员如下（部分）
	*	成员变量

			//指向存储数据的内存或显存块，一般是图像数据、权值或偏置值数据
			shared_ptr<SyncedMemory> data_;
			shared_ptr<SyncedMemory> diff_;//保存反向传递时的梯度信息
	*	成员函数

			void Reshape (const vector<int>& shape);
			//用于调整下标。C++数组不支持负数下标，当前函数的功能就是将负数转化为对应的正数下标
			inline int CanonicalAxisIndex (int axis_index) const;
			//blob中还有一些对数据进行简单计算的方法

#### SyncedMemory
*	SyncedMemory类主要用于数据在CPU和GPU之间的交换，本文不深入探索当前类。
*	SyncedMemory中的部分成员

		const void* cpu_data();
		void set_cpu_data (void* data);
		const void* gpu_data();
		void set_gpu_data (void* data);
		void* mutable_cpu_data();//与上面的cpu_data和gpu_data相对应，只不过内容可改
		void* mutable_gpu_data();

#### layer.hpp
*	layer是caffe中网络的基础，是基本的计算单元。卷积、池化、內积、sigmoid等神经元运算都在layer中完成。而且caffe中的layer不但实现前向运算，同时也可以实现反向运算，即检测与训练运算都包含在同一个层中。
*	caffe中所有的层都直接或间接继承于layer这个类
*	每一层的layer从底层连接获得输入数据，计算之后通过顶层连接输出计算结果。
*	每一层layer都必须实现3个关键函数：setup、forward、backward
	*	Setup: 在初始化模型的的时候初始化层的参数和连接
	*	Forward: 通过底层（bottom）给出的数据计算结果并传递给顶层（top）
	*	Backward: 通过顶层(top)给出的数据计算梯度并传递给底层(bottom)。 A layer with parameters computes the gradient w.r.t. to its parameters and stores it internally.
*	forward和backward函数有对应的CPU与GPU版，如果没有定义GPU版函数，则退化使用CPU函数
*	layer类中的成员有
	*	数据成员

			LayerParameter layer_param_;//当前层的参数，
			Phase phase_;//用于测试还是训练
			vector<shared_ptr<Blob<Dtype> > > blobs_;//指向需要训练的参数
			/** Vector indicating whether to compute the diff of each param blob. */
			vector<bool> param_propagate_down_;
			
			/** The vector that indicates whether each top blob has a non-zero weight in the objective function. */
			vector<Dtype> loss_;//损失函数值
	*	函数成员

            virtual void LayerSetUp (...);
			inline Dtype Forward (...);
            inline void Backward (...);
			//Returns the vector of learnable parameter blobs.
			vector<shared_ptr<Blob<Dtype> > >& blobs();
			//Writes the layer parameter to a protocol buffer
			virtual void ToProto (...);
			


##### LayerParameter
*	LayerParameter中定义了如下参数（这里只有部分，具体可参考caffe.proto文件）

		optional string name; // the layer name
		optional string type; // the layer type
		repeated string bottom; // the name of each bottom blob
		repeated string top; // the name of each top blob
		// The train / test phase for computation. 有些层只用于TEST
		optional Phase phase;
		// The blobs containing the numeric parameters of the layer.
		//BlobProto也是一种结构，同样定义在caffe.proto中，这个和blob.hpp中定义的结构不同
		//在内存中使用后者的结构，在硬盘中使用前者的结构
		repeated BlobProto blobs;
		// Rules controlling whether and when a layer is included in the network
		repeated NetStateRule include;
		repeated NetStateRule exclude;
		optional LossParameter loss_param;// Parameters shared by loss layers.
		// Layer type-specific parameters.
		optional AccuracyParameter accuracy_param;
		optional ArgMaxParameter argmax_param;
		optional BatchNormParameter batch_norm_param;
		...

##### input_layer（所有网络的第一层，以代码中的lenet为例）














### <span id="proto">proto详解</span>
*	proto [格式示例与解释][13]

		syntax = "proto2";//指明所使用的协议版本号
		
		//名字空间，自动生成代码时所有代码均在这个名字空间内，在C++中会像这样：namespace tutorial {...}
		package tutorial;
		
		//结构化数据被称为 Message
		//在hpp文件中会生成如下类和结构：
		//enum Person_PhoneType;
		//class Person_PhoneNumber : public ::google::protobuf::Message;
		//class Person :public ::google::protobuf::Message;
		//class AddressBook : public ::google::protobuf::Message;
		message Person {
		  
		  // 字段（类成员）定义格式：限定修饰符① | 数据类型② | 字段名称③ | = | 字段编码值④ | [字段默认值⑤]
          // 每一个变量前都需要使用下面三个词进行修饰：required、optional、repeated
		  // required表示是一个必须字段，必须相对于发送方（对应于数据的序列化方），在发送消息之前必须设置该字段的值，
		  // 对于接收方（对应于数据的反序列化方），必须能够识别该字段的意思。
		  // 发送之前没有设置required字段或者无法识别required字段都会引发编解码异常，导致消息被丢弃。
		  required string name = 1;

		  // 这里的数值"=1、=2..."是编码值，编码值用于通信双方互相识别对方的字段
          // 其中 1~15的编码时间和空间效率都是最高的，编码值越大，其编码的时间和空间效率就越低
		  // 1900~2000编码值为Google protobuf 系统内部保留值，建议不要在自己的项目中使用
		  required int32 id = 2;

		  // optional表示是一个可选字段，可选对于发送方，在发送消息时，可以有选择性的设置或者不设置该字段的值
		  // 对于接收方，如果能够识别可选字段就进行相应的处理，如果无法识别，则忽略该字段，消息中的其它字段正常处理
		  // 项目投入运营以后涉及到版本升级时的新增消息字段全部使用optional或者repeated，尽量不使用required
		  // 如果使用了required，需要全网统一升级，如果使用optional或者repeated可以平滑升级
		  optional string email = 3;
		
		  enum PhoneType {
		    MOBILE = 0;
		    HOME = 1;
		    WORK = 2;
		  }
		
		  message PhoneNumber {
		    required string number = 1;
		    optional PhoneType type = 2 [default = HOME];
		  }
		
		  // 表示该字段可以包含0~N个元素。其特性和optional一样，但是每一次可以包含多个值
		  // 可以看作是在传递一个数组的值
		  repeated PhoneNumber phones = 4;
		}
		
		message AddressBook {
		  repeated Person people = 1;
		}

*	The Protocol Buffer API（通过 Protobuf 编译器编译上面的proto文件所获得的）

		// name
		inline bool has_name() const;//判断信息中是否包含当前元素
		inline void clear_name();
		inline const ::std::string& name() const;//下面是getter和setter
		inline void set_name(const ::std::string& value);
		inline void set_name(const char* value);
		inline ::std::string* mutable_name();
		
		...

*	Writing A Message & Reading A Message（使用上面API的示例）
	
		//create Message 
		cout << "Enter person ID number: ";
		int id;
		cin >> id;
		person->set_id(id);
		cin.ignore(256, '\n');
		
		cout << "Enter name: ";
		getline(cin, *person->mutable_name());
		...

		//read massage
		cout << "Person ID: " << person.id() << endl;
		cout << "  Name: " << person.name() << endl;
		if (person.has_email()) {
			cout << "  E-mail address: " << person.email() << endl;
		}

















[1]:https://www.ibm.com/developerworks/cn/linux/l-cn-gpb/
[2]:http://blog.csdn.net/majianfei1023/article/details/45112415
[3]:https://developers.google.com/protocol-buffers/docs/cpptutorial
[4]:https://www.zhihu.com/question/20706270
[5]:http://caffe.berkeleyvision.org/tutorial/
[6]:https://sites.google.com/site/deeplearningcvpr2014/
[7]:http://neuralnetworksanddeeplearning.com/index.html
[8]:http://www.cnblogs.com/jiahu-Blog/p/6423962.html
[9]:https://www.zhihu.com/question/27982282
[10]:http://deeplearning.net/tutorial/deeplearning.pdf
[11]:http://alanse7en.github.io/caffedai-ma-jie-xi-1/
[12]:http://www.lxway.com/905144802.htm
[13]:http://blog.csdn.net/guyue35/article/details/51181845
[14]:http://blog.csdn.net/sunlylorn/article/details/7598295
[15]:http://www.cnblogs.com/louyihang-loves-baiyan/p/5149628.html
[16]:https://www.leiphone.com/news/201612/oZUj5d437bpSl5wc.html
[17]:http://caffe.berkeleyvision.org/doxygen/