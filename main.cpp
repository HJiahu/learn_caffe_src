//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <caffe/caffe.hpp>
//#include <memory>
//#include <string>
//using namespace caffe;
//using namespace std;
//
////如果num<0则随机生成0-9的图，返回的Mat元素是浮点数
////白色的背景黑色的数值，如果show_img为true则显示生成的图片
//std::pair<int, std::shared_ptr<float> > generate_test_img (int num = -1, bool show_img = false);
////在控制台显示caffe的预测结果
//void show_predict (const float *output);
//
//
//int main (int argc, char* argv[])
//{
//    ::google::InitGoogleLogging (argv[0]); //不在终端显示网络的载入过程
//    cout << "init ......" << endl;
//    //网络结构文件
//    const string lenet_prototxt_path (R"(I:\learn_caffe\learn_caffe\caffe_src\lenet_model\lenet.prototxt)");
//    //训练好的网络模型，这个模型的准确度不高，6和7都会被误检
//    const string lenet_model_path (R"(I:\learn_caffe\learn_caffe\caffe_src\lenet_model\lenet_iter_20000.caffemodel)");
//    typedef float type;
//    //生成用于测试的图片，默认图像的大小为28*28（灰度图）
//    auto data = generate_test_img (7, true);
//    //初始化caffe
//    //set cpu running software
//    Caffe::set_mode (Caffe::CPU);
//    //load net file	, caffe::TEST 用于测试时使用
//    Net<type> lenet (lenet_prototxt_path, caffe::TEST);
//    //load net train file caffemodel
//    //因为caffe使用protobuf实现数据的持久化所以网络结构文件与model文一一对应
//    lenet.CopyTrainedLayersFrom (lenet_model_path);
//    Blob<type> *input_ptr = lenet.input_blobs() [0];
//    input_ptr->Reshape (1, 1, 28, 28);
//    Blob<type> *output_ptr = lenet.output_blobs() [0];
//    //output_ptr->Reshape (1, 10, 1, 1);
//    //copy data from <ary> to <input_ptr>
//    input_ptr->set_cpu_data (data.second.get());
//    cout << "test num is :" << data.first << endl;
//    //begin once predict
//    lenet.Forward();
//    cout << "finshed." << endl;
//    show_predict (output_ptr->cpu_data());
//#ifdef _MSC_VER
//    system ("pause");
//#endif // _MSC_VER
//    return 0;
//}
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//std::pair<int, std::shared_ptr<float> > generate_test_img (int num, bool show_img)
//{
//    //生成用于测试的数据
//    char test_num[2] = { 0 };
//    
//    //如果num在1-9之内，则生成对应的值，否则随机生成1-9的数
//    if (num < 0 || num > 9)
//    {
//        srand (time (nullptr));
//        num = rand() % 10;
//        test_num[0] = num + '0';
//    }
//    
//    else
//    {
//        test_num[0] = static_cast<char> ('0' + num);
//    }
//    
//    std::shared_ptr<float> data_ptr (new float[28 * 28]);
//    //在28*28的图片颜色为RGB(255,255,255)背景上写RGB(0,0,0)数字.
//    cv::Mat gray (28, 28, CV_8UC1, cv::Scalar (255));
//    //如果把参数thickness改为2（即使得字体的笔画变粗）则预测结果误差比较大，6，9，都会被误检
//    cv::putText (gray, (char*) &test_num, cv::Point (4, 22), 5, 1.4, cv::Scalar (0), 1);
//    
//    //将图像的数值从uchar[0,255]转换成float[0.0f,1.0f],的数, 且颜色取相反的 .
//    for (int i = 0; i < 28 * 28; i++)
//    {
//        // f_val =(255-uchar_val)/255.0f
//        data_ptr.get() [i] = static_cast<float> (gray.data[i] ^ 0xFF) * 0.00390625;
//    }
//    
//    if (show_img)
//    {
//        cv::imshow ("test_num", gray);
//        cv::waitKey (1);
//    }
//    
//    return make_pair (num, data_ptr);
//}
//
//
//void show_predict (const float *output)
//{
//    cout << "\n\nPredicting Outcomes: " << endl;
//    // get the maximum index
//    int index = 0;
//    
//    for (int i = 1; i < 10; i++)
//    {
//        if (output[index] < output[i])
//        {
//            index = i;
//        }
//    }
//    
//    // 打印这次预测[0,9]的每一个置信度
//    for (int i = 0; i < 10; i++)
//    {
//        cout << i << "\t" << output[i] << endl;
//    }
//    
//    // 展示最后的预测结果
//    cout << "res:\t" << index << "\t" << output[index] << endl;
//}