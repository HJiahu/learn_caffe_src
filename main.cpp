//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <caffe/caffe.hpp>
//#include <memory>
//#include <string>
//using namespace caffe;
//using namespace std;
//
////���num<0���������0-9��ͼ�����ص�MatԪ���Ǹ�����
////��ɫ�ı�����ɫ����ֵ�����show_imgΪtrue����ʾ���ɵ�ͼƬ
//std::pair<int, std::shared_ptr<float> > generate_test_img (int num = -1, bool show_img = false);
////�ڿ���̨��ʾcaffe��Ԥ����
//void show_predict (const float *output);
//
//
//int main (int argc, char* argv[])
//{
//    ::google::InitGoogleLogging (argv[0]); //�����ն���ʾ������������
//    cout << "init ......" << endl;
//    //����ṹ�ļ�
//    const string lenet_prototxt_path (R"(I:\learn_caffe\learn_caffe\caffe_src\lenet_model\lenet.prototxt)");
//    //ѵ���õ�����ģ�ͣ����ģ�͵�׼ȷ�Ȳ��ߣ�6��7���ᱻ���
//    const string lenet_model_path (R"(I:\learn_caffe\learn_caffe\caffe_src\lenet_model\lenet_iter_20000.caffemodel)");
//    typedef float type;
//    //�������ڲ��Ե�ͼƬ��Ĭ��ͼ��Ĵ�СΪ28*28���Ҷ�ͼ��
//    auto data = generate_test_img (7, true);
//    //��ʼ��caffe
//    //set cpu running software
//    Caffe::set_mode (Caffe::CPU);
//    //load net file	, caffe::TEST ���ڲ���ʱʹ��
//    Net<type> lenet (lenet_prototxt_path, caffe::TEST);
//    //load net train file caffemodel
//    //��Ϊcaffeʹ��protobufʵ�����ݵĳ־û���������ṹ�ļ���model��һһ��Ӧ
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
//    //�������ڲ��Ե�����
//    char test_num[2] = { 0 };
//    
//    //���num��1-9֮�ڣ������ɶ�Ӧ��ֵ�������������1-9����
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
//    //��28*28��ͼƬ��ɫΪRGB(255,255,255)������дRGB(0,0,0)����.
//    cv::Mat gray (28, 28, CV_8UC1, cv::Scalar (255));
//    //����Ѳ���thickness��Ϊ2����ʹ������ıʻ���֣���Ԥ�������Ƚϴ�6��9�����ᱻ���
//    cv::putText (gray, (char*) &test_num, cv::Point (4, 22), 5, 1.4, cv::Scalar (0), 1);
//    
//    //��ͼ�����ֵ��uchar[0,255]ת����float[0.0f,1.0f],����, ����ɫȡ�෴�� .
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
//    // ��ӡ���Ԥ��[0,9]��ÿһ�����Ŷ�
//    for (int i = 0; i < 10; i++)
//    {
//        cout << i << "\t" << output[i] << endl;
//    }
//    
//    // չʾ����Ԥ����
//    cout << "res:\t" << index << "\t" << output[index] << endl;
//}