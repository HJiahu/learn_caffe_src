//#define USE_THIS_FILE
#ifdef USE_THIS_FILE
#include<caffe/caffe.hpp>

using namespace caffe;

int main()
{
    std::string model_prototxt_path{R"(E:\libs\caffe-yolo\examples\yolo\gnet_deploy.prototxt)"};
    std::string trained_model{ R"(I:\BaiduNetdiskDownload\gnet_yolo_iter_32000.caffemodel)" };
    Net<float> caffe_net (model_prototxt_path, caffe::TEST);
    caffe_net.CopyTrainedLayersFrom (trained_model);
	auto restult = caffe_net.Forward();

#ifdef _MSC_VER
    system ("pause");
#endif // _MSC_VER
}


#endif // USE_THIS_FILE