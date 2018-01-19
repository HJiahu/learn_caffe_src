#include "tools_config.h"
#ifdef TOOLS_TEST_NET_CPP

#include<caffe/caffe.hpp>
#include"my_configs.h"
using namespace caffe;

int main()
{
    std::string model_prototxt_path{ R"(E:\CNN_Models\SSD\MobileNetSSD\MobileNetSSD_deploy000.prototxt)" };
    std::string trained_model{ R"(E:\CNN_Models\SSD\MobileNetSSD\MobileNetSSD_deploy.caffemodel)" };
    Net<float> caffe_net (model_prototxt_path, caffe::TEST);
    caffe_net.CopyTrainedLayersFrom (trained_model);
    caffe_net.Forward();
#ifdef _MSC_VER
    system ("pause");
#endif // _MSC_VER
}
#endif // USE_THIS_FILE