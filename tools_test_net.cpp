#include "tools_config.h"
#ifdef TOOLS_TEST_NET_CPP
#include<caffe/caffe.hpp>
#include"my_configs.h"
using namespace caffe;

int main()
{
    std::string model_prototxt_path{R"(E:\libs\SqueezeNet\SqueezeNet_v1.1\deploy.prototxt)"};
    std::string trained_model{ R"(E:\libs\SqueezeNet\SqueezeNet_v1.1\squeezenet_v1.1.caffemodel)" };
    Net<float> caffe_net (model_prototxt_path, caffe::TEST);
    caffe_net.CopyTrainedLayersFrom (trained_model);
    caffe_net.Forward();
#ifdef _MSC_VER
    system ("pause");
#endif // _MSC_VER
}

#endif // USE_THIS_FILE