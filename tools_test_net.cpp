#include "tools_config.h"
#ifdef TOOLS_TEST_NET_CPP

#include<caffe/caffe.hpp>
#include"my_configs.h"
using namespace caffe;

int main()
{
    std::string model_prototxt_path{R"(I:\learn_caffe\learn_caffe\caffe_src\models\rfcn_model\ResNet-50-deploy.prototxt)"};
    std::string trained_model{ R"(E:\CNN_Models\RFCN\rfcn_models_ResNet-50L\models\pre_trained_models\ResNet-50L\ResNet-50-model.caffemodel)" };
    Net<float> caffe_net (model_prototxt_path, caffe::TEST);
    caffe_net.CopyTrainedLayersFrom (trained_model);
    caffe_net.Forward();
#ifdef _MSC_VER
    system ("pause");
#endif // _MSC_VER
}
#endif // USE_THIS_FILE