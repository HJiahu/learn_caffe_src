<<<<<<< HEAD
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
=======
//#define USE_THIS_FILE
#ifdef USE_THIS_FILE
#include "my_configs.h"
#include "caffe_classifier.h"

int main()
{
    std::string model_dir{ (root_path_g / "shufflenet_head_48x48_color").string() };
    std::string head_img_path{ (root_path_g / "shufflenet_head_48x48_color/Mikasa.jpg").string() };
    std::string nohead_img_path{ (root_path_g / "shufflenet_head_48x48_color/nohead_4749.jpg").string() };
    Classifier classifier (model_dir);
    cv::Mat head_img{cv::imread (head_img_path) };
    cv::Mat nohead_img{ cv::imread (nohead_img_path) };
    auto results = classifier.Classify (nohead_img);
    
    for (auto x : results)
    {
        std::cout << x << std::endl;
    }
    
>>>>>>> 0d0d59daf845ecd263ed90aebb61ba59a26ddaec
#ifdef _MSC_VER
    system ("pause");
#endif // _MSC_VER
}

#endif // USE_THIS_FILE