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
    
#ifdef _MSC_VER
    system ("pause");
#endif // _MSC_VER
}


#endif // USE_THIS_FILE