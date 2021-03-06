
#include "tools_config.h"
#ifdef TOOLS_TEST_CAFFE_CLASSIFIER_CPP

#include <string>
#include <opencv2/opencv.hpp>
#include "my_configs.h"
#include "caffe_classifier.h"

using namespace std;

int main()
{
    Path cifar10_trained_path{ model_root_path_g / "cifar10_model/cifar10_itr60000" };
    Path cifar10_test_img_dir{ model_root_path_g / "cifar10_model/sample_imgs"};
    assert (cifar10_test_img_dir.exists() && cifar10_trained_path.exists());
    Classifier classifer (cifar10_trained_path.string());
    cv::Mat img = cv::imread ( (cifar10_test_img_dir / "0_3295.jpg").string());
    auto results = classifer.Classify (img);
    
    for (auto x : results)
    {
        cout << x << endl;
    }
    
#ifdef _MSC_VER
    system ("pause");
#endif // _MSC_VER
}

#endif // USE_THIS_TILE

