
#include "tools_config.h"
#ifdef TOOLS_FORWORD_CIFAR10_CPP
/*
	这个文件不再使用，若想了解caffe对cifar10分类过程可以看文件 caffe_classifier.cpp
*/

#include <string>
#include <opencv2/opencv.hpp>
#include "my_configs.h"
#include "caffe_classifier.h"

using namespace std;

/* class in cifar10
		airplane   //0
		automobile //1
		bird
		cat
		deer
		dog
		frog
		horse
		ship
		truck
*/

int main()
{
    Path cifar10_trained_path{ model_root_path_g / "cifar10_model/cifar10_itr60000" };
    Path cifar10_test_img_dir{ model_root_path_g / "cifar10_model/sample_imgs" };
    assert (cifar10_test_img_dir.exists() && cifar10_trained_path.exists());
    Classifier classifer (cifar10_trained_path.string());
    cv::Mat img = cv::imread ( (cifar10_test_img_dir / "2_3454.jpg").string());
    auto results = classifer.Classify (img, 10);
    
    for (auto i : results)
    {
        cout << i << endl;
    }
    
#ifdef _MSC_VER
    system ("pause");
#endif // _MSC_VER
}

#endif // USE_THIS_TILE

