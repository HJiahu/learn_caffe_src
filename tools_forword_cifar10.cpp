#define USE_THIS_TILE
#ifdef USE_THIS_TILE

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
    Path cifar10_trained_path{ root_path_g / "cifar10_model/cifar10_itr60000" };
    Path cifar10_test_img_dir{ root_path_g / "cifar10_model/sample_imgs" };
    assert (cifar10_test_img_dir.exists() && cifar10_trained_path.exists());
    cv::Mat img = cv::imread ( (cifar10_test_img_dir / "0_3295.jpg").string());
    
#ifdef _MSC_VER
    system ("pause");
#endif // _MSC_VER
}

#endif // USE_THIS_TILE

