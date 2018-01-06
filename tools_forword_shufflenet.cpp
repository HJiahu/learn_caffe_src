#include "tools_config.h"
#ifdef TOOLS_FORWORD_SHUFFLENET_CPP
#include "caffe_classifier.h"
#include "my_configs.h"
#include <string>

using namespace std;

int main()
{
    TinyPath::path shufflenet_path{root_path_g / "shufflenet_head_48x48_color"};
    TinyPath::path img_path (root_path_g / "shufflenet_head_48x48_color/Mikasa.jpg");
    cv::Mat img (cv::imread (img_path.string()));
    Classifier classifier (shufflenet_path.string());
    auto results = classifier.Classify (img);
    
    for (auto x : results)
    {
        cout << x << endl;
    }
    
#ifdef _MSC_VER
    system ("pause");
#endif // _MSC_VER
}

#endif // TOOLS_FORWORD_SHUFFLENET_CPP


