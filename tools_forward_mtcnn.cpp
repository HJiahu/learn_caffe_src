#include"tools_config.h"
#ifdef TOOLS_FORWARD_MTCNN_CPP
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
// tool functions in MTCNN pycodes
vector<float> calculate_scaltes (cv::Mat img);



































































/******************    implement    **********************/
vector<float> calculate_scaltes (cv::Mat img)
{
    auto caffe_img = img.clone();
    float pr_scale = 1.0;
    int h = caffe_img.rows;
    int w = caffe_img.cols;
    int ch = caffe_img.channels;
    
	// 将图片的一个边缩放到1000像素左右
    if (std::min (w, h) > 1000)
    {
        pr_scale = 1000.0 / std::min (h, w);
        w = int (w * pr_scale);
        h = int (h * pr_scale);
    }
    
    else
        if (std::max (w, h) < 1000)
        {
            pr_scale = 1000.0 / std::max (h, w);
            w = int (w * pr_scale);
            h = int (h * pr_scale);
        }
        
    vector<float> scales;
    float factor = 0.709;
    int factor_count = 0;
    auto minl = std::min (h, w);
    
	// 以最短边为准对图片进行缩放以获得图片的图像金字塔
	// 返回对原图的缩放因子（浮点型数组），缩放因子从大到小排列
    while (minl >= 12)
    {
        scales.push_back (pr_scale * pow (factor, factor_count));
        minl *= factor;
        factor_count++;
    }
    
    return scales;
}

#endif // TOOLS_FORWARD_MTCNN_CPP


