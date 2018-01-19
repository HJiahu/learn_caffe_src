#ifndef CAFFE_CLASSFIER_H_
#define CAFFE_CLASSFIER_H_

//this header is create by HJiahu based on caffe\examples\cpp_classification\classification.cpp
//there is a cpp file for this header

#include <string>
#include <utility>
#include <vector>
#include <ostream>
#include <mutex>
#include <opencv2/opencv.hpp>


using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;
inline std::ostream& operator<< (std::ostream& out, const Prediction&pred)
{
    out << "[" << pred.first << ", " << pred.second * 100.0 << "%]";
    return out;
}

//下面这个函数主要用于Argmax，Argmax中按元素大小对其位置进行排序的算法可以借鉴
static bool PairCompare (const std::pair<float, int>& lhs, const std::pair<float, int>& rhs);
/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax (const std::vector<float>& v, int N);

/* Return the indices of the top N values of vector v. */
std::vector<int> Argmax (const std::vector<float>& v, int N);

class Classifier
{
    public:
        Classifier (const string& model_file,
                    const string& trained_file,
                    const string& mean_file,
                    const string& label_file);
        //使用约定的方式来初始化classifier，规定文件夹内文件命名方式如下：
        //model_file   :deploy.prototxt
        //trained_file :model.caffemodel
        //mean_file    :mean.binaryproto
        //label_file   :label.txt
        explicit Classifier (const string& cnn_model_path);
        ~Classifier();
        Classifier (const Classifier&) = delete;
        Classifier (Classifier&&) = delete;
        Classifier&operator = (const Classifier&) = delete;
        Classifier&operator = (Classifier&&);
        std::vector<Prediction> Classify (const cv::Mat& img, int N = 2);
        
    private:
		//将protobuf格式的图片转化为Mat型  float并赋予 mean_
        void SetMean (const string& mean_file);
        
        std::vector<float> Predict (const cv::Mat& img);
        
        void WrapInputLayer (std::vector<cv::Mat>* input_channels);
        
        void Preprocess (const cv::Mat& img, std::vector<cv::Mat>* input_channels);
        
        
    private:
        //std::shared_ptr<caffe::Net<float> > net_;
        void *net_;//为了减少头文件的依赖，这里使用void*指针
        cv::Size input_geometry_;//从图片的角度来看这里指的是图片的长宽
        int num_channels_;//图片的通道数，代码中是由input_layer中的水属性获得的而不是从输入的图片，当前对象将以layer为主对图像进行缩放与处理
        cv::Mat mean_;//均值图像其实就是所有图像对应位置的均值，使用时使用原图减去这个均值即可，故均值可以没有
        std::vector<string> labels_;//对应的标签，与输出维度匹配
};

inline bool PairCompare (const std::pair<float, int>& lhs,
                         const std::pair<float, int>& rhs)
{
    return lhs.first > rhs.first;
}



#endif // !CAFFE_CLASSFIER.H


