//#define READ_THIS_FILE
#ifdef READ_THIS_FILE

#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/FRCNN/util/frcnn_vis.hpp"
#include "caffe/FRCNN/api/api.hpp"

int main (int argc, char** argv)
{
    caffe::Caffe::set_mode (caffe::Caffe::CPU);
    std::string proto_file = R"(I:\learn_caffe\learn_caffe\caffe_src\faster_rcnn_model\vgg16_with_RPN_test.proto)";
    std::string model_file = R"(D:\Programs\caffe_deps\caffe_VS2015x64DCPU\bin\VGG16_faster_rcnn_final.caffemodel)";
    std::string default_config_file = R"(I:\learn_caffe\learn_caffe\caffe_src\faster_rcnn_model\frcnn_config\default_config.json)";
    API::Set_Config (default_config_file);
    API::Detector detector (proto_file, model_file);
    std::vector<caffe::Frcnn::BBox<float> > results;
    caffe::Timer time_;
    cv::Mat image = cv::imread (R"(I:\learn_caffe\learn_caffe\caffe_src\faster_rcnn_model\frcnn_config\test_images\004545.jpg)");
    detector.predict (image, results);
    
    for (size_t obj = 0; obj < results.size(); obj++)
    {
        LOG (INFO) << results[obj].to_string();
    }
    
    cv::Mat ori;
    ori = image.clone();
    
    for (int label = 0; label < caffe::Frcnn::FrcnnParam::n_classes; label++)
    {
        std::vector<caffe::Frcnn::BBox<float> > cur_res;
        
        for (size_t idx = 0; idx < results.size(); idx++)
        {
            if (results[idx].id == label)
            {
                cur_res.push_back (results[idx]);
            }
        }
        
        if (cur_res.size() == 0) { continue; }
        
        caffe::Frcnn::vis_detections (ori, cur_res, caffe::Frcnn::LoadVocClass());
        //std::cout << caffe::Frcnn::GetClassName (caffe::Frcnn::LoadVocClass(), label).c_str() << std::endl;
    }
    
    cv::imshow ("TEST FASTER_RCNN", ori);
    cv::waitKey (0);
    return 0;
}

#endif //READ_THIS_FILE