/*
	attention!
	The mobile ssd model is coming from https://github.com/chuanqi305/MobileNet-SSD
	there is an issue when you use CPP code to load this net:
	https://github.com/chuanqi305/MobileNet-SSD/issues/19#issuecomment-322992276
	I had add a line( sample_normalized *= 0.007843; ) to Preprocess function
*/

#include "tools_config.h"
#ifdef TOOLS_FORWARD_MOBILENET_SSD_CPP

#include <iostream>
#include <caffe/caffe.hpp>
#include "my_configs.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include "my_configs.h"

using namespace caffe;
using namespace cv;
using namespace std;


class Detector
{
    public:
        Detector (const string& model_file,
                  const string& weights_file,
                  const string& mean_file,
                  const string& mean_value);
        Detector (const string& model_file, const string& weights_file, std::vector<float> mean_value = { 127.5, 127.5, 127.5 });
        std::vector<vector<float> > Detect (const cv::Mat& img);
        
    private:
        void SetMean (const string& mean_file, const string& mean_value);
        
        void WrapInputLayer (std::vector<cv::Mat>* input_channels);
        
        void Preprocess (const cv::Mat& img,
                         std::vector<cv::Mat>* input_channels);
                         
    private:
        std::shared_ptr<Net<float> > net_;
        cv::Size input_geometry_;
        int num_channels_;
        cv::Mat mean_;
};


const size_t inWidth = 300;
const size_t inHeight = 300;
const float WHRatio = inWidth / (float) inHeight;
const float inScaleFactor = 0.007843f;
const float meanVal = 127.5;
const float confidence_threshold = 0.3;
const char* classNames[] = { "background",
                             "aeroplane", "bicycle", "bird", "boat",
                             "bottle", "bus", "car", "cat", "chair",
                             "cow", "diningtable", "dog", "horse",
                             "motorbike", "person", "pottedplant",
                             "sheep", "sofa", "train", "tvmonitor"
                           };


int main()
{
    std::string model_prototxt_path{ (root_path_g / "models/mobileNet_ssd/MobileNetSSD_deploy.prototxt").string() };
    std::string trained_model{ (root_path_g / "models/mobileNet_ssd/MobileNetSSD_deploy.caffemodel").string() };
    Detector detector (model_prototxt_path, trained_model, "", "127.5,127.5,127.5");
    VideoCapture cap ( (root_path_g / "models/mobileNet_ssd/handShake_0036.avi").string());
    
    if (!cap.isOpened()) // check if we succeeded
    {
        cap = VideoCapture (0);
        
        if (!cap.isOpened())
        {
            cout << "Couldn't find camera" << endl;
            return -1;
        }
    }
    
    cv::Mat img;
    int frame_count = 0;
    
    while (true)
    {
        bool success = cap.read (img);
        
        if (!success)
        {
            LOG (INFO) << "Process " << frame_count << " frames from " << "camera. ";
            break;
        }
        
        CHECK (!img.empty()) << "Error when read frame";
        chrono::system_clock::time_point start = chrono::system_clock::now();
        std::vector<vector<float> > detections = detector.Detect (img.clone());
        chrono::system_clock::time_point end = chrono::system_clock::now();
        cout << "detection ms: " << chrono::duration_cast<chrono::milliseconds> (end - start).count() << endl;
        
        /* Print the detection results. */
        for (int i = 0; i < detections.size(); ++i)
        {
            const vector<float>& d = detections[i];
            // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
            CHECK_EQ (d.size(), 7);
            const float score = d[2];// 对象置信度
            
            if (score >= confidence_threshold)
            {
                size_t object_class = static_cast<size_t> (d[1]);
                int lefttop_x = static_cast<int> (d[3] * img.cols);
                int lefttop_y = static_cast<int> (d[4] * img.rows);
                int rightbottom_x = static_cast<int> (d[5] * img.cols);
                int rightbottom_y = static_cast<int> (d[6] * img.rows);
                cv::rectangle (img,
                               cv::Rect (cv::Point (lefttop_x, lefttop_y),
                                         cv::Point (rightbottom_x, rightbottom_y)),
                               cv::Scalar (0, 255, 0));
                int baseLine = 0;
                string label = string (classNames[object_class]) + ": " + to_string (score);
                Size labelSize = getTextSize (label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                putText (img, label, Point (lefttop_x, lefttop_y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar (0, 0, 255));
            }
            
            imshow ("img with boxs", img);
            waitKey (1);
        }
        
        ++frame_count;
    }
    
    if (cap.isOpened())
    {
        cap.release();
    }
}


Detector::Detector (const string& model_file,
                    const string& weights_file,
                    const string& mean_file,
                    const string& mean_value)
{
#ifdef CPU_ONLY
    Caffe::set_mode (Caffe::CPU);
#else
    Caffe::set_mode (Caffe::GPU);
#endif
    /* Load the network. */
    net_.reset (new Net<float> (model_file, TEST));
    net_->CopyTrainedLayersFrom (weights_file);
    CHECK_EQ (net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ (net_->num_outputs(), 1) << "Network should have exactly one output.";
    Blob<float>* input_layer = net_->input_blobs() [0];
    num_channels_ = input_layer->channels();
    CHECK (num_channels_ == 3 || num_channels_ == 1)
            << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size (input_layer->width(), input_layer->height());
    /* Load the binaryproto mean file. */
    SetMean (mean_file, mean_value);
}

Detector::Detector (const string& model_file, const string& weights_file, std::vector<float> mean_value)
{
#ifdef CPU_ONLY
    Caffe::set_mode (Caffe::CPU);
#else
    Caffe::set_mode (Caffe::GPU);
#endif
    /* Load the network. */
    net_.reset (new Net<float> (model_file, TEST));
    net_->CopyTrainedLayersFrom (weights_file);
    CHECK_EQ (net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ (net_->num_outputs(), 1) << "Network should have exactly one output.";
    Blob<float>* input_layer = net_->input_blobs() [0];
    num_channels_ = input_layer->channels();
    CHECK (num_channels_ == 3 || num_channels_ == 1)
            << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size (input_layer->width(), input_layer->height());
    
    /* Load the binaryproto mean file. */
    for (int i = 0; i < 3; i++)
    { assert (mean_value[i] >= 0.0 && mean_value[i] <= 255.0); }
    
    SetMean (std::string(), std::to_string (mean_value[0]) + "," + std::to_string (mean_value[0]) + "," + std::to_string (mean_value[0]));
}

std::vector<vector<float> > Detector::Detect (const cv::Mat& img)
{
    Blob<float>* input_layer = net_->input_blobs() [0];
    input_layer->Reshape (1, num_channels_, input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();
    std::vector<cv::Mat> input_channels;
    WrapInputLayer (&input_channels);
    Preprocess (img, &input_channels);
    net_->Forward();
    /* Copy the output layer to a std::vector */
    Blob<float>* result_blob = net_->output_blobs() [0];
    const float* result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<vector<float> > detections;
    
    for (int k = 0; k < num_det; ++k)
    {
        if (result[0] == -1)
        {
            // Skip invalid detection.
            result += 7;
            continue;
        }
        
        vector<float> detection (result, result + 7);
        detections.push_back (detection);
        result += 7;
    }
    
    return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean (const string& mean_file, const string& mean_value)
{
    cv::Scalar channel_mean;
    
    if (!mean_file.empty())
    {
        CHECK (mean_value.empty()) << "Cannot specify mean_file and mean_value at the same time";
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie (mean_file.c_str(), &blob_proto);
        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto (blob_proto);
        CHECK_EQ (mean_blob.channels(), num_channels_) << "Number of channels of mean file doesn't match input layer.";
        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<cv::Mat> channels;
        float* data = mean_blob.mutable_cpu_data();
        
        for (int i = 0; i < num_channels_; ++i)
        {
            /* Extract an individual channel. */
            cv::Mat channel (mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back (channel);
            data += mean_blob.height() * mean_blob.width();
        }
        
        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge (channels, mean);
        /* Compute the global mean pixel value and create a mean image
        * filled with this value. */
        channel_mean = cv::mean (mean);
        mean_ = cv::Mat (input_geometry_, mean.type(), channel_mean);
    }
    
    if (!mean_value.empty())
    {
        CHECK (mean_file.empty()) << "Cannot specify mean_file and mean_value at the same time";
        stringstream ss (mean_value);
        vector<float> values;
        string item;
        
        while (getline (ss, item, ','))
        {
            float value = std::atof (item.c_str());
            values.push_back (value);
        }
        
        CHECK (values.size() == 1 || values.size() == num_channels_) <<
                "Specify either 1 mean_value or as many as channels: " << num_channels_;
        std::vector<cv::Mat> channels;
        
        for (int i = 0; i < num_channels_; ++i)
        {
            /* Extract an individual channel. */
            cv::Mat channel (input_geometry_.height, input_geometry_.width, CV_32FC1,
                             cv::Scalar (values[i]));
            channels.push_back (channel);
        }
        
        cv::merge (channels, mean_);
    }
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Detector::WrapInputLayer (std::vector<cv::Mat>* input_channels)
{
    Blob<float>* input_layer = net_->input_blobs() [0];
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel (height, width, CV_32FC1, input_data);
        input_channels->push_back (channel);
        input_data += width * height;
    }
}

void Detector::Preprocess (const cv::Mat& img, std::vector<cv::Mat>* input_channels)
{
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    
    if (img.channels() == 3 && num_channels_ == 1)
    {
        cv::cvtColor (img, sample, cv::COLOR_BGR2GRAY);
    }
    
    else
        if (img.channels() == 4 && num_channels_ == 1)
        {
            cv::cvtColor (img, sample, cv::COLOR_BGRA2GRAY);
        }
        
        else
            if (img.channels() == 4 && num_channels_ == 3)
            {
                cv::cvtColor (img, sample, cv::COLOR_BGRA2BGR);
            }
            
            else
                if (img.channels() == 1 && num_channels_ == 3)
                {
                    cv::cvtColor (img, sample, cv::COLOR_GRAY2BGR);
                }
                
                else
                {
                    sample = img;
                }
                
    cv::Mat sample_resized;
    
    if (sample.size() != input_geometry_)
    {
        cv::resize (sample, sample_resized, input_geometry_);
    }
    
    else
    {
        sample_resized = sample;
    }
    
    cv::Mat sample_float;
    
    if (num_channels_ == 3)
    {
        sample_resized.convertTo (sample_float, CV_32FC3);
    }
    
    else
    {
        sample_resized.convertTo (sample_float, CV_32FC1);
    }
    
    cv::Mat sample_normalized;
    cv::subtract (sample_float, mean_, sample_normalized);
    sample_normalized *= inScaleFactor; //this line was added by HJiahu. refer to https://github.com/chuanqi305/MobileNet-SSD/issues/19
    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split (sample_normalized, *input_channels);
    CHECK (reinterpret_cast<float*> (input_channels->at (0).data)
           == net_->input_blobs() [0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}

#endif // TOOLS_FORWARD_MOBILENET_SSD_CPP