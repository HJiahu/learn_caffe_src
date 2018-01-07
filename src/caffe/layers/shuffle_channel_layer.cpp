#include <algorithm>
#include <vector>

#include "caffe/layers/shuffle_channel_layer.hpp"

namespace caffe
{

    template <typename Dtype>
    void ShuffleChannelLayer<Dtype>::LayerSetUp (const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
    {
        group_ = this->layer_param_.shuffle_channel_param().group();
        CHECK_GT (group_, 0) << "group must be greater than 0";
        //temp_blob_.ReshapeLike(*bottom[0]);
        top[0]->ReshapeLike (*bottom[0]);
    }
    
    // 读forward_cpu函数可知下面函数的参数
    template <typename Dtype>
    void ShuffleChannelLayer<Dtype>::Resize_cpu (
        Dtype *output,// 输出层指针
        const Dtype *input,// 输入层指针
        int group_row,// shuffle层对输入层所有通道所分的组数
        int group_column,// 每组中包含的输入层channel的个数
        int len// 输入层每个feature map的像素数
    )
    {
		// 下面的代码将输入channel看作了一个矩阵，结构为(group_row,group_column)
        for (int i = 0; i < group_row; ++i) // 元素的行
        {
            for (int j = 0; j < group_column ; ++j) // 元素的列
            {
                const Dtype* p_i = input + (i * group_column + j) * len; //p_i依次指向每一个feature map在内存中的首地址
                Dtype* p_o = output + (j * group_row + i) * len; //行与列被互换，即矩阵的转置
                caffe_copy (len, p_i, p_o);
            }
        }
    }
    
    template <typename Dtype>
    void ShuffleChannelLayer<Dtype>::Reshape (const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
    {
        int channels_ = bottom[0]->channels();
        int height_ = bottom[0]->height();
        int width_ = bottom[0]->width();
        top[0]->Reshape (bottom[0]->num(), channels_, height_, width_);
    }
    
    template <typename Dtype>
    void ShuffleChannelLayer<Dtype>::Forward_cpu (const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top)
    {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        const int num = bottom[0]->shape (0);
        // 这里的名字取的似乎不太合适，这个变量代表了输入层参数的个数，即输入层所有feature map参数总和（像素数总和）
        const int feature_map_size = bottom[0]->count (1);
        const int sp_sz = bottom[0]->count (2);// 这个是每个输入feature map中参数的个数，即feature map中的像素数
        const int chs = bottom[0]->shape (1);// 输入层通道的个数，即feature map的数量
        int group_row = group_;// shufflenet层的group参数值，即shuffe层将输入层的通道分的组数
        int group_column = int (chs / group_row);// 每个组中输入通道的个数，即把输入通道分为group组，每组中通道的个数
        CHECK_EQ (chs, (group_column * group_row)) << "Wrong group size.";
        
        //Dtype* temp_data = temp_blob_.mutable_cpu_data();
        for (int n = 0; n < num; ++n)
        {
            // 因为在test的时候num恒为1所以下面的代码可以简化为：
            // Resize_cpu (top_data , bottom_data , group_row, group_column, sp_sz);
            Resize_cpu (top_data + n * feature_map_size, bottom_data + n * feature_map_size, group_row, group_column, sp_sz);
        }
        
        //caffe_copy(bottom[0]->count(), temp_blob_.cpu_data(), top_data);
    }
    
    template <typename Dtype>
    void ShuffleChannelLayer<Dtype>::Backward_cpu (const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom)
    {
        if (propagate_down[0])
        {
            const Dtype* top_diff = top[0]->cpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            const int num = bottom[0]->shape (0);
            const int feature_map_size = bottom[0]->count (1);
            const int sp_sz = bottom[0]->count (2);
            const int chs = bottom[0]->shape (1);
            int group_row = int (chs / group_);
            int group_column = group_;
            
            //Dtype* temp_diff = temp_blob_.mutable_cpu_diff();
            for (int n = 0; n < num; ++n)
            {
                Resize_cpu (bottom_diff + n * feature_map_size, top_diff + n * feature_map_size, group_row, group_column, sp_sz);
            }
            
            //caffe_copy(top[0]->count(), temp_blob_.cpu_diff(), bottom_diff);
        }
    }
    
    
#ifdef CPU_ONLY
    STUB_GPU (ShuffleChannelLayer);
#endif
    
    INSTANTIATE_CLASS (ShuffleChannelLayer);
    REGISTER_LAYER_CLASS (ShuffleChannel);
}  // namespace caffe
