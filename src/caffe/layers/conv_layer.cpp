#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe
{

    template <typename Dtype>
    void ConvolutionLayer<Dtype>::compute_output_shape()
    {
        const int* kernel_shape_data = this->kernel_shape_.cpu_data();
        const int* stride_data = this->stride_.cpu_data();
        const int* pad_data = this->pad_.cpu_data();
        const int* dilation_data = this->dilation_.cpu_data();
        this->output_shape_.clear();
        
        for (int i = 0; i < this->num_spatial_axes_; ++i) // num_spatial_axes_指的是卷积的维度，一般为2，即二维卷积
        {
            // i + 1 to skip channel axis
            const int input_dim = this->input_shape (i + 1);// 每个通道中数据的维度，以图片为例即宽高
            // 在这里进行卷积核的扩展操作
            const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
            // 在这里计算卷积过后生成的blob的高和宽
            const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent) / stride_data[i] + 1;
            this->output_shape_.push_back (output_dim);
        }
    }
    
    template <typename Dtype>
    void ConvolutionLayer<Dtype>::Forward_cpu (const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
    {
        /*
           因为lenet是单通道，shufflenet中有group所以以cifar10为例说明一些问题
           以cifar10的conv1为例(   num_output: 32    pad: 2    kernel_size: 5    stride: 1)：
        		bottom size: (1,3,32,32)   top size: (1,32,32,32)
        		blobs_[0] size: (32,3,5,5)
        
           以cifar10的conv2为例(    num_output: 32    pad: 2    kernel_size: 5    stride: 1)
        		bottom size: (1,32,16,16)   top size: (1,32,16,16)
        		blobs_[0] size: (32,32,5,5)
        
        	从上面的数据可以得到（不考虑group_）：caffe中卷积核的个数并不一定与 num_output 相同，卷积核的个数为 输入channels*输出channels
        	也就是每一个输出通道与每一个输入通道都有一个对应的卷积核
        */
        const Dtype* weight = this->blobs_[0]->cpu_data();
        
        for (int i = 0; i < bottom.size(); ++i)
        {
            const Dtype* bottom_data = bottom[i]->cpu_data();
            Dtype* top_data = top[i]->mutable_cpu_data();
            
            for (int n = 0; n < this->num_; ++n)
            {
                this->forward_cpu_gemm (bottom_data + n * this->bottom_dim_, weight, top_data + n * this->top_dim_);
                
                if (this->bias_term_)
                {
                    const Dtype* bias = this->blobs_[1]->cpu_data();
                    this->forward_cpu_bias (top_data + n * this->top_dim_, bias);
                }
            }
        }
    }
    
    template <typename Dtype>
    void ConvolutionLayer<Dtype>::Backward_cpu (const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
    {
        const Dtype* weight = this->blobs_[0]->cpu_data();
        Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
        
        for (int i = 0; i < top.size(); ++i)
        {
            const Dtype* top_diff = top[i]->cpu_diff();
            const Dtype* bottom_data = bottom[i]->cpu_data();
            Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
            
            // Bias gradient, if necessary.
            if (this->bias_term_ && this->param_propagate_down_[1])
            {
                Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
                
                for (int n = 0; n < this->num_; ++n)
                {
                    this->backward_cpu_bias (bias_diff, top_diff + n * this->top_dim_);
                }
            }
            
            if (this->param_propagate_down_[0] || propagate_down[i])
            {
                for (int n = 0; n < this->num_; ++n)
                {
                    // gradient w.r.t. weight. Note that we will accumulate diffs.
                    if (this->param_propagate_down_[0])
                    {
                        this->weight_cpu_gemm (bottom_data + n * this->bottom_dim_,
                                               top_diff + n * this->top_dim_, weight_diff);
                    }
                    
                    // gradient w.r.t. bottom data, if necessary.
                    if (propagate_down[i])
                    {
                        this->backward_cpu_gemm (top_diff + n * this->top_dim_, weight,
                                                 bottom_diff + n * this->bottom_dim_);
                    }
                }
            }
        }
    }
    
#ifdef CPU_ONLY
    STUB_GPU (ConvolutionLayer);
#endif
    
    INSTANTIATE_CLASS (ConvolutionLayer);
    
}  // namespace caffe
