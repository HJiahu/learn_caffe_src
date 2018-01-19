#include <algorithm>
#include <vector>

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

    // 因为全连接层越来越不受待见，所以下面以卷积层（pooling也是卷积层）为前导层来说明 softmax
    // 假设输入只有一张图片，即num = 1
    template <typename Dtype>
    void SoftmaxLayer<Dtype>::Reshape (const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top)
    {
        // softmax_axis_ 默认为 1，即在通道(feature map)间做softmax
        softmax_axis_ = bottom[0]->CanonicalAxisIndex (this->layer_param_.softmax_param().axis());
        top[0]->ReshapeLike (*bottom[0]);// top和bottom的形状必须一致
        vector<int> mult_dims (1, bottom[0]->shape (softmax_axis_));// 等于网络分类的类别数
		// sum_multiplier_是一个辅助变量，用于辅助矩阵的相乘，下面将其中元素均置为1.0
        sum_multiplier_.Reshape (mult_dims);// 注意这里mult_dim只有一个元素，所以sum_multiplier_也只有一维
        Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
        caffe_set (sum_multiplier_.count(), Dtype (1), multiplier_data);// 将 sum_multiplier_ 中的元素都初始化为1.0
        outer_num_ = bottom[0]->count (0, softmax_axis_);// num 值
        // 这里的 inner_num_ 指的是 softmax_axis_ 这一维度中单个元素的 float 数个数
        // 如果上一层是卷积，则 inner_num_ 为输入数据中单个 feature map 中元素（一般为浮点数）的个数
        inner_num_ = bottom[0]->count (softmax_axis_ + 1);// 如果维度2不存在（以cifar10为例），这里inner_num默认为1（参考 count 源码）
        vector<int> scale_dims = bottom[0]->shape();
        scale_dims[softmax_axis_] = 1;// 如果 softmax 的上一层是卷积层且输入num为1，那么scale_的尺寸与输入的feature map（一个）相同
        scale_.Reshape (scale_dims);
    }
    
    // softmaxlayer和withloss的区别应该是后者添加了一个loss函数，用于训练，前者用于TEST
    // http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/
    template <typename Dtype>
    void SoftmaxLayer<Dtype>::Forward_cpu (const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top)
    {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        Dtype* scale_data = scale_.mutable_cpu_data();// scale is an intermediate Blob to hold temporary results.
        int channels = bottom[0]->shape (softmax_axis_);
        // 如果上一层是卷积层，若 num = 1 则dim为所有输入 feature map 中浮点数的个数
        int dim = bottom[0]->count() / outer_num_;
        caffe_copy (bottom[0]->count(), bottom_data, top_data);// copy data from bottom to top
        
        // We need to subtract the max to avoid numerical issues, compute the exp,
        // and then normalize.
        for (int i = 0; i < outer_num_; ++i)
        {
            // initialize scale_data to the first plane
            // 假设num = 1，上一层为卷积层，则使用第一个 feature map 初始化 scale_data
            caffe_copy (inner_num_, bottom_data + i * dim, scale_data);
            
            // 在所有 feature map 的相同位置找最大值并置 scale_data 对应位置为最大值
            for (int j = 0; j < channels; j++)
            {
                for (int k = 0; k < inner_num_; k++)
                {
                    scale_data[k] = std::max (scale_data[k], bottom_data[i * dim + j * inner_num_ + k]);
                }
            }
            
			// softmax是logistic回归的引申，所以所求的概率不是按常规比例（http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92）
            // C=alpha*A*B + beta*C
            // subtraction，top中的所有元素都减去对应位置的最大值
            caffe_cpu_gemm<Dtype> (CblasNoTrans, CblasNoTrans,
                                   channels, inner_num_, 1,  
                                   -1.,// alpha
                                   sum_multiplier_.cpu_data(),// A，sum_multiplier_中的元素值均为 1.0
                                   scale_data,// B
                                   1.,// beta
                                   top_data //C
                                  );
            // exponentiation
            caffe_exp<Dtype> (dim, top_data, top_data);
            // sum after exp
            caffe_cpu_gemv<Dtype> (CblasTrans, channels, inner_num_, 1.,
                                   top_data, sum_multiplier_.cpu_data(), 0., scale_data);
                                   
            // division
            for (int j = 0; j < channels; j++)
            {
                caffe_div (inner_num_, top_data, scale_data, top_data);
                top_data += inner_num_;
            }
        }
    }
    
    template <typename Dtype>
    void SoftmaxLayer<Dtype>::Backward_cpu (const vector<Blob<Dtype>*>& top,
                                            const vector<bool>& propagate_down,
                                            const vector<Blob<Dtype>*>& bottom)
    {
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* top_data = top[0]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        Dtype* scale_data = scale_.mutable_cpu_data();
        int channels = top[0]->shape (softmax_axis_);
        int dim = top[0]->count() / outer_num_;
        caffe_copy (top[0]->count(), top_diff, bottom_diff);
        
        for (int i = 0; i < outer_num_; ++i)
        {
            // compute dot(top_diff, top_data) and subtract them from the bottom diff
            for (int k = 0; k < inner_num_; ++k)
            {
                scale_data[k] = caffe_cpu_strided_dot<Dtype> (channels,
                                bottom_diff + i * dim + k, inner_num_,
                                top_data + i * dim + k, inner_num_);
            }
            
            // subtraction
            caffe_cpu_gemm<Dtype> (CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
                                   -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
        }
        
        // elementwise multiplication
        caffe_mul (top[0]->count(), bottom_diff, top_data, bottom_diff);
    }
    
    
#ifdef CPU_ONLY
    STUB_GPU (SoftmaxLayer);
#endif
    
    INSTANTIATE_CLASS (SoftmaxLayer);
    
}  // namespace caffe
