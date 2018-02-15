#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

    template <typename Dtype>
    void BatchNormLayer<Dtype>::LayerSetUp (const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top)
    {
        // 可参考 http://blog.csdn.net/lanran2/article/details/56278072
        BatchNormParameter param = this->layer_param_.batch_norm_param();
        // BN层在训练的时候需要一个均值，这个均值是从整个训练集中的获得的
        // 但是我们在训练的时候一次只使用一小部分的数据所以均值在训练的时候
        // 需要更新，caffe没有使用简单的累加而是在累加的时候给旧值一个小于1的权值
        // 这里moving_average_fraction_就是这个权值
        moving_average_fraction_ = param.moving_average_fraction();
        // use_global_stats 是指Train还是Test，如果为True，那么就是指Test；
        use_global_stats_ = this->phase_ == TEST;
        
        if (param.has_use_global_stats())
        { use_global_stats_ = param.use_global_stats(); }
        
        if (bottom[0]->num_axes() == 1)
        { channels_ = 1; }
        
        else
        { channels_ = bottom[0]->shape (1); }
        
        // 归一化的时候需要除以方差，为了防止除以0，一般给方差添加一个非0的整数
        eps_ = param.eps();
        
        if (this->blobs_.size() > 0)
        {
            LOG (INFO) << "Skipping parameter initialization";
        }
        
        else
        {
            // 当前层有3个blob，分别保存每个通道的均值、方差和
            // 前两个blob的大小和通道数相同，最后一个blob中只有一个元素
            this->blobs_.resize (3);
            vector<int> sz;
            sz.push_back (channels_);
            this->blobs_[0].reset (new Blob<Dtype> (sz));
            this->blobs_[1].reset (new Blob<Dtype> (sz));
            sz[0] = 1;
            this->blobs_[2].reset (new Blob<Dtype> (sz));
            
            // 将所有blob中的值都初始化为0
            for (int i = 0; i < 3; ++i)
            {
                caffe_set (this->blobs_[i]->count(), Dtype (0),
                           this->blobs_[i]->mutable_cpu_data());
            }
        }
        
        // Mask statistics from optimization by setting local learning rates
        // for mean, variance, and the bias correction to zero.
        for (int i = 0; i < this->blobs_.size(); ++i)
        {
            if (this->layer_param_.param_size() == i)
            {
                ParamSpec* fixed_param_spec = this->layer_param_.add_param();
                fixed_param_spec->set_lr_mult (0.f);
            }
            
            else
            {
                CHECK_EQ (this->layer_param_.param (i).lr_mult(), 0.f)
                        << "Cannot configure batch normalization statistics as layer "
                        << "parameters.";
            }
        }
    }
    
    template <typename Dtype>
    void BatchNormLayer<Dtype>::Reshape (const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top)
    {
        if (bottom[0]->num_axes() >= 1)
        { CHECK_EQ (bottom[0]->shape (1), channels_); }
        
        // 因为当前层只是将输入归一化所以输入和输出的尺寸是相同的
        top[0]->ReshapeLike (*bottom[0]);
        vector<int> sz;
        // //定义mean_，variance_，temp_，x_norm_，batch_sum_multiplier_的形状
        sz.push_back (channels_);
        // 每一个channel对应一个均值与一个方差
        mean_.Reshape (sz);
        variance_.Reshape (sz);
        // temp_和x_norm_的形状和输入输出相同
        temp_.ReshapeLike (*bottom[0]);
        x_norm_.ReshapeLike (*bottom[0]);
        // batch_sum_multiplier_中元素的个数等于num
        sz[0] = bottom[0]->shape (0);
        batch_sum_multiplier_.Reshape (sz);
        // 每一个channel中元素的个数
        int spatial_dim = bottom[0]->count() / (channels_ * bottom[0]->shape (0));
        
        if (spatial_sum_multiplier_.num_axes() == 0 ||
                spatial_sum_multiplier_.shape (0) != spatial_dim)
        {
            sz[0] = spatial_dim;
            spatial_sum_multiplier_.Reshape (sz);
            Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
            caffe_set (spatial_sum_multiplier_.count(), Dtype (1), multiplier_data);
        }
        
        int numbychans = channels_ * bottom[0]->shape (0);
        
        // 定义 num_by_chans_ 的形状为channels_*bottom[0]->shape(0)
        if (num_by_chans_.num_axes() == 0 ||
                num_by_chans_.shape (0) != numbychans)
        {
            sz[0] = numbychans;
            num_by_chans_.Reshape (sz);
            caffe_set (batch_sum_multiplier_.count(), Dtype (1),
                       batch_sum_multiplier_.mutable_cpu_data());
        }
    }
    
    template <typename Dtype>
    void BatchNormLayer<Dtype>::Forward_cpu (const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top)
    {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        int num = bottom[0]->shape (0);
        int spatial_dim = bottom[0]->count() / (bottom[0]->shape (0) * channels_);
        
        // 如果bottom和top的值不相同，就把bottom中的值赋给top
        if (bottom[0] != top[0])
        {
            caffe_copy (bottom[0]->count(), bottom_data, top_data);
        }
        
        // 如果 use_global_stats_ 为true则说明是TEST，使用已计算好的均值与方差
        // 其中mean保存在blobs_[0]中，variance保存在blobs_[1]中
        if (use_global_stats_)
        {
            // use the stored mean/variance estimates.
            const Dtype scale_factor = this->blobs_[2]->cpu_data() [0] == 0 ?
                                       0 : 1 / this->blobs_[2]->cpu_data() [0];
            caffe_cpu_scale (variance_.count(), scale_factor,
                             this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
            caffe_cpu_scale (variance_.count(), scale_factor,
                             this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
        }
        
        // 如果 use_global_stats_ 为 true 则说明是TEST，则均值与方差需要通过计算获得
        else
        {
            // compute mean
            // 这个矩阵与向量相乘，目的是计算每个feature map的数值和，然后在除以1./(num*spatial_dim)
            // bottom_data: (channels_*num) x (spatial_dim)
            // spatial_sum_multiplier: spatial_dim x 1
            // alpha : 1./(num*spatial_dim); beta : 0
            // num_by_chans = alpha * (bottom_data x spatial_sum_multiplier) + beta * num_by_chans
            // 其中spatial_sum_multiplier的值都为1
            caffe_cpu_gemv<Dtype> (CblasNoTrans, channels_ * num, spatial_dim,
                                   1. / (num * spatial_dim), bottom_data,
                                   spatial_sum_multiplier_.cpu_data(), 0.,
                                   num_by_chans_.mutable_cpu_data());
            // 注意关键字是CblasTrans！！
            // num_by_chans_ : channels_ x num;
            // batch_sum_multiplier_ : num x 1;
            // mean_ = 1. x (num_by_chans_ x batch_sum_multiplier_)
            // mean_ : channels_ x 1
            // 计算得到对应channels的平均值，这也解释了为什么之前要除以1./(num*spatial_dim)
            // 而不是仅除以1./spatial_dim，这样减少了计算量
            caffe_cpu_gemv<Dtype> (CblasTrans, num, channels_, 1.,
                                   num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
                                   mean_.mutable_cpu_data());
        }
        
        // subtract mean
        caffe_cpu_gemm<Dtype> (CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
                               batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
                               num_by_chans_.mutable_cpu_data());
        // 最后的 top_data 保存的就是每个值减去对应channel的均值后的结果
        caffe_cpu_gemm<Dtype> (CblasNoTrans, CblasNoTrans, channels_ * num,
                               spatial_dim, 1, -1, num_by_chans_.cpu_data(),
                               spatial_sum_multiplier_.cpu_data(), 1., top_data);
                               
        if (!use_global_stats_)
        {
            // compute variance using var(X) = E((X-EX)^2)
            caffe_sqr<Dtype> (top[0]->count(), top_data,
                              temp_.mutable_cpu_data());  // (X-EX)^2
            caffe_cpu_gemv<Dtype> (CblasNoTrans, channels_ * num, spatial_dim,
                                   1. / (num * spatial_dim), temp_.cpu_data(),
                                   spatial_sum_multiplier_.cpu_data(), 0.,
                                   num_by_chans_.mutable_cpu_data());
            caffe_cpu_gemv<Dtype> (CblasTrans, num, channels_, 1.,
                                   num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
                                   variance_.mutable_cpu_data());  // E((X_EX)^2)
            // compute and save moving average
            this->blobs_[2]->mutable_cpu_data() [0] *= moving_average_fraction_;
            this->blobs_[2]->mutable_cpu_data() [0] += 1;
            caffe_cpu_axpby (mean_.count(), Dtype (1), mean_.cpu_data(),
                             moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());
            int m = bottom[0]->count() / channels_;
            Dtype bias_correction_factor = m > 1 ? Dtype (m) / (m - 1) : 1;
            caffe_cpu_axpby (variance_.count(), bias_correction_factor,
                             variance_.cpu_data(), moving_average_fraction_,
                             this->blobs_[1]->mutable_cpu_data());
        }
        
        // normalize variance
        caffe_add_scalar (variance_.count(), eps_, variance_.mutable_cpu_data());
        caffe_sqrt (variance_.count(), variance_.cpu_data(),
                    variance_.mutable_cpu_data());
        // replicate variance to input size
        caffe_cpu_gemm<Dtype> (CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
                               batch_sum_multiplier_.cpu_data(), variance_.cpu_data(), 0.,
                               num_by_chans_.mutable_cpu_data());
        caffe_cpu_gemm<Dtype> (CblasNoTrans, CblasNoTrans, channels_ * num,
                               spatial_dim, 1, 1., num_by_chans_.cpu_data(),
                               spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
        caffe_div (temp_.count(), top_data, temp_.cpu_data(), top_data);
        // TODO(cdoersch): The caching is only needed because later in-place layers
        //                 might clobber the data.  Can we skip this if they won't?
        caffe_copy (x_norm_.count(), top_data,
                    x_norm_.mutable_cpu_data());
    }
    
    template <typename Dtype>
    void BatchNormLayer<Dtype>::Backward_cpu (const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom)
    {
        const Dtype* top_diff;
        
        if (bottom[0] != top[0])
        {
            top_diff = top[0]->cpu_diff();
        }
        
        else
        {
            caffe_copy (x_norm_.count(), top[0]->cpu_diff(), x_norm_.mutable_cpu_diff());
            top_diff = x_norm_.cpu_diff();
        }
        
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        
        if (use_global_stats_)
        {
            caffe_div (temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
            return;
        }
        
        const Dtype* top_data = x_norm_.cpu_data();
        int num = bottom[0]->shape() [0];
        int spatial_dim = bottom[0]->count() / (bottom[0]->shape (0) * channels_);
        // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
        //
        // dE(Y)/dX =
        //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
        //     ./ sqrt(var(X) + eps)
        //
        // where \cdot and ./ are hadamard product and elementwise division,
        // respectively, dE/dY is the top diff, and mean/var/sum are all computed
        // along all dimensions except the channels dimension.  In the above
        // equation, the operations allow for expansion (i.e. broadcast) along all
        // dimensions except the channels dimension where required.
        // sum(dE/dY \cdot Y)
        caffe_mul (temp_.count(), top_data, top_diff, bottom_diff);
        caffe_cpu_gemv<Dtype> (CblasNoTrans, channels_ * num, spatial_dim, 1.,
                               bottom_diff, spatial_sum_multiplier_.cpu_data(), 0.,
                               num_by_chans_.mutable_cpu_data());
        caffe_cpu_gemv<Dtype> (CblasTrans, num, channels_, 1.,
                               num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
                               mean_.mutable_cpu_data());
        // reshape (broadcast) the above
        caffe_cpu_gemm<Dtype> (CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
                               batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
                               num_by_chans_.mutable_cpu_data());
        caffe_cpu_gemm<Dtype> (CblasNoTrans, CblasNoTrans, channels_ * num,
                               spatial_dim, 1, 1., num_by_chans_.cpu_data(),
                               spatial_sum_multiplier_.cpu_data(), 0., bottom_diff);
        // sum(dE/dY \cdot Y) \cdot Y
        caffe_mul (temp_.count(), top_data, bottom_diff, bottom_diff);
        // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
        caffe_cpu_gemv<Dtype> (CblasNoTrans, channels_ * num, spatial_dim, 1.,
                               top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
                               num_by_chans_.mutable_cpu_data());
        caffe_cpu_gemv<Dtype> (CblasTrans, num, channels_, 1.,
                               num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
                               mean_.mutable_cpu_data());
        // reshape (broadcast) the above to make
        // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
        caffe_cpu_gemm<Dtype> (CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
                               batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
                               num_by_chans_.mutable_cpu_data());
        caffe_cpu_gemm<Dtype> (CblasNoTrans, CblasNoTrans, num * channels_,
                               spatial_dim, 1, 1., num_by_chans_.cpu_data(),
                               spatial_sum_multiplier_.cpu_data(), 1., bottom_diff);
        // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
        caffe_cpu_axpby (temp_.count(), Dtype (1), top_diff,
                         Dtype (-1. / (num * spatial_dim)), bottom_diff);
        // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
        // pass.
        caffe_div (temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
    }
    
    
#ifdef CPU_ONLY
    STUB_GPU (BatchNormLayer);
#endif
    
    INSTANTIATE_CLASS (BatchNormLayer);
    REGISTER_LAYER_CLASS (BatchNorm);
}  // namespace caffe
