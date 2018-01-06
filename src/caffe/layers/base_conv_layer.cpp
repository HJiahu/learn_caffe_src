#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

    template <typename Dtype>
    void BaseConvolutionLayer<Dtype>::LayerSetUp (const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
    {
        // Configure the kernel size, padding, stride, and inputs.
        ConvolutionParameter conv_param = this->layer_param_.convolution_param();
        force_nd_im2col_ = conv_param.force_nd_im2col();//是否需要强制n维卷积 ，这个参数一般用不上
        /*
        	channel_axis_这个参数读取参数定义中的axis参数，默认为1（即(n,c,h,w)中的第二个参数），表示按channel求和，输入blob为(N,C,H,W)时，
        	一个输出通道对应的所有卷积核对输入blob上各通道做二维卷积，最后将输入各通道卷积的结果加起来，作为一张输出的特征子图
        */
        channel_axis_ = bottom[0]->CanonicalAxisIndex (conv_param.axis());
        const int first_spatial_axis = channel_axis_ + 1;//指示卷积输入图像的第一个轴，往往是H(height)
        const int num_axes = bottom[0]->num_axes();//得到bottom blob的维度
        num_spatial_axes_ = num_axes - first_spatial_axis;//卷积处理的维度数，一般是二维卷积
        CHECK_GE (num_spatial_axes_, 0);
        vector<int> spatial_dim_blob_shape (1, std::max (num_spatial_axes_, 1));//这个变量常见的值为{2}
        // Setup filter kernel dimensions (kernel_shape_).
        // 因为spatial_dim_blob_shape中只有一个元素所以kernel_shape_是一维的向量且有两个元素，分别表示高和宽
        kernel_shape_.Reshape (spatial_dim_blob_shape);//初始化卷积核的形状(高*宽)
        int* kernel_shape_data = kernel_shape_.mutable_cpu_data();// 得到记录卷积核形状数据地址
        
        // 设置卷积核的大小，如果没有指定长宽则长宽相同，下面代码使用相同的方法设置padding等参数
        if (conv_param.has_kernel_h() || conv_param.has_kernel_w())
        {
            CHECK_EQ (num_spatial_axes_, 2) << "kernel_h & kernel_w can only be used for 2D convolution.";
            CHECK_EQ (0, conv_param.kernel_size_size()) << "Either kernel_size or kernel_h/w should be specified; not both.";
            kernel_shape_data[0] = conv_param.kernel_h();
            kernel_shape_data[1] = conv_param.kernel_w();
        }
        
        else
        {
            const int num_kernel_dims = conv_param.kernel_size_size();
            CHECK (num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
                    << "kernel_size must be specified once, or once per spatial dimension "
                    << "(kernel_size specified " << num_kernel_dims << " times; "
                    << num_spatial_axes_ << " spatial dims).";
                    
            for (int i = 0; i < num_spatial_axes_; ++i)
            {
                kernel_shape_data[i] =
                    conv_param.kernel_size ( (num_kernel_dims == 1) ? 0 : i);
            }
        }
        
        for (int i = 0; i < num_spatial_axes_; ++i)
        {
            CHECK_GT (kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
        }
        
        // Setup stride dimensions (stride_).
        stride_.Reshape (spatial_dim_blob_shape);
        int* stride_data = stride_.mutable_cpu_data();
        
        if (conv_param.has_stride_h() || conv_param.has_stride_w())
        {
            CHECK_EQ (num_spatial_axes_, 2)
                    << "stride_h & stride_w can only be used for 2D convolution.";
            CHECK_EQ (0, conv_param.stride_size())
                    << "Either stride or stride_h/w should be specified; not both.";
            stride_data[0] = conv_param.stride_h();
            stride_data[1] = conv_param.stride_w();
        }
        
        else
        {
            const int num_stride_dims = conv_param.stride_size();
            CHECK (num_stride_dims == 0 || num_stride_dims == 1 ||
                   num_stride_dims == num_spatial_axes_)
                    << "stride must be specified once, or once per spatial dimension "
                    << "(stride specified " << num_stride_dims << " times; "
                    << num_spatial_axes_ << " spatial dims).";
            const int kDefaultStride = 1;
            
            for (int i = 0; i < num_spatial_axes_; ++i)
            {
                stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
                                 conv_param.stride ( (num_stride_dims == 1) ? 0 : i);
                CHECK_GT (stride_data[i], 0) << "Stride dimensions must be nonzero.";
            }
        }
        
        // Setup pad dimensions (pad_).
        pad_.Reshape (spatial_dim_blob_shape);
        int* pad_data = pad_.mutable_cpu_data();
        
        if (conv_param.has_pad_h() || conv_param.has_pad_w())
        {
            CHECK_EQ (num_spatial_axes_, 2)
                    << "pad_h & pad_w can only be used for 2D convolution.";
            CHECK_EQ (0, conv_param.pad_size())
                    << "Either pad or pad_h/w should be specified; not both.";
            pad_data[0] = conv_param.pad_h();
            pad_data[1] = conv_param.pad_w();
        }
        
        else
        {
            const int num_pad_dims = conv_param.pad_size();
            CHECK (num_pad_dims == 0 || num_pad_dims == 1 ||
                   num_pad_dims == num_spatial_axes_)
                    << "pad must be specified once, or once per spatial dimension "
                    << "(pad specified " << num_pad_dims << " times; "
                    << num_spatial_axes_ << " spatial dims).";
            const int kDefaultPad = 0;
            
            for (int i = 0; i < num_spatial_axes_; ++i)
            {
                pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
                              conv_param.pad ( (num_pad_dims == 1) ? 0 : i);
            }
        }
        
        // Setup dilation dimensions (dilation_).
        dilation_.Reshape (spatial_dim_blob_shape);
        //dilation_data的解释可以参考：http://blog.csdn.net/jiongnima/article/details/69487519
        int* dilation_data = dilation_.mutable_cpu_data();
        const int num_dilation_dims = conv_param.dilation_size();
        CHECK (num_dilation_dims == 0 || num_dilation_dims == 1 ||
               num_dilation_dims == num_spatial_axes_)
                << "dilation must be specified once, or once per spatial dimension "
                << "(dilation specified " << num_dilation_dims << " times; "
                << num_spatial_axes_ << " spatial dims).";
        const int kDefaultDilation = 1;
        
        for (int i = 0; i < num_spatial_axes_; ++i)
        {
            dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                               conv_param.dilation ( (num_dilation_dims == 1) ? 0 : i);
        }
        
        // Special case: im2col is the identity for 1x1 convolution with stride 1
        // and no padding, so flag for skipping the buffer and transformation.
        is_1x1_ = true;
        
        for (int i = 0; i < num_spatial_axes_; ++i)
        {
            is_1x1_ &= kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
            
            if (!is_1x1_) { break; }
        }
        
        // Configure output channels and groups.
        channels_ = bottom[0]->shape (channel_axis_);//获取卷积层输入的单blob的通道数
        num_output_ = this->layer_param_.convolution_param().num_output();//输出层 feature map 的个数
        CHECK_GT (num_output_, 0);
        group_ = this->layer_param_.convolution_param().group();
        CHECK_EQ (channels_ % group_, 0);
        CHECK_EQ (num_output_ % group_, 0)
                << "Number of output should be multiples of group.";
                
        if (reverse_dimensions()) //若需要反转卷积操作，则交换输入输出，否则不交换
        {
            conv_out_channels_ = channels_;
            conv_in_channels_ = num_output_;
        }
        
        else
        {
            conv_out_channels_ = num_output_;
            conv_in_channels_ = channels_;
        }
        
        // Handle the parameters: weights and biases.
        // - blobs_[0] holds the filter weights
        // - blobs_[1] holds the biases (optional)
        vector<int> weight_shape (2);//weight_shape一共4个变量
        weight_shape[0] = conv_out_channels_;//输出 feature map 的个数（num）
        // conv_in_channels_ 维输入层的通道数，如果 group_ 使用默认的值1，则weight的参数个数为：
        // 输出feature map的个数 * 输入通道的个数 * 一个卷积核参数的个数，即每一个输出feature map都对应了conv_in_channels_个卷积核
        // 如果group_不为1，则每一个输出feature map都对应了conv_in_channels_/group_个卷积核，这将减少权值的个数从而降低计算量
        weight_shape[1] = conv_in_channels_ / group_;//（channel）
        
        //设置后两维维卷积核的大小
        for (int i = 0; i < num_spatial_axes_; ++i)
        {
            weight_shape.push_back (kernel_shape_data[i]);
        }
        
        // 设置偏置层
        bias_term_ = this->layer_param_.convolution_param().bias_term();//是否使用偏置，默认为true
        vector<int> bias_shape (bias_term_, num_output_);
        
        // 初始化 weights 和 biases
        if (this->blobs_.size() > 0)
        {
            CHECK_EQ (1 + bias_term_, this->blobs_.size())
                    << "Incorrect number of weight blobs.";
                    
            if (weight_shape != this->blobs_[0]->shape())
            {
                Blob<Dtype> weight_shaped_blob (weight_shape);
                LOG (FATAL) << "Incorrect weight shape: expected shape "
                            << weight_shaped_blob.shape_string() << "; instead, shape was "
                            << this->blobs_[0]->shape_string();
            }
            
            if (bias_term_ && bias_shape != this->blobs_[1]->shape())
            {
                Blob<Dtype> bias_shaped_blob (bias_shape);
                LOG (FATAL) << "Incorrect bias shape: expected shape "
                            << bias_shaped_blob.shape_string() << "; instead, shape was "
                            << this->blobs_[1]->shape_string();
            }
            
            LOG (INFO) << "Skipping parameter initialization";
        }
        
        else
        {
            if (bias_term_)
            {
                this->blobs_.resize (2);
            }
            
            else
            {
                this->blobs_.resize (1);
            }
            
            // Initialize and fill the weights:
            // output channels x input channels per-group x kernel height x kernel width
            this->blobs_[0].reset (new Blob<Dtype> (weight_shape));
            shared_ptr<Filler<Dtype> > weight_filler (GetFiller<Dtype> (
                        this->layer_param_.convolution_param().weight_filler()));
            weight_filler->Fill (this->blobs_[0].get());
            
            // If necessary, initialize and fill the biases.
            if (bias_term_)
            {
                this->blobs_[1].reset (new Blob<Dtype> (bias_shape));
                shared_ptr<Filler<Dtype> > bias_filler (GetFiller<Dtype> (
                        this->layer_param_.convolution_param().bias_filler()));
                bias_filler->Fill (this->blobs_[1].get());
            }
        }
        
        //获取一个输出通道对应的所有卷积核对输入的一个卷积组所有通道操作一次处理数据量大小，为(输入总通道数/卷积组数)*卷积核高*卷积核宽
        //即计算某个feature map中某个点所需要的权值个数
        kernel_dim_ = this->blobs_[0]->count (1);//count(1)等价于count(1,blobs_.szie())，即求解每个 feature map 时需要的权值个数（需要考虑卷积层的权值共享）
        // 当group_大于1时所有weights要分group次才能完成计算，每次计算 weights 的个数为 weight_offset_
        weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;//获取权重的偏移量，理解为(conv_out_channels_/group_)* kernel_dim_
        // Propagate gradients to the parameters (as directed by backward pass).
        this->param_propagate_down_.resize (this->blobs_.size(), true);
    }
    
    template <typename Dtype>
    void BaseConvolutionLayer<Dtype>::Reshape (const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
    {
        const int first_spatial_axis = channel_axis_ + 1;//找到卷积操作处理的第一维的索引，通常为height
        CHECK_EQ (bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
                << "bottom num_axes may not change.";
        num_ = bottom[0]->count (0, channel_axis_);//获取卷积层操作输入的图片数目
        CHECK_EQ (bottom[0]->shape (channel_axis_), channels_)
                << "Input size incompatible with convolution kernel.";
                
        // TODO: generalize to handle inputs of different shapes.
        // 如果输入多个blob的话，检查所有blob是否具有相同的shape
        for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id)
        {
            CHECK (bottom[0]->shape() == bottom[bottom_id]->shape())
                    << "All inputs must have the same shape.";
        }
        
        // Shape the tops.
        bottom_shape_ = &bottom[0]->shape();
        compute_output_shape();
        vector<int> top_shape (bottom[0]->shape().begin(), bottom[0]->shape().begin() + channel_axis_);
        top_shape.push_back (num_output_);
        
        for (int i = 0; i < num_spatial_axes_; ++i)
        {
            top_shape.push_back (output_shape_[i]);
        }
        
        for (int top_id = 0; top_id < top.size(); ++top_id)
        {
            top[top_id]->Reshape (top_shape);
        }
        
        /*如果要反转卷积操作，conv_out_spatial_dim_初始化为卷积层输出单个blob（feature map）的单通道的数据量*/
        if (reverse_dimensions())
        {
            conv_out_spatial_dim_ = bottom[0]->count (first_spatial_axis);
        }
        
        /*否则，conv_out_spatial_dim_初始化为卷积层输出单位blob(top[0])的单通道的数据量 */
        else
        {
            conv_out_spatial_dim_ = top[0]->count (first_spatial_axis);
        }
        
        // col_offset 表征了一个输出通道对应的所有卷积核处理的一个卷积组的所有数据量
        // 求一个feature map所需要的乘法次数（权值相关）
        // 以cifar10的conv1层为例其输出层的feature map为32*32=1024，即conv_out_spatial_dim_的值在cifar conv1中为1024
        col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
        // output_offset_ 表征了一个卷积组输出的所有数据量
        output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
        // Setup input dimensions (conv_input_shape_).
        // 用于初始化卷积操作输入数据的形状，一般三维(C,H,W)
        vector<int> bottom_dim_blob_shape (1, num_spatial_axes_ + 1);
        conv_input_shape_.Reshape (bottom_dim_blob_shape);// 初始化卷积层输入shape，一般大小为3
        int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
        
        //初始化卷积层的输入参数，一般顺序为channel->height->width
        for (int i = 0; i < num_spatial_axes_ + 1; ++i)
        {
            if (reverse_dimensions())
            {
                conv_input_shape_data[i] = top[0]->shape (channel_axis_ + i);
            }
            
            else
            {
                conv_input_shape_data[i] = bottom[0]->shape (channel_axis_ + i);
            }
        }
        
        // The im2col result buffer will only hold one image at a time to avoid
        // overly large memory usage. In the special case of 1x1 convolution
        // it goes lazily unused to save memory.
        col_buffer_shape_.clear();
        //col_buffer_shape_加入(输入总通道数*卷积核高*卷积核宽)
        col_buffer_shape_.push_back (kernel_dim_ * group_);
        
        //col_buffer_shape_加入卷积层输出单通道的维度
        for (int i = 0; i < num_spatial_axes_; ++i)
        {
            if (reverse_dimensions())
            {
                col_buffer_shape_.push_back (input_shape (i + 1));
            }
            
            else
            {
                col_buffer_shape_.push_back (output_shape_[i]);
            }
        }
        
        //初始化col_buffer
        col_buffer_.Reshape (col_buffer_shape_);//以cifar10 conv1为例，col_buffer_shape_为(75,32,32)，75为kernel_dim_
        //bottom_dim_描述的是bottom blob的一个channel包含的数据量
        bottom_dim_ = bottom[0]->count (channel_axis_);
        //top_dim_描述的是top blob的一个channel包含的数据量
        top_dim_ = top[0]->count (channel_axis_);
        //描述了一个输出通道对应的所有卷积核对全部输入做卷积操作时转换生成的列向量的数量
        num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
        //描述了将生成的列向量还原卷积操作的区域图的数量
        num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
        // Set up the all ones "bias multiplier" for adding biases by BLAS
        out_spatial_dim_ = top[0]->count (first_spatial_axis);
        
        //描述了输出的单通道数据量
        if (bias_term_) //若启用了偏置，那么初始化偏置乘数blob
        {
            //偏置乘数的大小为输出的单通道数据量，因为对于每个输出数据乘数不一样
            vector<int> bias_multiplier_shape (1, out_spatial_dim_);
            bias_multiplier_.Reshape (bias_multiplier_shape);
            caffe_set (bias_multiplier_.count(), Dtype (1),
                       bias_multiplier_.mutable_cpu_data());
        }
    }
    
    template <typename Dtype>
    void BaseConvolutionLayer<Dtype>::forward_cpu_gemm (const Dtype* input,
            const Dtype* weights, Dtype* output, bool skip_im2col)
    {
        const Dtype* col_buff = input;
        
        if (!is_1x1_)
        {
            if (!skip_im2col)
            {
                conv_im2col_cpu (input, col_buffer_.mutable_cpu_data());
            }
            
            col_buff = col_buffer_.cpu_data();
        }
        
        // 当group_参数大于1时每次计算总输出的 1/gropu_，所以需要group_次计算才能实现所有输出的计算
        for (int g = 0; g < group_; ++g)
        {
            //C=alpha*A*B+beta*C
            caffe_cpu_gemm<Dtype> (CblasNoTrans, CblasNoTrans,//参加相乘的两个矩阵都不转置
                                   conv_out_channels_ / group_,//结果矩阵的行数
                                   conv_out_spatial_dim_,//结果矩阵的列数，一个feature map中像素点的个数
                                   kernel_dim_,//A*B，左边矩阵的列数（右边矩阵的行数），求解feature map中的一个点时需要权值的个数
                                   (Dtype) 1.,//alpha
                                   weights + weight_offset_ * g,//A
                                   col_buff + col_offset_ * g,//B
                                   (Dtype) 0.,//beta
                                   output + output_offset_ * g//保存位置的起点，也是C的源
                                  );
        }
    }
    
    template <typename Dtype>
    void BaseConvolutionLayer<Dtype>::forward_cpu_bias (Dtype* output,
            const Dtype* bias)
    {
        caffe_cpu_gemm<Dtype> (CblasNoTrans, CblasNoTrans, num_output_,
                               out_spatial_dim_, 1, (Dtype) 1., bias, bias_multiplier_.cpu_data(),
                               (Dtype) 1., output);
    }
    
    template <typename Dtype>
    void BaseConvolutionLayer<Dtype>::backward_cpu_gemm (const Dtype* output,
            const Dtype* weights, Dtype* input)
    {
        Dtype* col_buff = col_buffer_.mutable_cpu_data();
        
        if (is_1x1_)
        {
            col_buff = input;
        }
        
        for (int g = 0; g < group_; ++g)
        {
            caffe_cpu_gemm<Dtype> (CblasTrans, CblasNoTrans, kernel_dim_,
                                   conv_out_spatial_dim_, conv_out_channels_ / group_,
                                   (Dtype) 1., weights + weight_offset_ * g, output + output_offset_ * g,
                                   (Dtype) 0., col_buff + col_offset_ * g);
        }
        
        if (!is_1x1_)
        {
            conv_col2im_cpu (col_buff, input);
        }
    }
    
    template <typename Dtype>
    void BaseConvolutionLayer<Dtype>::weight_cpu_gemm (const Dtype* input,
            const Dtype* output, Dtype* weights)
    {
        const Dtype* col_buff = input;
        
        if (!is_1x1_)
        {
            conv_im2col_cpu (input, col_buffer_.mutable_cpu_data());
            col_buff = col_buffer_.cpu_data();
        }
        
        for (int g = 0; g < group_; ++g)
        {
            caffe_cpu_gemm<Dtype> (CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
                                   kernel_dim_, conv_out_spatial_dim_,
                                   (Dtype) 1., output + output_offset_ * g, col_buff + col_offset_ * g,
                                   (Dtype) 1., weights + weight_offset_ * g);
        }
    }
    
    template <typename Dtype>
    void BaseConvolutionLayer<Dtype>::backward_cpu_bias (Dtype* bias,
            const Dtype* input)
    {
        caffe_cpu_gemv<Dtype> (CblasNoTrans, num_output_, out_spatial_dim_, 1.,
                               input, bias_multiplier_.cpu_data(), 1., bias);
    }
    
#ifndef CPU_ONLY
    
    template <typename Dtype>
    void BaseConvolutionLayer<Dtype>::forward_gpu_gemm (const Dtype* input,
            const Dtype* weights, Dtype* output, bool skip_im2col)
    {
        const Dtype* col_buff = input;
        
        if (!is_1x1_)
        {
            if (!skip_im2col)
            {
                conv_im2col_gpu (input, col_buffer_.mutable_gpu_data());
            }
            
            col_buff = col_buffer_.gpu_data();
        }
        
        for (int g = 0; g < group_; ++g)
        {
            caffe_gpu_gemm<Dtype> (CblasNoTrans, CblasNoTrans, conv_out_channels_ /
                                   group_, conv_out_spatial_dim_, kernel_dim_,
                                   (Dtype) 1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
                                   (Dtype) 0., output + output_offset_ * g);
        }
    }
    
    template <typename Dtype>
    void BaseConvolutionLayer<Dtype>::forward_gpu_bias (Dtype* output,
            const Dtype* bias)
    {
        caffe_gpu_gemm<Dtype> (CblasNoTrans, CblasNoTrans, num_output_,
                               out_spatial_dim_, 1, (Dtype) 1., bias, bias_multiplier_.gpu_data(),
                               (Dtype) 1., output);
    }
    
    template <typename Dtype>
    void BaseConvolutionLayer<Dtype>::backward_gpu_gemm (const Dtype* output,
            const Dtype* weights, Dtype* input)
    {
        Dtype* col_buff = col_buffer_.mutable_gpu_data();
        
        if (is_1x1_)
        {
            col_buff = input;
        }
        
        for (int g = 0; g < group_; ++g)
        {
            caffe_gpu_gemm<Dtype> (CblasTrans, CblasNoTrans, kernel_dim_,
                                   conv_out_spatial_dim_, conv_out_channels_ / group_,
                                   (Dtype) 1., weights + weight_offset_ * g, output + output_offset_ * g,
                                   (Dtype) 0., col_buff + col_offset_ * g);
        }
        
        if (!is_1x1_)
        {
            conv_col2im_gpu (col_buff, input);
        }
    }
    
    template <typename Dtype>
    void BaseConvolutionLayer<Dtype>::weight_gpu_gemm (const Dtype* input,
            const Dtype* output, Dtype* weights)
    {
        const Dtype* col_buff = input;
        
        if (!is_1x1_)
        {
            conv_im2col_gpu (input, col_buffer_.mutable_gpu_data());
            col_buff = col_buffer_.gpu_data();
        }
        
        for (int g = 0; g < group_; ++g)
        {
            caffe_gpu_gemm<Dtype> (CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
                                   kernel_dim_, conv_out_spatial_dim_,
                                   (Dtype) 1., output + output_offset_ * g, col_buff + col_offset_ * g,
                                   (Dtype) 1., weights + weight_offset_ * g);
        }
    }
    
    template <typename Dtype>
    void BaseConvolutionLayer<Dtype>::backward_gpu_bias (Dtype* bias,
            const Dtype* input)
    {
        caffe_gpu_gemv<Dtype> (CblasNoTrans, num_output_, out_spatial_dim_, 1.,
                               input, bias_multiplier_.gpu_data(), 1., bias);
    }
    
#endif  // !CPU_ONLY
    
    INSTANTIATE_CLASS (BaseConvolutionLayer);
    
}  // namespace caffe
