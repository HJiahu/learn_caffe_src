#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe
{

    /**
     * @brief Abstract base class that factors out the BLAS code common to
     *        ConvolutionLayer and DeconvolutionLayer.
     */
    template <typename Dtype>
    class BaseConvolutionLayer : public Layer<Dtype>
    {
        public:
            explicit BaseConvolutionLayer (const LayerParameter& param)
                : Layer<Dtype> (param) {}
            virtual void LayerSetUp (const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
            virtual void Reshape (const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
            
            virtual inline int MinBottomBlobs() const { return 1; }
            virtual inline int MinTopBlobs() const { return 1; }
            virtual inline bool EqualNumBottomTopBlobs() const { return true; }
            
        protected:
            // Helper functions that abstract away the column buffer and gemm arguments.
            // The last argument in forward_cpu_gemm is so that we can skip the im2col if
            // we just called weight_cpu_gemm with the same input.
            // gemm的全称是：General Matrix Matrix Multiply，是矩阵与矩阵相乘的运算
            void forward_cpu_gemm (const Dtype* input, const Dtype* weights,  Dtype* output, bool skip_im2col = false);
            void forward_cpu_bias (Dtype* output, const Dtype* bias);
            void backward_cpu_gemm (const Dtype* input, const Dtype* weights, Dtype* output);
            void weight_cpu_gemm (const Dtype* input, const Dtype* output, Dtype* weights);
            void backward_cpu_bias (Dtype* bias, const Dtype* input);
            
#ifndef CPU_ONLY
            void forward_gpu_gemm (const Dtype* col_input, const Dtype* weights,
                                   Dtype* output, bool skip_im2col = false);
            void forward_gpu_bias (Dtype* output, const Dtype* bias);
            void backward_gpu_gemm (const Dtype* input, const Dtype* weights,
                                    Dtype* col_output);
            void weight_gpu_gemm (const Dtype* col_input, const Dtype* output, Dtype*
                                  weights);
            void backward_gpu_bias (Dtype* bias, const Dtype* input);
#endif
            
            /// @brief The spatial dimensions of the input.
            // 这个函数用于卷积层的输入blob的高(height)和宽(width)，注意参数i从1开始取，代表从channel的后一维开始
            inline int input_shape (int i)
            {
                return (*bottom_shape_) [channel_axis_ + i];
            }
            // reverse_dimensions should return true iff we are implementing deconv, so
            // that conv helpers know which dimensions are which.
            virtual bool reverse_dimensions() = 0;//是否执行反卷积
            // Compute height_out_ and width_out_ from other parameters.
            virtual void compute_output_shape() = 0;
            
            /// @brief The spatial dimensions of a filter kernel.
            Blob<int> kernel_shape_; //卷积核的形状，长*宽，如果size为1则说明长宽一致
            /// @brief The spatial dimensions of the stride.
            Blob<int> stride_;
            /// @brief The spatial dimensions of the padding.
            Blob<int> pad_;
            /// @brief The spatial dimensions of the dilation.
            Blob<int> dilation_;//描述卷积核的膨胀参数
            /// @brief The spatial dimensions of the convolution input.
            Blob<int> conv_input_shape_;
            /// @brief The spatial dimensions of the col_buffer.
            vector<int> col_buffer_shape_;//一个输出通道对应的所有卷积核的所有卷积区域转化成一列向量的形状
            /// @brief The spatial dimensions of the output.
            vector<int> output_shape_;
            const vector<int>* bottom_shape_;//这个指针指向了输入blob的shape
            
            int num_spatial_axes_;//这个参数描述的是卷积处理的维度数，一般为2，表示二维卷积
            int bottom_dim_;//bottom_dim_描述的是bottom blob的一个channel包含的数据量
            int top_dim_;//top_dim_描述的是top blob的一个channel包含的数据量
            
            int channel_axis_;//这个参数一般为1，指示卷积核是按照输入blob各通道进行卷积
            int num_;//这个参数代表卷积操作输入图片的数目
            int channels_;//代表卷积层输入的单blob的通道数
            int group_;
            int out_spatial_dim_;//卷积后的图像大小
            int weight_offset_;//权重的偏移量，尤其适用于卷积组大于1的情况
            int num_output_;//表示该卷积层输出的通道数
            bool bias_term_;//是否启用偏置，默认为true
            bool is_1x1_;//判断是不是1*1卷积，要求卷积核为1*1，步长为1，pad为0
            bool force_nd_im2col_;//是否需要强制n维卷积
            
        private:
            // wrap im2col/col2im so we don't have to remember the (long) argument lists
            // 下面两个函数是caffe中计算卷积的辅助函数，分别将图片转化为矩阵和其逆运算
            // 这样可以使得整个一层的卷积运算可以一次计算完成
            // 具体解释可以参考：
            inline void conv_im2col_cpu (const Dtype* data, Dtype* col_buff)
            {
                if (!force_nd_im2col_ && num_spatial_axes_ == 2)
                {
                    im2col_cpu (data, conv_in_channels_,
                                conv_input_shape_.cpu_data() [1], conv_input_shape_.cpu_data() [2],
                                kernel_shape_.cpu_data() [0], kernel_shape_.cpu_data() [1],
                                pad_.cpu_data() [0], pad_.cpu_data() [1],
                                stride_.cpu_data() [0], stride_.cpu_data() [1],
                                dilation_.cpu_data() [0], dilation_.cpu_data() [1], col_buff);
                }
                
                else
                {
                    im2col_nd_cpu (data, num_spatial_axes_, conv_input_shape_.cpu_data(),
                                   col_buffer_shape_.data(), kernel_shape_.cpu_data(),
                                   pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
                }
            }
            inline void conv_col2im_cpu (const Dtype* col_buff, Dtype* data)
            {
                if (!force_nd_im2col_ && num_spatial_axes_ == 2)
                {
                    col2im_cpu (col_buff, conv_in_channels_,
                                conv_input_shape_.cpu_data() [1], conv_input_shape_.cpu_data() [2],
                                kernel_shape_.cpu_data() [0], kernel_shape_.cpu_data() [1],
                                pad_.cpu_data() [0], pad_.cpu_data() [1],
                                stride_.cpu_data() [0], stride_.cpu_data() [1],
                                dilation_.cpu_data() [0], dilation_.cpu_data() [1], data);
                }
                
                else
                {
                    col2im_nd_cpu (col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
                                   col_buffer_shape_.data(), kernel_shape_.cpu_data(),
                                   pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
                }
            }
#ifndef CPU_ONLY
            inline void conv_im2col_gpu (const Dtype* data, Dtype* col_buff)
            {
                if (!force_nd_im2col_ && num_spatial_axes_ == 2)
                {
                    im2col_gpu (data, conv_in_channels_,
                                conv_input_shape_.cpu_data() [1], conv_input_shape_.cpu_data() [2],
                                kernel_shape_.cpu_data() [0], kernel_shape_.cpu_data() [1],
                                pad_.cpu_data() [0], pad_.cpu_data() [1],
                                stride_.cpu_data() [0], stride_.cpu_data() [1],
                                dilation_.cpu_data() [0], dilation_.cpu_data() [1], col_buff);
                }
                
                else
                {
                    im2col_nd_gpu (data, num_spatial_axes_, num_kernels_im2col_,
                                   conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
                                   kernel_shape_.gpu_data(), pad_.gpu_data(),
                                   stride_.gpu_data(), dilation_.gpu_data(), col_buff);
                }
            }
            inline void conv_col2im_gpu (const Dtype* col_buff, Dtype* data)
            {
                if (!force_nd_im2col_ && num_spatial_axes_ == 2)
                {
                    col2im_gpu (col_buff, conv_in_channels_,
                                conv_input_shape_.cpu_data() [1], conv_input_shape_.cpu_data() [2],
                                kernel_shape_.cpu_data() [0], kernel_shape_.cpu_data() [1],
                                pad_.cpu_data() [0], pad_.cpu_data() [1],
                                stride_.cpu_data() [0], stride_.cpu_data() [1],
                                dilation_.cpu_data() [0], dilation_.cpu_data() [1], data);
                }
                
                else
                {
                    col2im_nd_gpu (col_buff, num_spatial_axes_, num_kernels_col2im_,
                                   conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
                                   kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
                                   dilation_.gpu_data(), data);
                }
            }
#endif
            
            int num_kernels_im2col_;//im2col操作生成的列向量数量
            int num_kernels_col2im_; //col2im操作还原得到的卷积操作处理的的小窗的数量
            int conv_out_channels_;//描述卷积层输出的通道数
            int conv_in_channels_;//描述卷积层输入的通道数
            int conv_out_spatial_dim_;//卷积操作输出的单通道数据量
            int kernel_dim_;//表示一个输出通道对应的所有卷积核对输入的一个卷积组的所有通道卷积操作一次处理数据量大小
            int col_offset_;//表示一个输出通道对应的所有卷积核处理的一个卷积组的所有数据量
            int output_offset_;//表示一个卷积组输出的所有数据量
            
            Blob<Dtype> col_buffer_;//存储了一个输出通道对应的所有卷积核的所有卷积区域转化成的众多单列向量
            Blob<Dtype> bias_multiplier_;//存储了偏置乘数
    };
    
}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
