#include <vector>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

    // Function uses casting from int to unsigned to compare if value of
    // parameter a is greater or equal to zero and lower than value of
    // parameter b. The b parameter is of type signed and is always positive,
    // therefore its value is always lower than 0x800... where casting
    // negative value of a parameter converts it to value higher than 0x800...
    // The casting allows to use one condition instead of two.
    //若a大于0且严格小于b，则返回真，否则返回假，该函数的作用是判断矩阵上某元的输出是否为pad的0。
    inline bool is_a_ge_zero_and_a_lt_b (int a, int b)
    {
        return static_cast<unsigned> (a) < static_cast<unsigned> (b);
    }
    
    // 以RGB图像为例：
    // data_col最高维为3，代表着每个通道的信息
    // 设dilation_w为1，且padding的设置使得feature map大小与图像大小相同
    // 那么下一维大小为height * width，即每一个点对应一个卷积核
	// 再下一维大小为为kernel_h * kernel_w（参加卷积时对应位置原图的一个信息）
    template <typename Dtype>
    void im2col_cpu (const Dtype* data_im,//图片数据
                     const int channels,//图片的通道数
                     const int height, const int width,//图片的几何信息
                     const int kernel_h, const int kernel_w,//卷积核的几何信息
                     const int pad_h, const int pad_w,//图像边缘的padding
                     const int stride_h, const int stride_w,//滑动步长
                     const int dilation_h, const int dilation_w,//图像下采样参数（具体见下面代码）
                     Dtype* data_col//输出位置
                    )
    {
        //输出矩阵的行数与宽数
        const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
        const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
        const int channel_size = height * width;
        
        for (int channel = channels; channel--; data_im += channel_size) //通道
        {
            for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) //卷积核的行
            {
                for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) //核的列
                {
                    // dilation_h这个变量是每隔多少个像素取值，比如dilation_h=2，图像下采样
                    // 一般dialtion_h为 1，
                    // 这里input_row取值从-pad_h到output_h+pad_h，多个通道依次排放，
                    // 每个通道从上到下从左到右以卷积核为基本单元展开写进内存中
                    int input_row = -pad_h + kernel_row * dilation_h;
                    
                    for (int output_rows = output_h; output_rows; output_rows--)
                    {
                        //首先判断将要写行位置是不是padding，如果是这些(#`O′)直接置0即可
                        if (!is_a_ge_zero_and_a_lt_b (input_row, height))
                        {
                            for (int output_cols = output_w; output_cols; output_cols--)
                            {
                                * (data_col++) = 0;
                            }
                        }
                        
                        else
                        {
                            int input_col = -pad_w + kernel_col * dilation_w;
                            
                            for (int output_col = output_w; output_col; output_col--)
                            {
                                //同样要判断写的列是不是padding
                                if (is_a_ge_zero_and_a_lt_b (input_col, width))
                                {
                                    * (data_col++) = data_im[input_row * width + input_col];
                                }
                                
                                else
                                {
                                    * (data_col++) = 0;
                                }
                                
                                input_col += stride_w;
                            }
                        }
                        
                        input_row += stride_h;
                    }
                }
            }
        }
    }
    
    // Explicit instantiation
    template void im2col_cpu<float> (const float* data_im, const int channels,
                                     const int height, const int width, const int kernel_h, const int kernel_w,
                                     const int pad_h, const int pad_w, const int stride_h,
                                     const int stride_w, const int dilation_h, const int dilation_w,
                                     float* data_col);
    template void im2col_cpu<double> (const double* data_im, const int channels,
                                      const int height, const int width, const int kernel_h, const int kernel_w,
                                      const int pad_h, const int pad_w, const int stride_h,
                                      const int stride_w, const int dilation_h, const int dilation_w,
                                      double* data_col);
                                      
    template <typename Dtype>
    inline void im2col_nd_core_cpu (const Dtype* data_input, const bool im2col,
                                    const int num_spatial_axes, const int* im_shape, const int* col_shape,
                                    const int* kernel_shape, const int* pad, const int* stride,
                                    const int* dilation, Dtype* data_output)
    {
        if (!im2col)
        {
            int im_size = im_shape[0];
            
            for (int i = 0; i < num_spatial_axes; ++i)
            {
                im_size *= im_shape[1 + i];
            }
            
            caffe_set (im_size, Dtype (0), data_output);
        }
        
        int kernel_size = 1;
        
        for (int i = 0; i < num_spatial_axes; ++i)
        {
            kernel_size *= kernel_shape[i];
        }
        
        const int channels_col = col_shape[0];
        vector<int> d_offset (num_spatial_axes, 0);
        vector<int> d_iter (num_spatial_axes, 0);
        
        for (int c_col = 0; c_col < channels_col; ++c_col)
        {
            // Loop over spatial axes in reverse order to compute a per-axis offset.
            int offset = c_col;
            
            for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i)
            {
                if (d_i < num_spatial_axes - 1)
                {
                    offset /= kernel_shape[d_i + 1];
                }
                
                d_offset[d_i] = offset % kernel_shape[d_i];
            }
            
            for (bool incremented = true; incremented;)
            {
                // Loop over spatial axes in forward order to compute the indices in the
                // image and column, and whether the index lies in the padding.
                int index_col = c_col;
                int index_im = c_col / kernel_size;
                bool is_padding = false;
                
                for (int d_i = 0; d_i < num_spatial_axes; ++d_i)
                {
                    const int d = d_iter[d_i];
                    const int d_im = d * stride[d_i] - pad[d_i] +
                                     d_offset[d_i] * dilation[d_i];
                    is_padding |= d_im < 0 || d_im >= im_shape[d_i + 1];
                    index_col *= col_shape[d_i + 1];
                    index_col += d;
                    index_im *= im_shape[d_i + 1];
                    index_im += d_im;
                }
                
                if (im2col)
                {
                    if (is_padding)
                    {
                        data_output[index_col] = 0;
                    }
                    
                    else
                    {
                        data_output[index_col] = data_input[index_im];
                    }
                }
                
                else
                    if (!is_padding)    // col2im
                    {
                        data_output[index_im] += data_input[index_col];
                    }
                    
                // Loop over spatial axes in reverse order to choose an index,
                // like counting.
                incremented = false;
                
                for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i)
                {
                    const int d_max = col_shape[d_i + 1];
                    DCHECK_LT (d_iter[d_i], d_max);
                    
                    if (d_iter[d_i] == d_max - 1)
                    {
                        d_iter[d_i] = 0;
                    }
                    
                    else      // d_iter[d_i] < d_max - 1
                    {
                        ++d_iter[d_i];
                        incremented = true;
                        break;
                    }
                }
            }  // while(incremented) {
        }  // for (int c = 0; c < channels_col; ++c) {
    }
    
    template <typename Dtype>
    void im2col_nd_cpu (const Dtype* data_im, const int num_spatial_axes,
                        const int* im_shape, const int* col_shape,
                        const int* kernel_shape, const int* pad, const int* stride,
                        const int* dilation, Dtype* data_col)
    {
        const bool kIm2Col = true;
        im2col_nd_core_cpu (data_im, kIm2Col, num_spatial_axes, im_shape, col_shape,
                            kernel_shape, pad, stride, dilation, data_col);
    }
    
    // Explicit instantiation
    template void im2col_nd_cpu<float> (const float* data_im,
                                        const int num_spatial_axes,
                                        const int* im_shape, const int* col_shape,
                                        const int* kernel_shape, const int* pad, const int* stride,
                                        const int* dilation, float* data_col);
    template void im2col_nd_cpu<double> (const double* data_im,
                                         const int num_spatial_axes,
                                         const int* im_shape, const int* col_shape,
                                         const int* kernel_shape, const int* pad, const int* stride,
                                         const int* dilation, double* data_col);
                                         
    template <typename Dtype>
    void col2im_cpu (const Dtype* data_col, const int channels,
                     const int height, const int width, const int kernel_h, const int kernel_w,
                     const int pad_h, const int pad_w,
                     const int stride_h, const int stride_w,
                     const int dilation_h, const int dilation_w,
                     Dtype* data_im)
    {
        caffe_set (height * width * channels, Dtype (0), data_im);
        const int output_h = (height + 2 * pad_h -
                              (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
        const int output_w = (width + 2 * pad_w -
                              (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
        const int channel_size = height * width;
        
        for (int channel = channels; channel--; data_im += channel_size)
        {
            for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++)
            {
                for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++)
                {
                    int input_row = -pad_h + kernel_row * dilation_h;
                    
                    for (int output_rows = output_h; output_rows; output_rows--)
                    {
                        if (!is_a_ge_zero_and_a_lt_b (input_row, height))
                        {
                            data_col += output_w;
                        }
                        
                        else
                        {
                            int input_col = -pad_w + kernel_col * dilation_w;
                            
                            for (int output_col = output_w; output_col; output_col--)
                            {
                                if (is_a_ge_zero_and_a_lt_b (input_col, width))
                                {
                                    data_im[input_row * width + input_col] += *data_col;
                                }
                                
                                data_col++;
                                input_col += stride_w;
                            }
                        }
                        
                        input_row += stride_h;
                    }
                }
            }
        }
    }
    
    // Explicit instantiation
    template void col2im_cpu<float> (const float* data_col, const int channels,
                                     const int height, const int width, const int kernel_h, const int kernel_w,
                                     const int pad_h, const int pad_w, const int stride_h,
                                     const int stride_w, const int dilation_h, const int dilation_w,
                                     float* data_im);
    template void col2im_cpu<double> (const double* data_col, const int channels,
                                      const int height, const int width, const int kernel_h, const int kernel_w,
                                      const int pad_h, const int pad_w, const int stride_h,
                                      const int stride_w, const int dilation_h, const int dilation_w,
                                      double* data_im);
                                      
    template <typename Dtype>
    void col2im_nd_cpu (const Dtype* data_col, const int num_spatial_axes,
                        const int* im_shape, const int* col_shape,
                        const int* kernel_shape, const int* pad, const int* stride,
                        const int* dilation, Dtype* data_im)
    {
        const bool kIm2Col = false;
        im2col_nd_core_cpu (data_col, kIm2Col, num_spatial_axes, im_shape, col_shape,
                            kernel_shape, pad, stride, dilation, data_im);
    }
    
    // Explicit instantiation
    template void col2im_nd_cpu<float> (const float* data_col,
                                        const int num_spatial_axes,
                                        const int* im_shape, const int* col_shape,
                                        const int* kernel_shape, const int* pad, const int* stride,
                                        const int* dilation, float* data_im);
    template void col2im_nd_cpu<double> (const double* data_col,
                                         const int num_spatial_axes,
                                         const int* im_shape, const int* col_shape,
                                         const int* kernel_shape, const int* pad, const int* stride,
                                         const int* dilation, double* data_im);
                                         
                                         
}  // namespace caffe
