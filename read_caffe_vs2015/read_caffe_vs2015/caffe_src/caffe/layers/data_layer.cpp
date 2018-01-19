#ifdef USE_OPENCV
    #include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe
{

    template <typename Dtype>
    DataLayer<Dtype>::DataLayer (const LayerParameter& param)
        : BasePrefetchingDataLayer<Dtype> (param), offset_()
    {
        //设置数据读取对象，不同格式的数据使用不同的对象读取，这里使用LMDB
        db_.reset (db::GetDB (param.data_param().backend()));
        db_->Open (param.data_param().source(), db::READ);
        //类似于fseek，设置文件的游标
        cursor_.reset (db_->NewCursor());
    }
    
    template <typename Dtype>
    DataLayer<Dtype>::~DataLayer()
    {
        this->StopInternalThread();
    }
    
    template <typename Dtype>
    void DataLayer<Dtype>::DataLayerSetUp (const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top)
    {
        const int batch_size = this->layer_param_.data_param().batch_size();
        // Read a data point, and use it to initialize the top blob.
        Datum datum;
        datum.ParseFromString (cursor_->value());
        // Use data_transformer to infer the expected blob shape from datum.
        // 对于数据层而言其是没有bottom的，以lenet为例，经过下一行这里的top_shape为{1,1,28,28}，下面会修改batch_size
        // transfer对于data层而言指crop、padding等操作
        vector<int> top_shape = this->data_transformer_->InferBlobShape (datum);
        this->transformed_data_.Reshape (top_shape);
        // Reshape top[0] and prefetch_data according to the batch_size.
        top_shape[0] = batch_size;
        top[0]->Reshape (top_shape);
        
        //prefetch应该用于从硬盘中预读训练数据，prefetch中的基本元素的size应该是batch_size
        for (int i = 0; i < this->prefetch_.size(); ++i) //这个prefetch_的大小似乎由其他地方设定，一般按照系统的吞吐能力自动设置
        {
            this->prefetch_[i]->data_.Reshape (top_shape);
        }
        
        LOG_IF (INFO, Caffe::root_solver())
                << "output data size: " << top[0]->num() << ","
                << top[0]->channels() << "," << top[0]->height() << ","
                << top[0]->width();
                
        // label
        if (this->output_labels_)
        {
            vector<int> label_shape (1, batch_size);
            top[1]->Reshape (label_shape);
            
            for (int i = 0; i < this->prefetch_.size(); ++i)
            {
                this->prefetch_[i]->label_.Reshape (label_shape);
            }
        }
    }
    
    template <typename Dtype>
    bool DataLayer<Dtype>::Skip()
    {
        int size = Caffe::solver_count();
        int rank = Caffe::solver_rank();
        bool keep = (offset_ % size) == rank ||
                    // In test mode, only rank 0 runs, so avoid skipping
                    this->layer_param_.phase() == TEST;
        return !keep;
    }
    
    template<typename Dtype>
    void DataLayer<Dtype>::Next()
    {
        cursor_->Next();
        
        if (!cursor_->valid())
        {
            LOG_IF (INFO, Caffe::root_solver())
                    << "Restarting data prefetching from start.";
            cursor_->SeekToFirst();
        }
        
        offset_++;
    }
    
    // This function is called on prefetch thread
    template<typename Dtype>
    void DataLayer<Dtype>::load_batch (Batch<Dtype>* batch)
    {
        CPUTimer batch_timer;
        batch_timer.Start();
        double read_time = 0;
        double trans_time = 0;
        CPUTimer timer;
        CHECK (batch->data_.count());
        CHECK (this->transformed_data_.count());
        const int batch_size = this->layer_param_.data_param().batch_size();
        Datum datum;
        
        for (int item_id = 0; item_id < batch_size; ++item_id)
        {
            timer.Start();
            
            while (Skip())
            {
                Next();
            }
            
            datum.ParseFromString (cursor_->value());
            read_time += timer.MicroSeconds();
            
            if (item_id == 0)
            {
                // Reshape according to the first datum of each batch
                // on single input batches allows for inputs of varying dimension.
                // Use data_transformer to infer the expected blob shape from datum.
                vector<int> top_shape = this->data_transformer_->InferBlobShape (datum);
                this->transformed_data_.Reshape (top_shape);
                // Reshape batch according to the batch_size.
                top_shape[0] = batch_size;
                batch->data_.Reshape (top_shape);
            }
            
            // Apply data transformations (mirror, scale, crop...)
            timer.Start();
            int offset = batch->data_.offset (item_id);
            Dtype* top_data = batch->data_.mutable_cpu_data();
            this->transformed_data_.set_cpu_data (top_data + offset);
            this->data_transformer_->Transform (datum, & (this->transformed_data_));
            
            // Copy label.
            if (this->output_labels_)
            {
                Dtype* top_label = batch->label_.mutable_cpu_data();
                top_label[item_id] = datum.label();
            }
            
            trans_time += timer.MicroSeconds();
            Next();
        }
        
        timer.Stop();
        batch_timer.Stop();
        DLOG (INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
        DLOG (INFO) << "     Read time: " << read_time / 1000 << " ms.";
        DLOG (INFO) << "Transform time: " << trans_time / 1000 << " ms.";
    }
    
    INSTANTIATE_CLASS (DataLayer);
    REGISTER_LAYER_CLASS (Data);
    
}  // namespace caffe
