#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }

namespace caffe
{

    /**
     * Virtual class encapsulate boost::thread for use in base class
     * The child class will acquire the ability to run a single thread,
     * by reimplementing the virtual function InternalThreadEntry.
     */
    //boost thread的封装，一个对象中有一个执行线程，可启动（start）可中断（stop）
    class InternalThread
    {
        public:
            InternalThread() : thread_() {}
            virtual ~InternalThread();
            
            /**
             * Caffe's thread local state will be initialized using the current
             * thread values, e.g. device id, solver index etc. The random seed
             * is initialized using caffe_rng_rand.
             */
            void StartInternalThread();
            
            /** Will not return until the internal thread has exited. */
            //这里使用了boost中提供的函数interrupt来中断一个线程，
            //在C++11 中不提供interrupt函数，这是在考虑异常的情况下做出的决定，如果需要需要自己实现
            /*
              boost中实现thread stop的方法是“标记法”，在线程的某些阶段（如sleep、wait）等函数中
              插入一个判断标志，标志为真则退出线程，所谓的某些阶段也可以自己定义，即在线程代码中插入
              boost提供的interruption_point()，interruption_point将自动判断标志并动作。在boost
              中可使用成员函数interrupt将对应的标志位设为真
            */
            void StopInternalThread();
            
            bool is_started() const;
            
        protected:
            /* Implement this method in your subclass
                with the code you want your thread to run. */
            virtual void InternalThreadEntry() {}
            
            /* Should be tested when running loops to exit when requested. */
            bool must_stop();
            
        private:
            void entry (int device, Caffe::Brew mode, int rand_seed,
                        int solver_count, int solver_rank, bool multiprocess);
                        
            shared_ptr<boost::thread> thread_;
    };
    
}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
