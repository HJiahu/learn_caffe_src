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
    //boost thread�ķ�װ��һ����������һ��ִ���̣߳���������start�����жϣ�stop��
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
            //����ʹ����boost���ṩ�ĺ���interrupt���ж�һ���̣߳�
            //��C++11 �в��ṩinterrupt�����������ڿ����쳣������������ľ����������Ҫ��Ҫ�Լ�ʵ��
            /*
              boost��ʵ��thread stop�ķ����ǡ���Ƿ��������̵߳�ĳЩ�׶Σ���sleep��wait���Ⱥ�����
              ����һ���жϱ�־����־Ϊ�����˳��̣߳���ν��ĳЩ�׶�Ҳ�����Լ����壬�����̴߳����в���
              boost�ṩ��interruption_point()��interruption_point���Զ��жϱ�־����������boost
              �п�ʹ�ó�Ա����interrupt����Ӧ�ı�־λ��Ϊ��
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
