#ifndef INCLUDE_CAFFE_UTIL_SIGNAL_HANDLER_H_
#define INCLUDE_CAFFE_UTIL_SIGNAL_HANDLER_H_

#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"

namespace caffe
{
    //Caffe在train或者test的过程中都有可能会遇到系统信号(用户按下ctrl+c或者关掉了控制的terminal)
    //我们可以通过对sigint_effect和sighup_effect来设置遇到系统信号的时候希望进行的处理方式
    class SignalHandler
    {
        public:
            // Contructor. Specify what action to take when a signal is received.
            SignalHandler (SolverAction::Enum SIGINT_action,
                           SolverAction::Enum SIGHUP_action);
            ~SignalHandler();
            ActionCallback GetActionFunction();
        private:
            SolverAction::Enum CheckForSignals() const;
            SolverAction::Enum SIGINT_action_;
            SolverAction::Enum SIGHUP_action_;
    };
    
}  // namespace caffe

#endif  // INCLUDE_CAFFE_UTIL_SIGNAL_HANDLER_H_
