﻿#define READ_THIS_FILE
#ifdef READ_THIS_FILE
#ifdef WITH_PYTHON_LAYER
    #include "boost/python.hpp"
    namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

// 下面定义可执行文件在执行时可用的参数，例如我们使用caffe进行训练时使用的指令为：
// caffe train -solver ./*solver.prototxt -gpu 0，可使用参数solver和gpu等就是在下面定义
// 在可执行文件执行时时，可以使用指定的变量提取这些参数，例如gup参数值可以通过全局变量FLAGS_gpu获得
DEFINE_string (gpu, "",
               "Optional; run in GPU mode on given device IDs separated by ','."
               "Use '-gpu all' to run on all available GPUs. The effective training "
               "batch size is multiplied by the number of devices.");
DEFINE_string (solver, "", "The solver definition protocol buffer text file.");
DEFINE_string (model, "", "The model definition protocol buffer text file.");
DEFINE_string (phase, "", "Optional; network phase (TRAIN or TEST). Only used for 'time'.");
DEFINE_int32 (level, 0, "Optional; network level.");
DEFINE_string (stage, "",
               "Optional; network stages (not to be confused with phase), "
               "separated by ','.");
DEFINE_string (snapshot, "", "Optional; the snapshot solver state to resume training.");
DEFINE_string (weights, "",
               "Optional; the pretrained weights to initialize finetuning, "
               "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32 (iterations, 50, "The number of iterations to run.");
DEFINE_string (sigint_effect, "stop",
               "Optional; action to take when a SIGINT signal is received: "
               "snapshot, stop or none.");
DEFINE_string (sighup_effect, "snapshot",
               "Optional; action to take when a SIGHUP signal is received: "
               "snapshot, stop or none.");

// A simple registry for caffe commands.
typedef int (*BrewFunction) ();//定义函数指针
typedef std::map<caffe::string, BrewFunction> BrewMap;//通过名称查找对应的函数
BrewMap g_brew_map;

//编译器将为匿名名字空间生成唯一的名称xx并自动添加一条指令：using xx;
//编译器将把生成的名字空间名字添加到匿名空间中的变量签名中，
//这样在其他文件中就无法链接这些变量，与static效果相同（来保证生成的符号是局部的）
//主要原因是名字xx是随机且唯一的所以外界难以连接并不是说不能，还有static不能修饰class，而匿名空间可以
//在系统把控制权交给main函数入口前完成一些函数的注册
#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { /*利用构造函数将函数指针写进全局变量g_brew_map中*/\
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; /*定义对象，自动调用构造函数*/\
}

//从g_brew_map中提取指定的函数指针
static BrewFunction GetBrewFunction (const caffe::string& name)
{
    if (g_brew_map.count (name))
    {
        return g_brew_map[name];
    }
    
    else
    {
        LOG (ERROR) << "Available caffe actions:";
        
        for (BrewMap::iterator it = g_brew_map.begin();
                it != g_brew_map.end(); ++it)
        {
            LOG (ERROR) << "\t" << it->first;
        }
        
        LOG (FATAL) << "Unknown action: " << name;
        return NULL;  // not reachable, just to suppress old compiler warnings.
    }
}

//设置使用的GPUs，可以指定某个（若gpu 0）或所有
// Parse GPU ids or use all available devices
static void get_gpus (vector<int>* gpus)
{
    if (FLAGS_gpu == "all")
    {
        int count = 0;
#ifndef CPU_ONLY
        CUDA_CHECK (cudaGetDeviceCount (&count));
#else
        NO_GPU;//#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."
#endif
        
        for (int i = 0; i < count; ++i)
        {
            gpus->push_back (i);
        }
    }
    
    else
        if (FLAGS_gpu.size())
        {
            vector<string> strings;
            boost::split (strings, FLAGS_gpu, boost::is_any_of (","));
            
            for (int i = 0; i < strings.size(); ++i)
            {
                gpus->push_back (boost::lexical_cast<int> (strings[i]));
            }
        }
        
        else
        {
            CHECK_EQ (gpus->size(), 0);
        }
}

// 从指令中提取caffe的运行phase
// Parse phase from flags
caffe::Phase get_phase_from_flags (caffe::Phase default_value)
{
    if (FLAGS_phase == "")
    { return default_value; }
    
    if (FLAGS_phase == "TRAIN")
    { return caffe::TRAIN; }
    
    if (FLAGS_phase == "TEST")
    { return caffe::TEST; }
    
    LOG (FATAL) << "phase must be \"TRAIN\" or \"TEST\"";
    return caffe::TRAIN;  // Avoid warning
}

//不明白stage是什么
// Parse stages from flags
vector<string> get_stages_from_flags()
{
    vector<string> stages;
    boost::split (stages, FLAGS_stage, boost::is_any_of (","));
    return stages;
}

//具体指令的实现如：train、test等
// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// 显示指定gpu的一些信息，例如：caffe device_query -gpu 0
// 将显示gpu 0的一些信息，例如内存大小、共享内存大小等
// Device Query: show diagnostic information for a GPU device.
int device_query()
{
    LOG (INFO) << "Querying GPUs " << FLAGS_gpu;
    vector<int> gpus;
    get_gpus (&gpus);
    
    for (int i = 0; i < gpus.size(); ++i)
    {
        caffe::Caffe::SetDevice (gpus[i]);
        caffe::Caffe::DeviceQuery();
    }
    
    return 0;
}
RegisterBrewFunction (device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers (caffe::Solver<float>* solver, const std::string& model_list)
{
    std::vector<std::string> model_names;
    boost::split (model_names, model_list, boost::is_any_of (","));
    
    for (int i = 0; i < model_names.size(); ++i)
    {
        LOG (INFO) << "Finetuning from " << model_names[i];
        solver->net()->CopyTrainedLayersFrom (model_names[i]);
        
        for (int j = 0; j < solver->test_nets().size(); ++j)
        {
            solver->test_nets() [j]->CopyTrainedLayersFrom (model_names[i]);
        }
    }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction (const std::string& flag_value)
{
    if (flag_value == "stop")
    {
        return caffe::SolverAction::STOP;
    }
    
    if (flag_value == "snapshot")
    {
        return caffe::SolverAction::SNAPSHOT;
    }
    
    if (flag_value == "none")
    {
        return caffe::SolverAction::NONE;
    }
    
    LOG (FATAL) << "Invalid signal effect \"" << flag_value << "\" was specified";
    return caffe::SolverAction::NONE;
}

/***********************************************  train  *************************************************/
// Train / Finetune a model.
//常见的命令格式：caffe train --solver=./lenet_solver.prototxt 2>&1 | tee train_doc.log
int train()
{
    //训练的时候必须提供一个 solver
    auto solver_jh = FLAGS_solver;
    CHECK_GT (FLAGS_solver.size(), 0) << "Need a solver definition to train.";
    //也可以提供一个已经训练好的 model 在其基础上进行训练
    auto snapshot_jh = FLAGS_snapshot;
    CHECK (!FLAGS_snapshot.size() || !FLAGS_weights.size())
            << "Give a snapshot to resume training or weights to finetune but not both.";
    //stages 一般是用户自定义的，用于对网络进行重构，具体可参考：http://caffecn.cn/?/question/104
    vector<string> stages = get_stages_from_flags();
    //从solver文件中读取solver信息到指定的对象中
    caffe::SolverParameter solver_param;
    auto slover_jh = FLAGS_solver;//"I:/learn_caffe/learn_caffe/caffe_src/lenet_model/digits_10000/lenet_files/lenet_solver.prototxt"
    caffe::ReadSolverParamsFromTextFileOrDie (FLAGS_solver, &solver_param);
    //FLAGS_level与stages的功能相似，一般是用户自定义的，用于对网络进行重构（不同的条件使用不同的网络），具体可参考：http://caffecn.cn/?/question/104
    auto level_jh = FLAGS_level;//一般而言初学caffe时是不会用到FLAGS_level和stages这两个参数的
    solver_param.mutable_train_state()->set_level (FLAGS_level);
    
    for (int i = 0; i < stages.size(); i++)
    {
        solver_param.mutable_train_state()->add_stage (stages[i]);
    }
    
    // If the gpus flag is not provided, allow the mode and device to be set
    // in the solver prototxt.
    auto &gpu_jh = FLAGS_gpu;
    
    if (FLAGS_gpu.size() == 0
            && solver_param.has_solver_mode()
            && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU)
    {
        if (solver_param.has_device_id())
        {
            FLAGS_gpu = "" +
                        boost::lexical_cast<string> (solver_param.device_id());
        }
        
        else      // Set default GPU if unspecified
        {
            FLAGS_gpu = "" + boost::lexical_cast<string> (0);
        }
    }
    
    vector<int> gpus;
    get_gpus (&gpus);
    //use cpu
    gpus.clear();
    
    if (gpus.size() == 0)
    {
        LOG (INFO) << "Use CPU.";
        Caffe::set_mode (Caffe::CPU);
    }
    
    else
    {
        ostringstream s;
        
        for (int i = 0; i < gpus.size(); ++i)
        {
            s << (i ? ", " : "") << gpus[i];
        }
        
        LOG (INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
        cudaDeviceProp device_prop;
        
        for (int i = 0; i < gpus.size(); ++i)
        {
            cudaGetDeviceProperties (&device_prop, gpus[i]);
            LOG (INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
        }
        
#endif
        solver_param.set_device_id (gpus[0]);
        Caffe::SetDevice (gpus[0]);
        Caffe::set_mode (Caffe::GPU);
        Caffe::set_solver_count (gpus.size());
    }
    
    //Caffe在train或者test的过程中都有可能会遇到系统信号(用户按下ctrl+c或者关掉了控制的terminal)
    //我们可以通过对sigint_effect和sighup_effect来设置遇到系统信号的时候希望进行的处理方式
    //如果用户不设定(大部分时候我自己就没设定)，sigint的默认值为”stop”，sighup的默认值为”snapshot”。
    //参考：http://blog.csdn.net/junmuzi/article/details/52619585?locationNum=9
    caffe::SignalHandler signal_handler (GetRequestedAction (FLAGS_sigint_effect), GetRequestedAction (FLAGS_sighup_effect));
    //创建 solver 并初始化网络
    shared_ptr<caffe::Solver<float> >  solver (caffe::SolverRegistry<float>::CreateSolver (solver_param));
    solver->SetActionFunction (signal_handler.GetActionFunction());
    //auto &snapshot_jh = FLAGS_snapshot;
    
    //是否从已有的模型开始训练（fine tuning）
    if (FLAGS_snapshot.size())
    {
        LOG (INFO) << "Resuming from " << FLAGS_snapshot;
        solver->Restore (FLAGS_snapshot.c_str());
    }
    
    else
        if (FLAGS_weights.size())
        {
            CopyLayers (solver.get(), FLAGS_weights);
        }
        
    LOG (INFO) << "Starting Optimization";
    
    if (gpus.size() > 1)
    {
#ifdef USE_NCCL
        caffe::NCCL<float> nccl (solver);
        nccl.Run (gpus, FLAGS_snapshot.size() > 0 ? FLAGS_snapshot.c_str() : NULL);
#else
        LOG (FATAL) << "Multi-GPU execution not available - rebuild with USE_NCCL";
#endif
    }
    
    else
    {
        //初始化完solve之后就开始读取训练数据、开始训练了
        solver->Solve();
    }
    
    LOG (INFO) << "Optimization Done.";
    return 0;
}
RegisterBrewFunction (train);

/***********************************************  test  *************************************************/
//一般的指令格式为：caffe test -model .\lenet_train_test.prototxt -weights .\lenet_iter_20000.caffemodel -iterations 100
//caffe将使用训练模型中的test数据对网络进行iterations次测试并给出网络的精度
//如果想读caffe具体的forward过程可以看
// Test: score a model.
int test()
{
    CHECK_GT (FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT (FLAGS_weights.size(), 0) << "Need model weights to score.";
    vector<string> stages = get_stages_from_flags();
    // Set device id and mode
    vector<int> gpus;
    get_gpus (&gpus);
    
    if (gpus.size() != 0)
    {
        LOG (INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties (&device_prop, gpus[0]);
        LOG (INFO) << "GPU device name: " << device_prop.name;
#endif
        Caffe::SetDevice (gpus[0]);
        Caffe::set_mode (Caffe::GPU);
    }
    
    else
    {
        LOG (INFO) << "Use CPU.";
        Caffe::set_mode (Caffe::CPU);
    }
    
    // Instantiate the caffe net.
    Net<float> caffe_net (FLAGS_model, caffe::TEST, FLAGS_level, &stages);
    caffe_net.CopyTrainedLayersFrom (FLAGS_weights);
    LOG (INFO) << "Running for " << FLAGS_iterations << " iterations.";
    vector<int> test_score_output_id;
    vector<float> test_score;
    float loss = 0;
    
    for (int i = 0; i < FLAGS_iterations; ++i)
    {
        float iter_loss;
        const vector<Blob<float>*>& result =
            caffe_net.Forward (&iter_loss);
        loss += iter_loss;
        int idx = 0;
        
        for (int j = 0; j < result.size(); ++j)
        {
            const float* result_vec = result[j]->cpu_data();
            
            for (int k = 0; k < result[j]->count(); ++k, ++idx)
            {
                const float score = result_vec[k];
                
                if (i == 0)
                {
                    test_score.push_back (score);
                    test_score_output_id.push_back (j);
                }
                
                else
                {
                    test_score[idx] += score;
                }
                
                const std::string& output_name = caffe_net.blob_names() [
                                      caffe_net.output_blob_indices() [j]];
                LOG (INFO) << "Batch " << i << ", " << output_name << " = " << score;
            }
        }
    }
    
    loss /= FLAGS_iterations;
    LOG (INFO) << "Loss: " << loss;
    
    for (int i = 0; i < test_score.size(); ++i)
    {
        const std::string& output_name = caffe_net.blob_names() [
                                      caffe_net.output_blob_indices() [test_score_output_id[i]]];
        const float loss_weight = caffe_net.blob_loss_weights() [
                               caffe_net.output_blob_indices() [test_score_output_id[i]]];
        std::ostringstream loss_msg_stream;
        const float mean_score = test_score[i] / FLAGS_iterations;
        
        if (loss_weight)
        {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * mean_score << " loss)";
        }
        
        LOG (INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
    }
    
    return 0;
}
RegisterBrewFunction (test);

/***********************************************  time  *************************************************/
// Time: benchmark the execution time of a model.
int time()
{
    CHECK_GT (FLAGS_model.size(), 0) << "Need a model definition to time.";
    caffe::Phase phase = get_phase_from_flags (caffe::TRAIN);
    vector<string> stages = get_stages_from_flags();
    // Set device id and mode
    vector<int> gpus;
    get_gpus (&gpus);
    
    if (gpus.size() != 0)
    {
        LOG (INFO) << "Use GPU with device ID " << gpus[0];
        Caffe::SetDevice (gpus[0]);
        Caffe::set_mode (Caffe::GPU);
    }
    
    else
    {
        LOG (INFO) << "Use CPU.";
        Caffe::set_mode (Caffe::CPU);
    }
    
    // Instantiate the caffe net.
    Net<float> caffe_net (FLAGS_model, phase, FLAGS_level, &stages);
    // Do a clean forward and backward pass, so that memory allocation are done
    // and future iterations will be more stable.
    LOG (INFO) << "Performing Forward";
    // Note that for the speed benchmark, we will assume that the network does
    // not take any input blobs.
    float initial_loss;
    caffe_net.Forward (&initial_loss);
    LOG (INFO) << "Initial loss: " << initial_loss;
    LOG (INFO) << "Performing Backward";
    caffe_net.Backward();
    const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
    const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
    const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
    const vector<vector<bool> >& bottom_need_backward =
        caffe_net.bottom_need_backward();
    LOG (INFO) << "*** Benchmark begins ***";
    LOG (INFO) << "Testing for " << FLAGS_iterations << " iterations.";
    Timer total_timer;
    total_timer.Start();
    Timer forward_timer;
    Timer backward_timer;
    Timer timer;
    std::vector<double> forward_time_per_layer (layers.size(), 0.0);
    std::vector<double> backward_time_per_layer (layers.size(), 0.0);
    double forward_time = 0.0;
    double backward_time = 0.0;
    
    for (int j = 0; j < FLAGS_iterations; ++j)
    {
        Timer iter_timer;
        iter_timer.Start();
        forward_timer.Start();
        
        for (int i = 0; i < layers.size(); ++i)
        {
            timer.Start();
            layers[i]->Forward (bottom_vecs[i], top_vecs[i]);
            forward_time_per_layer[i] += timer.MicroSeconds();
        }
        
        forward_time += forward_timer.MicroSeconds();
        backward_timer.Start();
        
        for (int i = layers.size() - 1; i >= 0; --i)
        {
            timer.Start();
            layers[i]->Backward (top_vecs[i], bottom_need_backward[i],
                                 bottom_vecs[i]);
            backward_time_per_layer[i] += timer.MicroSeconds();
        }
        
        backward_time += backward_timer.MicroSeconds();
        LOG (INFO) << "Iteration: " << j + 1 << " forward-backward time: "
                   << iter_timer.MilliSeconds() << " ms.";
    }
    
    LOG (INFO) << "Average time per layer: ";
    
    for (int i = 0; i < layers.size(); ++i)
    {
        const caffe::string& layername = layers[i]->layer_param().name();
        LOG (INFO) << std::setfill (' ') << std::setw (10) << layername <<
                   "\tforward: " << forward_time_per_layer[i] / 1000 /
                   FLAGS_iterations << " ms.";
        LOG (INFO) << std::setfill (' ') << std::setw (10) << layername  <<
                   "\tbackward: " << backward_time_per_layer[i] / 1000 /
                   FLAGS_iterations << " ms.";
    }
    
    total_timer.Stop();
    LOG (INFO) << "Average Forward pass: " << forward_time / 1000 /
               FLAGS_iterations << " ms.";
    LOG (INFO) << "Average Backward pass: " << backward_time / 1000 /
               FLAGS_iterations << " ms.";
    LOG (INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
               FLAGS_iterations << " ms.";
    LOG (INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
    LOG (INFO) << "*** Benchmark ends ***";
    return 0;
}
RegisterBrewFunction (time);

/*********************************************   main is here  *************************************************/

int main (int argc, char** argv)
{
    //至少在vs2015中调试代码时argc与argv依旧可用此时argc==1
    if (argc == 1)
    {
        //caffe test -model .\lenet_train_test.prototxt -weights .\lenet_iter_20000.caffemodel -iterations 100
        //修改argc和argv，在函数内部提供参数，便于调试
        //指令形式：caffe train -solver ./*solver.prototxt
        //#define USE_CIFAR10
#define USE_LENET
#ifdef USE_CIFAR10
        char* solver_file_path = "I:/learn_caffe/learn_caffe/caffe_src/cifar10_model/cifar10_full_solver.prototxt";
#elif defined(USE_LENET)
        char* solver_file_path = "I:/learn_caffe/learn_caffe/caffe_src/lenet_model/digits_10000/lenet_files/lenet_solver.prototxt";
#endif
#define TRAIN
//#define TEST
#ifdef TRAIN
        char * (command_vec[]) = { "caffe", "train", "-solver", solver_file_path };
        argc = 4;
        argv = command_vec;
#elif defined(TEST) // TRAIN
        char* model_path = R"(D:\age_gender\deepid\gender/deploy.prototxt)";
        char* trained_path = R"(D:\age_gender\deepid\gender/model.caffemodel)";
        char* (command_vec[]) = { "caffe", "test", "-model", model_path, "-weights", trained_path, "-iterations", "100" };
        argc = 8;
        argv = command_vec;
#endif
    }
    
    // Print output to stderr (while still logging).
    FLAGS_alsologtostderr = 1;
    // Set version
    gflags::SetVersionString (AS_STRING (CAFFE_VERSION));
    // Usage message.
    gflags::SetUsageMessage ("command line brew\n"
                             "usage: caffe <command> <args>\n\n"
                             "commands:\n"
                             "  train           train or finetune a model\n"
                             "  test            score a model\n"
                             "  device_query    show GPU diagnostic information\n"
                             "  time            benchmark model execution time");
    // Run tool or show usage.
    caffe::GlobalInit (&argc, &argv);
    
    if (argc == 2)
    {
#ifdef WITH_PYTHON_LAYER
    
        try
        {
#endif
            return GetBrewFunction (caffe::string (argv[1])) ();
#ifdef WITH_PYTHON_LAYER
        }
        
        catch (bp::error_already_set)
        {
            PyErr_Print();
            return 1;
        }
        
#endif
    }
    
    else
    {
        gflags::ShowUsageWithFlagsRestrict (argv[0], "tools/caffe");
    }
}

#endif