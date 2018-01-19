#ifndef JH_MY_CONFIGS_H
#define JH_MY_CONFIGS_H
#include"path.h"

using Path = TinyPath::path;

#ifdef _MSC_VER
//所有代码和材料的根文件夹
const Path root_path_g{ R"(C:\read_caffe_src)" };
const Path model_root_path_g{R"(C:\read_caffe_src\models)"};
#else
const Path root_path_g { R"()" };
const Path model_root_path_g{ R"()" };
#endif // _MSC_VER




#endif // !JH_MY_CONFIGS_H
