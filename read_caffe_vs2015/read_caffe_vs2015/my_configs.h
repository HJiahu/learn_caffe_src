#ifndef JH_MY_CONFIGS_H
#define JH_MY_CONFIGS_H
#include"path.h"

using Path = TinyPath::path;

#ifdef _MSC_VER
//���д���Ͳ��ϵĸ��ļ���
const Path root_path_g{ R"(C:\read_caffe_src)" };
const Path model_root_path_g{R"(C:\read_caffe_src\models)"};
#else
const Path root_path_g { R"()" };
const Path model_root_path_g{ R"()" };
#endif // _MSC_VER




#endif // !JH_MY_CONFIGS_H
