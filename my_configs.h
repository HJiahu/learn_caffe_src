#ifndef JH_MY_CONFIGS_H
#define JH_MY_CONFIGS_H
#include"path.h"

using Path = TinyPath::path;

<<<<<<< HEAD
//���д���Ͳ��ϵĸ��ļ���
const Path root_path_g{R"(I:\learn_caffe\learn_caffe\caffe_src)"};
=======
#ifdef _MSC_VER
//���д���Ͳ��ϵĸ��ļ���
const Path root_path_g{ R"(I:\learn_caffe\learn_caffe\caffe_src)" };
#else
const Path root_path_g { R"()" };
#endif // _MSC_VER


>>>>>>> 0d0d59daf845ecd263ed90aebb61ba59a26ddaec


#endif // !JH_MY_CONFIGS_H
