//#define READ_THIS_FILE
#ifdef READ_THIS_FILE

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool (gray, false, "When this option is on, treat images as grayscale ones");
DEFINE_bool (shuffle, false, "Randomly shuffle the order of images and their labels");
DEFINE_string (backend, "lmdb", "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32 (resize_width, 0, "Width images are resized to");
DEFINE_int32 (resize_height, 0, "Height images are resized to");
DEFINE_bool (check_size, false, "When this option is on, check that all the datum have the same size");
DEFINE_bool (encoded, false, "When this option is on, the encoded image will be save in datum");
DEFINE_string (encode_type, "", "Optional: What type should we encode the image as ('png','jpg',...).");

int main (int argc, char** argv)
{
    char* mock_argv[] =
    {
        "",
        "--shuffle",
        "--gray",
        R"(--backend=lmdb)",
        "--resize_height=48",
        "--resize_width=48",
        "I:/learn_caffe/learn_caffe/caffe_src/lenet_model/digits_10000/sample_imgs/",
        "I:/learn_caffe/learn_caffe/caffe_src/lenet_model/digits_10000/sample_imgs_label.txt",
        "I:/learn_caffe/learn_caffe/caffe_src/lenet_model/digits_10000/LMDB_TEST"
    };
    argc = 9;
    mock_argv[0] = argv[0];
    argv = mock_argv;
#ifdef USE_OPENCV
    ::google::InitGoogleLogging (argv[0]);
    // Print output to stderr (while still logging)
    FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif
    gflags::SetUsageMessage ("Convert a set of images to the leveldb/lmdb\n"
                             "format used as input for Caffe.\n"
                             "Usage:\n"
                             "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
                             "The ImageNet dataset for the training demo is at\n"
                             "    http://www.image-net.org/download-images\n");
    gflags::ParseCommandLineFlags (&argc, &argv, true);
    
    if (argc < 4)
    {
        gflags::ShowUsageWithFlagsRestrict (argv[0], "tools/convert_imageset");
        return 1;
    }
    
    //һ��ָ���ʽ�����ڣ�
    //convert_imageset --shuffle --gray --backend="lmdb" --resize_height=48 --resize_width=48 imgs_path list_file  dst_lmdb_file
    const bool is_color = !FLAGS_gray;//--gray
    const bool check_size = FLAGS_check_size;//no
    const bool encoded = FLAGS_encoded;//no
    const string encode_type = FLAGS_encode_type;//no
    string argv2_jh{argv[2]};//list_file
    std::ifstream infile (argv[2]);
    std::vector<std::pair<std::string, int> > lines;//�����ļ����ͱ�ǩ�����ͣ�
    std::string line;
    size_t pos;
    int label;
    
    //��list_file�������ļ����Ͷ�Ӧ�ı�ǩpair�Ž�lines��
    while (std::getline (infile, line))
    {
        pos = line.find_last_of (' ');//��Ϊ������ԭ�򣬱�ǩ�к���Ҫֱ�Ӹ����з�
        label = atoi (line.substr (pos + 1).c_str());//��ǩ
        lines.push_back (std::make_pair (line.substr (0, pos), label));
    }
    
    if (FLAGS_shuffle)
    {
        // randomly shuffle data
        LOG (INFO) << "Shuffling data";
        shuffle (lines.begin(), lines.end());
    }
    
    LOG (INFO) << "A total of " << lines.size() << " images.";
    
    if (encode_type.size() && !encoded)
    { LOG (INFO) << "encode_type specified, assuming encoded=true."; }
    
    int resize_height = std::max<int> (0, FLAGS_resize_height);
    int resize_width = std::max<int> (0, FLAGS_resize_width);
    // Create new DB
    auto backend_jh = FLAGS_backend;
    scoped_ptr<db::DB> db (db::GetDB (FLAGS_backend));
    string argv3_jh{ argv[3] };//dst_lmdb_file
    db->Open (argv[3], db::NEW);
    //txn��transaction����д
    scoped_ptr<db::Transaction> txn (db->NewTransaction());//���ݿ���������txn��װ��caffe����ʹ�õ��������ݿ�(leveldb��LMDB)��һЩ����
    // Storing to db
    std::string root_folder (argv[1]);//imgs_path
    Datum datum;//���������ݿ��е����ݶ���Ҫ��Datum������װ������caffe�����������ʵ����Datum���������
    int count = 0;
    int data_size = 0;
    bool data_size_initialized = false;
    
    for (int line_id = 0; line_id < lines.size(); ++line_id)
    {
        bool status;
        std::string enc = encode_type;
        
        if (encoded && !enc.size())
        {
            // Guess the encoding type from the file name
            string fn = lines[line_id].first;
            size_t p = fn.rfind ('.');
            
            if (p == fn.npos)
            { LOG (WARNING) << "Failed to guess the encoding of '" << fn << "'"; }
            
            enc = fn.substr (p);
            std::transform (enc.begin(), enc.end(), enc.begin(), ::tolower);
        }
        
        status = ReadImageToDatum (root_folder + lines[line_id].first,//ͼƬ��Datum::data_
                                   lines[line_id].second, //��ǩ��Datum::label_
                                   resize_height,//Datum::height_
                                   resize_width,//Datum::width_
                                   is_color,//�ڰ׻��ɫ�������ڱ���ǰ���е�����
                                   enc,//
                                   &datum);
                                   
        if (status == false) { continue; }
        
        if (check_size) //no check in jh
        {
            if (!data_size_initialized)
            {
                data_size = datum.channels() * datum.height() * datum.width();
                data_size_initialized = true;
            }
            
            else
            {
                const std::string& data = datum.data();
                CHECK_EQ (data.size(), data_size) << "Incorrect data field size "
                                                  << data.size();
            }
        }
        
        // sequential
        //Ϊÿһ��Datum��������һ��Ψһ�ı�־�ַ���
        //format_int (line_id, 8) ����һ������Ϊ8���ַ�����ʾ�����������0������� 1 ������ 00000001
        string key_str = caffe::format_int (line_id, 8) + "_" + lines[line_id].first;
        // Put in db
        string out;
        CHECK (datum.SerializeToString (&out));//��Datum���л����ַ���out��
        txn->Put (key_str, out);//��Datum��ӵ����ݿ���
        
        if (++count % 1000 == 0)
        {
            // Commit db
            txn->Commit();
            txn.reset (db->NewTransaction());
            LOG (INFO) << "Processed " << count << " files.";
        }
    }
    
    // write the last batch
    if (count % 1000 != 0)
    {
        txn->Commit();//д��������
        LOG (INFO) << "Processed " << count << " files.";
    }
    
#else
    LOG (FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
#ifdef _MSC_VER
    system ("pause");
#endif // _MSC_VER
    return 0;
}

#endif