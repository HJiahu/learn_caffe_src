#!/usr/bin/en sh

#images and list_file path
imgs_path="H:/Experiment/test_data/mixed/"
list_file="./test_label.txt"
#生成的LMDB文件的名称
dst_lmdb_file="test_2500_gray_lmdb"

#tools path 
toos_path="C:/Projects/caffe/build/tools"

rm -rf $dst_lmdb_file
$toos_path/caffe --shuffle --gray --backend="lmdb" \
--resize_height=28 --resize_width=28 $imgs_path $list_file  $dst_lmdb_file