#!/usr/bin/en sh
#tools path 
toos_path="F:/libs/caffe/build/install/bin"

$toos_path/caffe train --solver=./lenet_solver.prototxt 2>&1 | tee train_doc.log