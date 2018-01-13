# refer to https://www.cnblogs.com/Shirlies/p/4282182.html
CC = g++
SUBDIRS = ./src/caffe
# current proj root path
ROOT_DIR=$(shell pwd)
# ROOT_DIR = I:/learn_caffe/learn_caffe/caffe_src
# final obj name
BIN = caffe_cpu.lib
# dir to store objs not final obj BIN
OBJS_DIR = $(ROOT_DIR)/debug/obj
# dir to store BIN file
BIN_DIR = $(ROOT_DIR)/debug
# get all cpp or c files in current folder
CUR_SOURCE = ${wildcard *.cpp}
# objs going to be generated
CUR_OBJS = ${patsubst %.cpp, %.o, $(CUR_SOURCE)}
# MACROs
CAFFE_MACROS = -DCPU_ONLY -DUSE_OPENCV
# include folders for caffe
CAFFE_INCLUDES = -I$(ROOT_DIR)/include  -ID:/Programs/caffe_deps/INCLUDEs 
# export environment values, they will be used by child makefile
export CC BIN OBJS_DIR BIN_DIR ROOT_DIR CAFFE_INCLUDES CAFFE_MACROS


# generating objs
all:$(SUBDIRS) $(CUR_OBJS) DEBUG

$(SUBDIRS):ECHO
	make -C $@
DEBUG:ECHO
	make -C debug
ECHO:
	@echo $(SUBDIRS)

$(CUR_OBJS):%.o:%.cpp
	$(CC) -c -I$(CAFFE_INCLUDES_DIR) $^ -o $(ROOT_DIR)/$(OBJS_DIR)/$@
CLEAN:
	@rm $(OBJS_DIR)/*.o
	@rm -rf $(BIN_DIR)/*