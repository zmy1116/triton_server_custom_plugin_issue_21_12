# triton_server_custom_plugin_issue_21_12
This repository reproduce triton server 21.12-py3 issues with custom TensorRT plugin. We create a simple model takes input of size `1x10`  go through a simple cliping plugin

The plugin is directly copied from https://github.com/NVIDIA/TensorRT/tree/main/samples/python/uff_custom_plugin It's almost exactly the same except fixing the input to be 1x10 and cuda code to take block size 1 and thread number 10



# Setup 
```buildoutcfg
# clone repository 
git clone https://github.com/zmy1116/triton_server_custom_plugin_issue_21_12

# TensorRT docker pull 
docker pull nvcr.io/nvidia/tensorrt:21.12-py3

# Triton server docker pull 
docker pull nvcr.io/nvidia/tritonserver:21.12-py3
```

# Create TensorRT engine 

The engine and plugin are already in the repository 
```buildoutcfg
TRT model engine: model  
plugin: libclipplugin.so
```

To reproduce the engine and plugin

load the docker TRT environment
```buildoutcfg
docker run --gpus all -it -p8889:8889 --rm -v /home/ubuntu:/workspace/ubuntu nvcr.io/nvidia/tensorrt:21.12-py3
```

generate plugin 
```buildoutcfg
cd triton_server_custom_plugin_issue_21_12/custom_plugin
mkdir build
cd build 
cmake ..
make
```
generate model TRT engine 
```buildoutcfg
python create_engine.py
```

# Launch the Triton Server and Produce error 

The folder `dummy_repository` already has everything organized.  One can organize the produced `model` and `libclipplugin.so` into a model repository to reproduce.

To run the repository 
```buildoutcfg
docker run --gpus=all --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v /home/ubuntu/dummy_repository:/models  -eLD_PRELOAD=/models/dummy/libclipplugin.so nvcr.io/nvidia/tritonserver:21.12-py3 tritonserver --model-repository=/models --strict-model-config=false
```

And we see the following errors printing on sreen 
```buildoutcfg
E0105 05:32:15.146025 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146037 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146045 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146054 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146064 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146072 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146084 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146094 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146102 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146111 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146122 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146131 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146152 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146160 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146171 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146180 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146189 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146201 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146210 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146219 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146227 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146237 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146246 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146257 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146267 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146275 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146284 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146294 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146303 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146314 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146323 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
E0105 05:32:15.146332 1 logging.cc:43] 1: [checkMacros.cpp::catchCudaError::272] Error Code 1: Cuda Runtime (initialization error)
```

