# [FasterTransformer/docs/gpt_guide](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md)

## Install

```
docker pull nvcr.io/nvidia/pytorch:22.09-py3
```

```
docker run --gpus all -it --shm-size 5g --rm nvcr.io/nvidia/pytorch:22.09-py3 bash
```

```
git clone https://github.com/NVIDIA/FasterTransformer.git
mkdir -p FasterTransformer/build
cd FasterTransformer/build
git submodule init && git submodule update
```

```
root@4c14de504e57:/workspace# git clone https://github.com/NVIDIA/FasterTransformer.git
Cloning into 'FasterTransformer'...
remote: Enumerating objects: 6806, done.
remote: Counting objects: 100% (486/486), done.
remote: Compressing objects: 100% (173/173), done.
remote: Total 6806 (delta 352), reused 368 (delta 313), pack-reused 6320
Receiving objects: 100% (6806/6806), 67.36 MiB | 8.24 MiB/s, done.
Resolving deltas: 100% (4882/4882), done.
root@4c14de504e57:/workspace# mkdir -p FasterTransformer/build
root@4c14de504e57:/workspace# cd FasterTransformer/build
root@4c14de504e57:/workspace/FasterTransformer/build# git submodule init && git submodule update
Submodule '3rdparty/Megatron-LM' (https://github.com/NVIDIA/Megatron-LM.git) registered for path '../3rdparty/Megatron-LM'
Submodule '3rdparty/cutlass' (https://github.com/NVIDIA/cutlass.git) registered for path '../3rdparty/cutlass'
Submodule 'examples/pytorch/swin/Swin-Transformer-Quantization/SwinTransformer' (https://github.com/microsoft/Swin-Transformer) registered for path '../examples/pytorch/swin/Swin-Transformer-Quantization/SwinTransformer'
Submodule 'examples/pytorch/vit/ViT-quantization/ViT-pytorch' (https://github.com/jeonsworld/ViT-pytorch) registered for path '../examples/pytorch/vit/ViT-quantization/ViT-pytorch'
Submodule 'examples/tensorflow/bert/tensorflow_bert/bert' (https://github.com/google-research/bert.git) registered for path '../examples/tensorflow/bert/tensorflow_bert/bert'
Cloning into '/workspace/FasterTransformer/3rdparty/Megatron-LM'...
Cloning into '/workspace/FasterTransformer/3rdparty/cutlass'...
Cloning into '/workspace/FasterTransformer/examples/pytorch/swin/Swin-Transformer-Quantization/SwinTransformer'...
Cloning into '/workspace/FasterTransformer/examples/pytorch/vit/ViT-quantization/ViT-pytorch'...
Cloning into '/workspace/FasterTransformer/examples/tensorflow/bert/tensorflow_bert/bert'...
Submodule path '../3rdparty/Megatron-LM': checked out 'e156d2fea7fc5c98e645f7742eb86b643956d840'
Submodule path '../3rdparty/cutlass': checked out 'cc85b64cf676c45f98a17e3a47c0aafcf817f088'
Submodule path '../examples/pytorch/swin/Swin-Transformer-Quantization/SwinTransformer': checked out 'b720b4191588c19222ccf129860e905fb02373a7'
Submodule path '../examples/pytorch/vit/ViT-quantization/ViT-pytorch': checked out '460a162767de1722a014ed2261463dbbc01196b6'
Submodule path '../examples/tensorflow/bert/tensorflow_bert/bert': checked out 'eedf5716ce1268e56f0a50264a88cafad334ac61'

```


```
cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
make -j32
```

<br>

****

<br>

如下是8卡A100 40G下的编译结果：

```
root@5186281c8082:/workspace/FasterTransformer/build# cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
-- The CXX compiler identification is GNU 9.4.0
-- The CUDA compiler identification is NVIDIA 11.8.89
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Looking for C++ include pthread.h
-- Looking for C++ include pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE
-- Found CUDA: /usr/local/cuda (found suitable version "11.8", minimum required is "10.2")
CUDA_VERSION 11.8 is greater or equal than 11.0, enable -DENABLE_BF16 flag
-- Found CUDNN: /usr/lib/x86_64-linux-gnu/libcudnn.so
-- Add DBUILD_CUTLASS_MOE, requires CUTLASS. Increases compilation time
-- Add DBUILD_CUTLASS_MIXED_GEMM, requires CUTLASS. Increases compilation time
-- Running submodule update to fetch cutlass
-- Add DBUILD_MULTI_GPU, requires MPI and NCCL
-- Found MPI_CXX: /opt/hpcx/ompi/lib/libmpi.so (found version "3.1")
-- Found MPI: TRUE (found version "3.1")
-- Found NCCL: /usr/include
-- Determining NCCL version from /usr/include/nccl.h...
-- Looking for NCCL_VERSION_CODE
-- Looking for NCCL_VERSION_CODE - not found
-- Found NCCL (include: /usr/include, library: /usr/lib/x86_64-linux-gnu/libnccl.so.2.15.1)
-- NVTX is enabled.
-- Assign GPU architecture (sm=70,75,80,86)
CMAKE_CUDA_FLAGS_RELEASE: -O3 -DNDEBUG -Xcompiler -O3 -DCUDA_PTX_FP8_F2FP_ENABLED --use_fast_math
-- COMMON_HEADER_DIRS: /workspace/FasterTransformer;/usr/local/cuda/include;/workspace/FasterTransformer/3rdparty/cutlass/include;/workspace/FasterTransformer/src/fastertransformer/cutlass_extensions/include;/workspace/FasterTransformer/3rdparty/trt_fp8_fmha/src;/workspace/FasterTransformer/3rdparty/trt_fp8_fmha/generated
-- Found CUDA: /usr/local/cuda (found version "11.8")
-- Caffe2: CUDA detected: 11.8
-- Caffe2: CUDA nvcc is: /usr/local/cuda/bin/nvcc
-- Caffe2: CUDA toolkit directory: /usr/local/cuda
-- Caffe2: Header version is: 11.8
CMake Warning (dev) at /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Caffe2/public/cuda.cmake:117 (find_package):
  Policy CMP0074 is not set: find_package uses <PackageName>_ROOT variables.
  Run "cmake --help-policy CMP0074" for policy details.  Use the cmake_policy
  command to set the policy and suppress this warning.

  CMake variable CUDNN_ROOT is set to:

    /usr/local/cuda

  For compatibility, CMake is ignoring the variable.
Call Stack (most recent call first):
  /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Caffe2/Caffe2Config.cmake:92 (include)
  /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch/TorchConfig.cmake:68 (find_package)
  CMakeLists.txt:257 (find_package)
This warning is for project developers.  Use -Wno-dev to suppress it.

-- Found cuDNN: v8.6.0  (include: /usr/include, library: /usr/lib/x86_64-linux-gnu/libcudnn.so)
-- /usr/local/cuda/lib64/libnvrtc.so shorthash is 672ee683
-- Added CUDA NVCC flags for: -gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86
CMake Warning at /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch/TorchConfig.cmake:22 (message):
  static library kineto_LIBRARY-NOTFOUND not found.
Call Stack (most recent call first):
  /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch/TorchConfig.cmake:127 (append_torchlib_if_found)
  CMakeLists.txt:257 (find_package)


-- Found Torch: /opt/conda/lib/python3.8/site-packages/torch/lib/libtorch.so
-- USE_CXX11_ABI=True

-- The C compiler identification is GNU 9.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Found Python: /opt/conda/bin/python3.8 (found version "3.8.13") found components: Interpreter
-- Configuring done
-- Generating done
-- Build files have been written to: /workspace/FasterTransformer/build

```

```

```

<br>

****

<br>


```
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -P ../models
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -P ../models
```


```
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
mkdir -p ../models/megatron-models/345m
unzip megatron_lm_345m_v0.0.zip -d ../models/megatron-models/345m
export PYTHONPATH=$PWD/..:${PYTHONPATH}
python ../examples/pytorch/gpt/utils/megatron_ckpt_convert.py \
        -head_num 16 \
        -i ../models/megatron-models/345m/release/ \
        -o ../models/megatron-models/c-model/345m/ \
        -t_g 1 \
        -i_g 1 \
        --vocab-path ../models/gpt2-vocab.json \
        --merges-path ../models/gpt2-merges.txt
python ../examples/pytorch/gpt/utils/megatron_ckpt_convert.py \
        -head_num 16 \
        -i ../models/megatron-models/345m/release/ \
        -o ../models/megatron-models/c-model/345m/ \
        -t_g 1 \
        -i_g 8 \
        --vocab-path ../models/gpt2-vocab.json \
        --merges-path ../models/gpt2-merges.txt

python ../examples/pytorch/gpt/utils/megatron_ckpt_convert.py \
        -head_num 16 \
        -i ../models/megatron-models/345m/release/ \
        -o ../models/megatron-models/c-model/345m/ \
        -t_g 1 \
        -i_g 2 \
        --vocab-path ../models/gpt2-vocab.json \
        --merges-path ../models/gpt2-merges.txt        
```


8卡A100
```
root@5186281c8082:/workspace/FasterTransformer/build# mkdir -p ../models/megatron-models/345m
root@5186281c8082:/workspace/FasterTransformer/build# unzip megatron_lm_345m_v0.0.zip -d ../models/megatron-models/345m
Archive:  megatron_lm_345m_v0.0.zip
  inflating: ../models/megatron-models/345m/latest_checkpointed_iteration.txt
  inflating: ../models/megatron-models/345m/release/mp_rank_00/model_optim_rng.pt

root@5186281c8082:/workspace/FasterTransformer/build#
root@5186281c8082:/workspace/FasterTransformer/build# export PYTHONPATH=$PWD/..:${PYTHONPATH}
root@5186281c8082:/workspace/FasterTransformer/build# python ../examples/pytorch/gpt/utils/megatron_ckpt_convert.py \
>         -head_num 16 \
>         -i ../models/megatron-models/345m/release/ \
>         -o ../models/megatron-models/c-model/345m/ \
>         -t_g 1 \
>         -i_g 1 \
>         --vocab-path ../models/gpt2-vocab.json \
>         --merges-path ../models/gpt2-merges.txt
[ERROR] add project root directory to your PYTHONPATH with 'export PYTHONPATH=/workspace/FasterTransformer/build/..:${PYTHONPATH}'
/workspace/FasterTransformer/examples/pytorch/gpt/utils/gpt.py:221: SyntaxWarning: assertion is always true, perhaps remove parentheses?
  assert(self.pre_embed_idx < self.post_embed_idx, "Pre decoder embedding index should be lower than post decoder embedding index.")

=============== Argument ===============
saved_dir: ../models/megatron-models/c-model/345m/
in_file: ../models/megatron-models/345m/release/
infer_gpu_num: 1
head_num: 16
trained_tensor_parallel_size: 1
processes: 16
weight_data_type: fp32
load_checkpoints_to_cpu: 1
vocab_path: ../models/gpt2-vocab.json
merges_path: ../models/gpt2-merges.txt
========================================

[INFO] Spent 0:00:10.920340 (h:m:s) to convert the model
root@5186281c8082:/workspace/FasterTransformer/build#
root@5186281c8082:/workspace/FasterTransformer/build# python ../examples/pytorch/gpt/utils/megatron_ckpt_convert.py \
>         -head_num 16 \
>         -i ../models/megatron-models/345m/release/ \
>         -o ../models/megatron-models/c-model/345m/ \
>         -t_g 1 \
>         -i_g 8 \
>         --vocab-path ../models/gpt2-vocab.json \
>         --merges-path ../models/gpt2-merges.txt
[ERROR] add project root directory to your PYTHONPATH with 'export PYTHONPATH=/workspace/FasterTransformer/build/..:${PYTHONPATH}'

=============== Argument ===============
saved_dir: ../models/megatron-models/c-model/345m/
in_file: ../models/megatron-models/345m/release/
infer_gpu_num: 8
head_num: 16
trained_tensor_parallel_size: 1
processes: 16
weight_data_type: fp32
load_checkpoints_to_cpu: 1
vocab_path: ../models/gpt2-vocab.json
merges_path: ../models/gpt2-merges.txt
========================================
[INFO] Spent 0:00:08.642389 (h:m:s) to convert the model

```


[Run GPT](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md#run-gpt)

```

root@5186281c8082:/workspace/FasterTransformer/build# mpirun -n 8 --allow-run-as-root python ../examples/pytorch/gpt/multi_gpu_gpt_example.py --tensor_para_size=8 --pipeline_para_size=1 --ckpt_path="/workspace/FasterTransformer/models/megatron-models/c-model/345m/8-gpu"
Loading layer_num from config.ini,    previous: 24,    current: 24
Loading max_seq_len from config.ini,    previous: 1024,    current: 1024
Loading weights_data_type from config.ini,    previous: fp32,    current: fp32
Loading head_num from config.ini,    previous: 16,    current: 16
Loading size_per_head from config.ini,    previous: 64,    current: 64
Loading tensor_para_size from config.ini,    previous: 8,    current: 8

=================== Arguments ===================
layer_num.....................: 24
input_len.....................: 1
output_len....................: 32
head_num......................: 16
size_per_head.................: 64
vocab_size....................: 50304
beam_width....................: 1
top_k.........................: 1
top_p.........................: 0.0
temperature...................: 1.0
len_penalty...................: 0.0
beam_search_diversity_rate....: 0.0
tensor_para_size..............: 8
pipeline_para_size............: 1
ckpt_path.....................: /workspace/FasterTransformer/models/megatron-models/c-model/345m/8-gpu
lib_path......................: ./lib/libth_transformer.so
vocab_file....................: ../models/gpt2-vocab.json
merges_file...................: ../models/gpt2-merges.txt
start_id......................: 50256
end_id........................: 50256
max_batch_size................: 8
repetition_penalty............: 1.0
presence_penalty..............: 0.0
min_length....................: 0
max_seq_len...................: 1024
inference_data_type...........: fp32
time..........................: False
sample_input_file.............: None
sample_output_file............: None
enable_random_seed............: False
skip_end_tokens...............: False
detokenize....................: True
use_jieba_tokenizer...........: False
int8_mode.....................: 0
weights_data_type.............: fp32
return_cum_log_probs..........: 0
shared_contexts_ratio.........: 1.0
banned_words..................:
use_gpt_decoder_ops...........: False
=================================================

Loading layer_num from config.ini,    previous: 24,    current: 24
Loading max_seq_len from config.ini,    previous: 1024,    current: 1024
Loading weights_data_type from config.ini,    previous: fp32,    current: fp32
Loading head_num from config.ini,    previous: 16,    current: 16
Loading size_per_head from config.ini,    previous: 64,    current: 64
Loading tensor_para_size from config.ini,    previous: 8,    current: 8

=================== Arguments ===================
layer_num.....................: 24
input_len.....................: 1
output_len....................: 32
head_num......................: 16
size_per_head.................: 64
vocab_size....................: 50304
beam_width....................: 1
top_k.........................: 1
top_p.........................: 0.0
temperature...................: 1.0
len_penalty...................: 0.0
beam_search_diversity_rate....: 0.0
tensor_para_size..............: 8
pipeline_para_size............: 1
ckpt_path.....................: /workspace/FasterTransformer/models/megatron-models/c-model/345m/8-gpu
lib_path......................: ./lib/libth_transformer.so
vocab_file....................: ../models/gpt2-vocab.json
merges_file...................: ../models/gpt2-merges.txt
start_id......................: 50256
end_id........................: 50256
max_batch_size................: 8
repetition_penalty............: 1.0
presence_penalty..............: 0.0
min_length....................: 0
max_seq_len...................: 1024
inference_data_type...........: fp32
time..........................: False
sample_input_file.............: None
sample_output_file............: None
enable_random_seed............: False
skip_end_tokens...............: False
detokenize....................: True
use_jieba_tokenizer...........: False
int8_mode.....................: 0
weights_data_type.............: fp32
return_cum_log_probs..........: 0
shared_contexts_ratio.........: 1.0
banned_words..................:
use_gpt_decoder_ops...........: False
=================================================

Loading layer_num from config.ini,    previous: 24,    current: 24
Loading max_seq_len from config.ini,    previous: 1024,    current: 1024
Loading weights_data_type from config.ini,    previous: fp32,    current: fp32
Loading head_num from config.ini,    previous: 16,    current: 16
Loading size_per_head from config.ini,    previous: 64,    current: 64
Loading tensor_para_size from config.ini,    previous: 8,    current: 8

=================== Arguments ===================
layer_num.....................: 24
input_len.....................: 1
output_len....................: 32
head_num......................: 16
size_per_head.................: 64
vocab_size....................: 50304
beam_width....................: 1
top_k.........................: 1
top_p.........................: 0.0
temperature...................: 1.0
len_penalty...................: 0.0
beam_search_diversity_rate....: 0.0
tensor_para_size..............: 8
pipeline_para_size............: 1
ckpt_path.....................: /workspace/FasterTransformer/models/megatron-models/c-model/345m/8-gpu
lib_path......................: ./lib/libth_transformer.so
vocab_file....................: ../models/gpt2-vocab.json
merges_file...................: ../models/gpt2-merges.txt
start_id......................: 50256
end_id........................: 50256
max_batch_size................: 8
repetition_penalty............: 1.0
presence_penalty..............: 0.0
min_length....................: 0
max_seq_len...................: 1024
inference_data_type...........: fp32
time..........................: False
sample_input_file.............: None
sample_output_file............: None
enable_random_seed............: False
skip_end_tokens...............: False
detokenize....................: True
use_jieba_tokenizer...........: False
int8_mode.....................: 0
weights_data_type.............: fp32
return_cum_log_probs..........: 0
shared_contexts_ratio.........: 1.0
banned_words..................:
use_gpt_decoder_ops...........: False
=================================================

Loading layer_num from config.ini,    previous: 24,    current: 24
Loading max_seq_len from config.ini,    previous: 1024,    current: 1024
Loading weights_data_type from config.ini,    previous: fp32,    current: fp32
Loading head_num from config.ini,    previous: 16,    current: 16
Loading size_per_head from config.ini,    previous: 64,    current: 64
Loading tensor_para_size from config.ini,    previous: 8,    current: 8

=================== Arguments ===================
layer_num.....................: 24
input_len.....................: 1
output_len....................: 32
head_num......................: 16
size_per_head.................: 64
vocab_size....................: 50304
beam_width....................: 1
top_k.........................: 1
top_p.........................: 0.0
temperature...................: 1.0
len_penalty...................: 0.0
beam_search_diversity_rate....: 0.0
tensor_para_size..............: 8
pipeline_para_size............: 1
ckpt_path.....................: /workspace/FasterTransformer/models/megatron-models/c-model/345m/8-gpu
lib_path......................: ./lib/libth_transformer.so
vocab_file....................: ../models/gpt2-vocab.json
merges_file...................: ../models/gpt2-merges.txt
start_id......................: 50256
end_id........................: 50256
max_batch_size................: 8
repetition_penalty............: 1.0
presence_penalty..............: 0.0
min_length....................: 0
max_seq_len...................: 1024
inference_data_type...........: fp32
time..........................: False
sample_input_file.............: None
sample_output_file............: None
enable_random_seed............: False
skip_end_tokens...............: False
detokenize....................: True
use_jieba_tokenizer...........: False
int8_mode.....................: 0
weights_data_type.............: fp32
return_cum_log_probs..........: 0
shared_contexts_ratio.........: 1.0
banned_words..................:
use_gpt_decoder_ops...........: False
=================================================

Initializing tensor and pipeline parallel...
Loading layer_num from config.ini,    previous: 24,    current: 24
Loading max_seq_len from config.ini,    previous: 1024,    current: 1024
Loading weights_data_type from config.ini,    previous: fp32,    current: fp32
Loading head_num from config.ini,    previous: 16,    current: 16
Loading size_per_head from config.ini,    previous: 64,    current: 64
Loading tensor_para_size from config.ini,    previous: 8,    current: 8

=================== Arguments ===================
layer_num.....................: 24
input_len.....................: 1
output_len....................: 32
head_num......................: 16
size_per_head.................: 64
vocab_size....................: 50304
beam_width....................: 1
top_k.........................: 1
top_p.........................: 0.0
temperature...................: 1.0
len_penalty...................: 0.0
beam_search_diversity_rate....: 0.0
tensor_para_size..............: 8
pipeline_para_size............: 1
ckpt_path.....................: /workspace/FasterTransformer/models/megatron-models/c-model/345m/8-gpu
lib_path......................: ./lib/libth_transformer.so
vocab_file....................: ../models/gpt2-vocab.json
merges_file...................: ../models/gpt2-merges.txt
start_id......................: 50256
end_id........................: 50256
max_batch_size................: 8
repetition_penalty............: 1.0
presence_penalty..............: 0.0
min_length....................: 0
max_seq_len...................: 1024
inference_data_type...........: fp32
time..........................: False
sample_input_file.............: None
sample_output_file............: None
enable_random_seed............: False
skip_end_tokens...............: False
detokenize....................: True
use_jieba_tokenizer...........: False
int8_mode.....................: 0
weights_data_type.............: fp32
return_cum_log_probs..........: 0
shared_contexts_ratio.........: 1.0
banned_words..................:
use_gpt_decoder_ops...........: False
=================================================

Loading layer_num from config.ini,    previous: 24,    current: 24
Loading max_seq_len from config.ini,    previous: 1024,    current: 1024
Loading weights_data_type from config.ini,    previous: fp32,    current: fp32
Loading head_num from config.ini,    previous: 16,    current: 16
Loading size_per_head from config.ini,    previous: 64,    current: 64
Loading tensor_para_size from config.ini,    previous: 8,    current: 8

=================== Arguments ===================
layer_num.....................: 24
input_len.....................: 1
output_len....................: 32
head_num......................: 16
size_per_head.................: 64
vocab_size....................: 50304
beam_width....................: 1
top_k.........................: 1
top_p.........................: 0.0
temperature...................: 1.0
len_penalty...................: 0.0
beam_search_diversity_rate....: 0.0
tensor_para_size..............: 8
pipeline_para_size............: 1
ckpt_path.....................: /workspace/FasterTransformer/models/megatron-models/c-model/345m/8-gpu
lib_path......................: ./lib/libth_transformer.so
vocab_file....................: ../models/gpt2-vocab.json
merges_file...................: ../models/gpt2-merges.txt
start_id......................: 50256
end_id........................: 50256
max_batch_size................: 8
repetition_penalty............: 1.0
presence_penalty..............: 0.0
min_length....................: 0
max_seq_len...................: 1024
inference_data_type...........: fp32
time..........................: False
sample_input_file.............: None
sample_output_file............: None
enable_random_seed............: False
skip_end_tokens...............: False
detokenize....................: True
use_jieba_tokenizer...........: False
int8_mode.....................: 0
weights_data_type.............: fp32
return_cum_log_probs..........: 0
shared_contexts_ratio.........: 1.0
banned_words..................:
use_gpt_decoder_ops...........: False
=================================================

Initializing tensor and pipeline parallel...
Loading layer_num from config.ini,    previous: 24,    current: 24
Loading max_seq_len from config.ini,    previous: 1024,    current: 1024
Loading weights_data_type from config.ini,    previous: fp32,    current: fp32
Loading head_num from config.ini,    previous: 16,    current: 16
Loading size_per_head from config.ini,    previous: 64,    current: 64
Loading tensor_para_size from config.ini,    previous: 8,    current: 8

=================== Arguments ===================
layer_num.....................: 24
input_len.....................: 1
output_len....................: 32
head_num......................: 16
size_per_head.................: 64
vocab_size....................: 50304
beam_width....................: 1
top_k.........................: 1
top_p.........................: 0.0
temperature...................: 1.0
len_penalty...................: 0.0
beam_search_diversity_rate....: 0.0
tensor_para_size..............: 8
pipeline_para_size............: 1
ckpt_path.....................: /workspace/FasterTransformer/models/megatron-models/c-model/345m/8-gpu
lib_path......................: ./lib/libth_transformer.so
vocab_file....................: ../models/gpt2-vocab.json
merges_file...................: ../models/gpt2-merges.txt
start_id......................: 50256
end_id........................: 50256
max_batch_size................: 8
repetition_penalty............: 1.0
presence_penalty..............: 0.0
min_length....................: 0
max_seq_len...................: 1024
inference_data_type...........: fp32
time..........................: False
sample_input_file.............: None
sample_output_file............: None
enable_random_seed............: False
skip_end_tokens...............: False
detokenize....................: True
use_jieba_tokenizer...........: False
int8_mode.....................: 0
weights_data_type.............: fp32
return_cum_log_probs..........: 0
shared_contexts_ratio.........: 1.0
banned_words..................:
use_gpt_decoder_ops...........: False
=================================================

Loading layer_num from config.ini,    previous: 24,    current: 24
Loading max_seq_len from config.ini,    previous: 1024,    current: 1024
Loading weights_data_type from config.ini,    previous: fp32,    current: fp32
Loading head_num from config.ini,    previous: 16,    current: 16
Loading size_per_head from config.ini,    previous: 64,    current: 64
Loading tensor_para_size from config.ini,    previous: 8,    current: 8

=================== Arguments ===================
layer_num.....................: 24
input_len.....................: 1
output_len....................: 32
head_num......................: 16
size_per_head.................: 64
vocab_size....................: 50304
beam_width....................: 1
top_k.........................: 1
top_p.........................: 0.0
temperature...................: 1.0
len_penalty...................: 0.0
beam_search_diversity_rate....: 0.0
tensor_para_size..............: 8
pipeline_para_size............: 1
ckpt_path.....................: /workspace/FasterTransformer/models/megatron-models/c-model/345m/8-gpu
lib_path......................: ./lib/libth_transformer.so
vocab_file....................: ../models/gpt2-vocab.json
merges_file...................: ../models/gpt2-merges.txt
start_id......................: 50256
end_id........................: 50256
max_batch_size................: 8
repetition_penalty............: 1.0
presence_penalty..............: 0.0
min_length....................: 0
max_seq_len...................: 1024
inference_data_type...........: fp32
time..........................: False
sample_input_file.............: None
sample_output_file............: None
enable_random_seed............: False
skip_end_tokens...............: False
detokenize....................: True
use_jieba_tokenizer...........: False
int8_mode.....................: 0
weights_data_type.............: fp32
return_cum_log_probs..........: 0
shared_contexts_ratio.........: 1.0
banned_words..................:
use_gpt_decoder_ops...........: False
=================================================

Initializing tensor and pipeline parallel...
Initializing tensor and pipeline parallel...
Initializing tensor and pipeline parallel...
Initializing tensor and pipeline parallel...
Initializing tensor and pipeline parallel...
Initializing tensor and pipeline parallel...
[INFO] WARNING: Have initialized the process group
[INFO] WARNING: Have initialized the process group
[INFO] WARNING: Have initialized the process group
[INFO] WARNING: Have initialized the process group
[INFO] WARNING: Have initialized the process group
[INFO] WARNING: Have initialized the process group
[INFO] WARNING: Have initialized the process group
[INFO] WARNING: Have initialized the process group
[WARNING] gemm_config.in is not found; using default GEMM algo
[WARNING] gemm_config.in is not found; using default GEMM algo
[WARNING] gemm_config.in is not found; using default GEMM algo
[WARNING] gemm_config.in is not found; using default GEMM algo
[WARNING] gemm_config.in is not found; using default GEMM algo
[WARNING] gemm_config.in is not found; using default GEMM algo
[WARNING] gemm_config.in is not found; using default GEMM algo
[WARNING] gemm_config.in is not found; using default GEMM algo
[FT][INFO] NCCL initialized rank=7 world_size=8 tensor_para=NcclParam[rank=7, world_size=8, nccl_comm=0x555777e93370] pipeline_para=NcclParam[rank=0, world_size=1, nccl_comm=0x5557781a01a0]
[FT][INFO] Device NVIDIA A100-SXM4-40GB
[FT][INFO] NCCL initialized rank=6 world_size=8 tensor_para=NcclParam[rank=6, world_size=8, nccl_comm=0x558cdd853780] pipeline_para=NcclParam[rank=0, world_size=1, nccl_comm=0x558cdf9dfc40]
[FT][INFO] Device NVIDIA A100-SXM4-40GB
[FT][INFO] NCCL initialized rank=5 world_size=8 tensor_para=NcclParam[rank=5, world_size=8, nccl_comm=0x55f13eb76540] pipeline_para=NcclParam[rank=0, world_size=1, nccl_comm=0x55f1429047e0]
[FT][INFO] Device NVIDIA A100-SXM4-40GB
[FT][INFO] NCCL initialized rank=4 world_size=8 tensor_para=NcclParam[rank=4, world_size=8, nccl_comm=0x5603ebe2b740] pipeline_para=NcclParam[rank=0, world_size=1, nccl_comm=0x5603efbb9610]
[FT][INFO] NCCL initialized rank=2 world_size=8 tensor_para=NcclParam[rank=2, world_size=8, nccl_comm=0x55d3938b4760] pipeline_para=NcclParam[rank=0, world_size=1, nccl_comm=0x55d3966417a0]
[FT][INFO] NCCL initialized rank=3 world_size=8 tensor_para=NcclParam[rank=3, world_size=8, nccl_comm=0x55a3c5171430] pipeline_para=NcclParam[rank=0, world_size=1, nccl_comm=0x55a3c72fd9c0]
[FT][INFO] Device NVIDIA A100-SXM4-40GB
[FT][INFO] Device NVIDIA A100-SXM4-40GB
[FT][INFO] Device NVIDIA A100-SXM4-40GB
[FT][INFO] NCCL initialized rank=1 world_size=8 tensor_para=NcclParam[rank=1, world_size=8, nccl_comm=0x55ff2bbe55d0] pipeline_para=NcclParam[rank=0, world_size=1, nccl_comm=0x55ff2e972900]
[FT][INFO] NCCL initialized rank=0 world_size=8 tensor_para=NcclParam[rank=0, world_size=8, nccl_comm=0x561e2ba92ee0] pipeline_para=NcclParam[rank=0, world_size=1, nccl_comm=0x561e2c0912a0]
[FT][INFO] Device NVIDIA A100-SXM4-40GB
[FT][INFO] Device NVIDIA A100-SXM4-40GB
[INFO] batch 0, beam 0:
[Context]
<|endoftext|>

[Output]
The first time I saw the movie "The Big Short," I was in the middle of a long, long day. I was working at a bank in New

[INFO] batch 1, beam 0:
[Context]
<|endoftext|>

[Output]
The first time I saw the movie "The Big Short," I was in the middle of a long, long day. I was working at a bank in New

[INFO] batch 2, beam 0:
[Context]
<|endoftext|>

[Output]
The first time I saw the movie "The Big Short," I was in the middle of a long, long day. I was working at a bank in New

[INFO] batch 3, beam 0:
[Context]
<|endoftext|>

[Output]
The first time I saw the movie "The Big Short," I was in the middle of a long, long day. I was working at a bank in New

[INFO] batch 4, beam 0:
[Context]
<|endoftext|>

[Output]
The first time I saw the movie "The Big Short," I was in the middle of a long, long day. I was working at a bank in New

[INFO] batch 5, beam 0:
[Context]
<|endoftext|>

[Output]
The first time I saw the movie "The Big Short," I was in the middle of a long, long day. I was working at a bank in New

[INFO] batch 6, beam 0:
[Context]
<|endoftext|>

[Output]
The first time I saw the movie "The Big Short," I was in the middle of a long, long day. I was working at a bank in New

[INFO] batch 7, beam 0:
[Context]
<|endoftext|>

[Output]
The first time I saw the movie "The Big Short," I was in the middle of a long, long day. I was working at a bank in New



```

<br>

```
mpirun -n 8 --allow-run-as-root python ../examples/pytorch/gpt/multi_gpu_gpt_example.py --tensor_para_size=1 --pipeline_para_size=8 --ckpt_path="/workspace/FasterTransformer/models/megatron-models/c-model/345m/1-gpu"
```


## Ref

1. [NCCL error when running distributed training](https://discuss.pytorch.org/t/nccl-error-when-running-distributed-training/129301)