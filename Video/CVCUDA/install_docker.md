# 在Docker安装CV-CUDA并运行示例

## 下载镜像并启动镜像

* 下载TensorRT镜像：[TensorRT](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt)

    ```
    docker pull nvcr.io/nvidia/tensorrt:23.03-py3
    ```
* 下载完成后，启动镜像
    ```
    docker run --shm-size=10gb --gpus all -it --name TENSORRT_CVCUDA --net=host -p 7022:22 -p 7006:6006 -p 7064:6064 -p 7888:8888 -v /home/nvidia/weidong/docker/workspace:/workspace nvcr.io/nvidia/tensorrt:23.03-py3 bash
    ```


## 安装cvcuda

* 参照：[Installation](https://cvcuda.github.io/installation.html#pre-requisites)
  * [cv-cuda （cvcuda、nvcv）教程——Python安装](https://blog.csdn.net/u012863603/article/details/128647822)
  * CV-CUDA安装包下载地址：[CV-CUDA Release v0.2.1](https://github.com/CVCUDA/CV-CUDA/releases)
  * 根据需要下载依赖库，并将依赖库放到`/workspace`目录下：
    ![](./images/install_docker/cvcuda_releases.JPG)
    ```
    wget -c https://github.com/CVCUDA/CV-CUDA/releases/download/v0.2.1-alpha/nvcv-dev-0.2.1_alpha-cuda11-x86_64-linux.deb
    wget -c https://github.com/CVCUDA/CV-CUDA/releases/download/v0.2.1-alpha/nvcv-lib-0.2.1_alpha-cuda11-x86_64-linux.deb
    wget -c https://github.com/CVCUDA/CV-CUDA/releases/download/v0.2.1-alpha/nvcv-python3.8-0.2.1_alpha-cuda11-x86_64-linux.deb
    ```
  * 安装：参照：[Installation](https://cvcuda.github.io/installation.html#pre-requisites)
    ```
    dpkg -i nvcv-lib-0.2.1_alpha-cuda11-x86_64-linux.deb
    dpkg -i nvcv-dev-0.2.1_alpha-cuda11-x86_64-linux.deb
    dpkg -i nvcv-python3.8-0.2.1_alpha-cuda11-x86_64-linux.deb
    ```

## 安装其它库

* **pytorch**：https://pytorch.org/
  ```
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

* **torchnvjpeg**:

<br>

## 下载CV-CUDA代码

```
https://github.com/CVCUDA/CV-CUDA.git
```

<br>

## 运行Demo

### 安装依赖
参照：[CV-CUDA Samples](https://github.com/CVCUDA/CV-CUDA/tree/release_v0.2.x/samples)
![Pre-requisites](./images/install_docker/cvcuda_samples_prerequisites.JPG)

* **VPF**: [VideoProcessingFramework](https://github.com/NVIDIA/VideoProcessingFramework)

    * 可能遇到的问题：
        * ![](./images/install_docker/vpf_install_error1.JPG)
        * 参考：[cmake can not find cuda](https://github.com/NVIDIA/VideoProcessingFramework/issues/16)
        * 
        ```
        export CUDACXX=/usr/local/cuda/bin/nvcc
        ```
    ```
    apt install -y \
            libavfilter-dev \
            libavformat-dev \
            libavcodec-dev \
            libswresample-dev \
            libavutil-dev\
            wget \
            build-essential \
            git

    # 这一步比较重要，否则可能报无法找到cuda的错误
    export PATH=/usr/local/cuda/bin:$PATH


    # pip3 install git+https://github.com/NVIDIA/VideoProcessingFramework

    cd /workspace/code/acclib/
    git clone https://github.com/NVIDIA/VideoProcessingFramework.git
    cd VideoProcessingFramework
    pip3 install .

    pip install src/PytorchNvCodec
    ```


* **torchnvjpeg**: https://github.com/itsliupeng/torchnvjpeg

    ```
    cd /workspace/code/acclib/
    git clone https://github.com/itsliupeng/torchnvjpeg.git
    torchnvjpeg/
    python setup.py bdist_wheel
    cd dist/
    pip install torchnvjpeg-0.1.0-cp38-cp38-linux_x86_64.whl
    ```
    * 可能遇到的问题：`ImportError: libc10.so: cannot open shared object file: No such file or directory`
      * [CV-CUDA NVIDIA GPU前后处理库入门](https://blog.csdn.net/qq_40734883/article/details/130052987)
      * 记得在加载torchnvjpeg之前导入torch的包，不然会报某些动态库找不到的错.

* **av: `ModuleNotFoundError: No module named 'av'`**: 
    ```
    pip install av==10.0.0
    ```

* **pyNvCodec: ModuleNotFoundError: No module named 'PyNvCodec'**
  * 如果遇到问题，可以参照VPF的安装


<br>

### 运行Demo
* https://github.com/CVCUDA/CV-CUDA/blob/release_v0.2.x/samples/segmentation/python/inference.py

切换到代码路径，运行如下命令：
```
python inference.py -i /workspace/data/Google_Street_View_camera_cars_in_Hong_Kong_2009.jpg  -o /workspace/data/ -c car
```

运行过程如下：
```
cd /workspace/code/acclib/CV-CUDA/samples/segmentation/python

root@b7e37412e2ea:/workspace/code/acclib/CV-CUDA/samples/segmentation/python# python inference.py -i /workspace/data/Google_Street_View_camera_cars_in_Hong_Kong_2009.jpg  -o /workspace/data/ -c car
Processing batch 1 of 1
        Saving the overlay result for car class for to: /workspace/data/out_Google_Street_View_camera_cars_in_Hong_Kong_2009.jpg

```

![](./images/install_docker/Google_Street_View_camera_cars_in_Hong_Kong_2009.jpg)
![](./images/install_docker/out_Google_Street_View_camera_cars_in_Hong_Kong_2009.jpg)

上述示例中，默认使用的是`pytorch backend`，如果使用`tensorrt backend`，运行代码如下：

```
python inference.py -i /workspace/data/Google_Street_View_camera_cars_in_Hong_Kong_2009.jpg  -o /workspace/data/ -c car -bk tensorrt
```

运行过程错误：

```
root@b7e37412e2ea:/workspace/code/acclib/CV-CUDA/samples/segmentation/python# python inference.py -i /workspace/data/Google_Street_View_camera_cars_in_Hong_Kong_2009.jpg  -o /workspace/data/ -c car -bk tensorrt
Downloading: "https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth" to /root/.cache/torch/hub/checkpoints/fcn_resnet101_coco-7ecb50ca.pth
100.0%
============= Diagnostic Run torch.onnx.export version 2.0.0+cu118 =============
verbose: False, log level: Level.ERROR
======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================

Traceback (most recent call last):
  File "/usr/local/lib/python3.8/dist-packages/torch/onnx/_internal/onnx_proto_utils.py", line 219, in _add_onnxscript_fn
    import onnx
ModuleNotFoundError: No module named 'onnx'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "inference.py", line 755, in <module>
    main()
  File "inference.py", line 751, in main
    sample.run()
  File "inference.py", line 411, in run
    model_info = self.setup_model()
  File "inference.py", line 255, in setup_model
    torch.onnx.export(
  File "/usr/local/lib/python3.8/dist-packages/torch/onnx/utils.py", line 506, in export
    _export(
  File "/usr/local/lib/python3.8/dist-packages/torch/onnx/utils.py", line 1620, in _export
    proto = onnx_proto_utils._add_onnxscript_fn(
  File "/usr/local/lib/python3.8/dist-packages/torch/onnx/_internal/onnx_proto_utils.py", line 221, in _add_onnxscript_fn
    raise errors.OnnxExporterError("Module onnx is not installed!") from e
torch.onnx.errors.OnnxExporterError: Module onnx is not installed!

```

解决问题：

```
pip install onnx
```

继续运行程序，并记录过程及结果如下：

```
python inference.py -i /workspace/data/Google_Street_View_camera_cars_in_Hong_Kong_2009.jpg  -o /workspace/data/ -c car -bk tensorrt
============= Diagnostic Run torch.onnx.export version 2.0.0+cu118 =============
verbose: False, log level: Level.ERROR
======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================

Generated an ONNX model and saved at: /workspace/data/model.1.224.224.onnx
Using TensorRT version: 8.5.3.1
[04/23/2023-02:38:08] [TRT] [I] [MemUsageChange] Init CUDA: CPU +315, GPU +0, now: CPU 3113, GPU 1833 (MiB)
[04/23/2023-02:38:09] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +442, GPU +118, now: CPU 3610, GPU 1951 (MiB)
[04/23/2023-02:38:09] [TRT] [W] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage. See `CUDA_MODULE_LOADING` in https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
Using precision : float16
Loading ONNX file from path /workspace/data/model.1.224.224.onnx
[04/23/2023-02:38:09] [TRT] [W] onnx2trt_utils.cpp:377: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
INPUT[0] : input (1, 3, 224, 224)
OUTPUT[0] : output (1, 21, 224, 224)
OUTPUT[1] : 1022 (1, 21, 224, 224)
Completed parsing of ONNX file.
Building an engine from file /workspace/data/model.1.224.224.onnx. This may take a while...
[04/23/2023-02:38:09] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 3819, GPU 1959 (MiB)
[04/23/2023-02:38:09] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +0, GPU +10, now: CPU 3819, GPU 1969 (MiB)
[04/23/2023-02:38:09] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[04/23/2023-02:38:18] [TRT] [I] Some tactics do not have sufficient workspace memory to run. Increasing workspace size will enable more tactics, please check verbose output for requested sizes.
[04/23/2023-02:38:39] [TRT] [I] Total Activation Memory: 1167678464
[04/23/2023-02:38:39] [TRT] [I] Detected 1 inputs and 2 output network tensors.
[04/23/2023-02:38:39] [TRT] [I] Total Host Persistent Memory: 345472
[04/23/2023-02:38:39] [TRT] [I] Total Device Persistent Memory: 15360
[04/23/2023-02:38:39] [TRT] [I] Total Scratch Memory: 0
[04/23/2023-02:38:39] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 121 MiB, GPU 589 MiB
[04/23/2023-02:38:39] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 112 steps to complete.
[04/23/2023-02:38:39] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 0.608082ms to assign 5 blocks to 112 nodes requiring 8467456 bytes.
[04/23/2023-02:38:39] [TRT] [I] Total Activation Memory: 8467456
[04/23/2023-02:38:39] [TRT] [W] TensorRT encountered issues when converting weights between types and that could affect accuracy.
[04/23/2023-02:38:39] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to adjust the magnitude of the weights.
[04/23/2023-02:38:39] [TRT] [W] Check verbose logs for the list of affected weights.
[04/23/2023-02:38:39] [TRT] [W] - 158 weights are affected by this issue: Detected subnormal FP16 values.
[04/23/2023-02:38:39] [TRT] [W] - 109 weights are affected by this issue: Detected values less than smallest positive FP16 subnormal value and converted them to the FP16 minimum subnormalized value.
[04/23/2023-02:38:39] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +103, GPU +104, now: CPU 103, GPU 104 (MiB)
Completed creating Engine. Saving on disk...
Saved to file /workspace/data/model.1.224.224.trtmodel
Generated TensorRT engine in: /workspace/data/model.1.224.224.trtmodel
[04/23/2023-02:38:39] [TRT] [I] The logger passed into createInferRuntime differs from one already provided for an existing builder, runtime, or refitter. Uses of the global logger, returned by nvinfer1::getLogger(), will return the existing value.

[04/23/2023-02:38:39] [TRT] [I] Loaded engine size: 104 MiB
[04/23/2023-02:38:39] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +103, now: CPU 0, GPU 103 (MiB)
[04/23/2023-02:38:39] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +8, now: CPU 0, GPU 111 (MiB)
[04/23/2023-02:38:39] [TRT] [W] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage. See `CUDA_MODULE_LOADING` in https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
TensorRT Binding[0]: input := shape: (1, 3, 224, 224) dtype: float32
TensorRT Binding[1]: output := shape: (1, 21, 224, 224) dtype: float32
        Allocated the binding as an output.
TensorRT Binding[2]: 1022 := shape: (1, 21, 224, 224) dtype: float32
        Allocated the binding as an output.
Processing batch 1 of 1
        Saving the overlay result for car class for to: /workspace/data/out_Google_Street_View_camera_cars_in_Hong_Kong_2009.jpg
```

我们将使用其中的模型推理部分重复100次，测试一下，使用`pytorch backend`和`tensorrt backend`在推理效率上的差别:


```
root@b7e37412e2ea:/workspace/code/acclib/CV-CUDA/samples/segmentation/python# python inference.py -i /workspace/data/Google_Street_View_camera_cars_in_Hong_Kong_2009.jpg  -o /workspace/data/ -c car -bk tensorrt
Using a pre-built TensorRT engine from: /workspace/data/model.1.224.224.trtmodel
[04/23/2023-03:39:07] [TRT] [I] Loaded engine size: 104 MiB
[04/23/2023-03:39:07] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +103, now: CPU 0, GPU 103 (MiB)
[04/23/2023-03:39:07] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +8, now: CPU 0, GPU 111 (MiB)
[04/23/2023-03:39:07] [TRT] [W] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage. See `CUDA_MODULE_LOADING` in https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
TensorRT Binding[0]: input := shape: (1, 3, 224, 224) dtype: float32
TensorRT Binding[1]: output := shape: (1, 21, 224, 224) dtype: float32
        Allocated the binding as an output.
TensorRT Binding[2]: 1022 := shape: (1, 21, 224, 224) dtype: float32
        Allocated the binding as an output.
Processing batch 1 of 1
Model inference avg time is: 0.00160ms
        Saving the overlay result for car class for to: /workspace/data/out_Google_Street_View_camera_cars_in_Hong_Kong_2009.jpg
root@b7e37412e2ea:/workspace/code/acclib/CV-CUDA/samples/segmentation/python#
root@b7e37412e2ea:/workspace/code/acclib/CV-CUDA/samples/segmentation/python#
root@b7e37412e2ea:/workspace/code/acclib/CV-CUDA/samples/segmentation/python#
root@b7e37412e2ea:/workspace/code/acclib/CV-CUDA/samples/segmentation/python#
root@b7e37412e2ea:/workspace/code/acclib/CV-CUDA/samples/segmentation/python# python inference.py -i /workspace/data/Google_Street_View_camera_cars_in_Hong_Kong_2009.jpg  -o /workspace/data/ -c car -bk pytorch
Processing batch 1 of 1
Model inference avg time is: 0.00631ms
        Saving the overlay result for car class for to: /workspace/data/out_Google_Street_View_camera_cars_in_Hong_Kong_2009.jpg
0
```

可见，仅就模型推理这一步，TensorRT的backend相比于Pytorch的backend，推理效率有很大提升：`0.00160ms` vs `0.00631ms`. 

<br><br>


### 运行VPF Demo

```
cd /workspace/code/acclib/VideoProcessingFramework/samples

root@b7e37412e2ea:/workspace/code/acclib/VideoProcessingFramework/samples# python SampleTorchSegmentation.py
Traceback (most recent call last):
  File "SampleTorchSegmentation.py", line 22, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'
```
解决问题：
```
pip install opencv-python
```

进一步可能遇到问题：
```
ImportError: libGL.so.1: cannot open shared object file: No such file or dire
```

解决问题，参照：[ImportError: libGL.so.1: cannot open shared object file: No such file or directory](https://github.com/open-mmlab/mmsegmentation/issues/1567)或[ImportError: libGL.so.1: cannot open shared object file: No such file or directory](https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo)

```
apt install libgl1-mesa-dev
```

继续运行程序：

```
root@b7e37412e2ea:/workspace/code/acclib/VideoProcessingFramework/samples# python SampleTorchSegmentation.py 0 /workspace/data/dancing.mp4
/usr/local/lib/python3.8/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=SSD300_VGG16_Weights.COCO_V1`. You can also use `weights=SSD300_VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth" to /root/.cache/torch/hub/checkpoints/ssd300_vgg16_coco-b556d3b4.pth
100.0%
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/usr/local/lib/python3.8/dist-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb.

Aborted (core dumped)
```

解决问题：参照：[solved: Could not load the Qt platform plugin "xcb"](https://github.com/NVlabs/instant-ngp/discussions/300)

* 解释如下：
  
    ```
    I ran into this issue, it seems to be from open-cv opencv/opencv-python#46.

    Using open-cv-python-headless instead seems to work for me: https://stackoverflow.com/a/72090539

    pip uninstall opencv-python
    pip install opencv-python-headless
    ```

    ```
    pip uninstall opencv-python
    ```

    ```
    root@b7e37412e2ea:/workspace/code/acclib/VideoProcessingFramework/samples# pip install opencv-python-headless
    Collecting opencv-python-headless
    Downloading opencv_python_headless-4.7.0.72-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (49.2 MB)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 49.2/49.2 MB 22.6 MB/s eta 0:00:00
    Requirement already satisfied: numpy>=1.17.3 in /usr/local/lib/python3.8/dist-packages (from opencv-python-headless) (1.23.5)
    Installing collected packages: opencv-python-headless
    Successfully installed opencv-python-headless-4.7.0.72
    WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv

    [notice] A new release of pip is available: 23.0.1 -> 23.1
    [notice] To update, run: python -m pip install --upgrade pip

    ```

继续运行程序：

```
root@b7e37412e2ea:/workspace/code/acclib/VideoProcessingFramework/samples# python SampleTorchSegmentation.py 0 /workspace/data/dancing.mp4
/usr/local/lib/python3.8/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=SSD300_VGG16_Weights.COCO_V1`. You can also use `weights=SSD300_VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Traceback (most recent call last):
  File "SampleTorchSegmentation.py", line 295, in <module>
    run_inference_on_video(gpu_id, input_video)
  File "SampleTorchSegmentation.py", line 277, in run_inference_on_video
    cv2.imshow("Decode image", image)
cv2.error: OpenCV(4.7.0) /io/opencv/modules/highgui/src/window.cpp:1272: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'

```

解决问题：
```
cv2.imshow("Decode image", image)，程序中涉及到了cv2.imshow，我们现在采用的是程序远程链接，注释掉程序中这段代码即可。
```

继续运行程序, 成功执行：

```
root@b7e37412e2ea:/workspace/code/acclib/VideoProcessingFramework/samples# python SampleTorchSegmentation.py 0 /workspace/data/dancing.mp4
/usr/local/lib/python3.8/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=SSD300_VGG16_Weights.COCO_V1`. You can also use `weights=SSD300_VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Can not decode frame
root@b7e37412e2ea:/workspace/code/acclib/VideoProcessingFramework/samples#

```