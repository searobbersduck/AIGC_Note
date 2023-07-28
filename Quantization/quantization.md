# Quantization

<br>

## Pytorch

* **[QUANTIZATION](https://pytorch.org/docs/stable/quantization.html)**
  * 最全的官方文档
  * Frequently Asked Questions
    * How can I do quantized inference on GPU?:
      * We don’t have official GPU support yet, but this is an area of active development, you can find more information here
    * Where can I get ONNX support for my quantized model?:
      * You can open an issue in GitHub - onnx/onnx when you encounter problems with ONNX, or reach out to people in this list: PyTorch Governance | Maintainers | ONNX exporter
    * How can I use quantization with LSTM’s?:
      * LSTM is supported through our custom module api in both eager mode and fx graph mode quantization. Examples can be found at Eager Mode: pytorch/test_quantized_op.py TestQuantizedOps.test_custom_module_lstm FX Graph Mode: pytorch/test_quantize_fx.py TestQuantizeFx.test_static_lstm
* **[Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/)**
  * 科普的blog，看过两遍了
* **[PYTORCH NUMERIC SUITE TUTORIAL](https://pytorch.org/tutorials/prototype/numeric_suite_tutorial.html)**
  * One important step of debugging is to measure the statistics of the float model and its corresponding quantized model to know where are they differ most. We built a suite of numeric tools called PyTorch Numeric Suite in PyTorch quantization to enable the measurement of the statistics between quantized module and float module to support quantization debugging efforts.
  * PyTorch Numeric Suite currently supports models quantized through both static quantization and dynamic quantization with unified APIs.
* **[STATIC QUANTIZATION WITH EAGER MODE IN PYTORCH](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)**
  * 量化现在还不支持GPU???
  * In this tutorial, we showed two quantization methods - post-training static quantization, and quantization-aware training - describing what they do “under the hood” and how to use them in PyTorch.
  * 真实模型的量化示例，示例讲的很清晰了，找时间实践一遍；

* **[PyTorch的量化](https://zhuanlan.zhihu.com/p/299108528)**
  * 评论区里的话：这是我看过的讲的最透彻的pytorch量化，这篇blog还没细看，之后再浏览一遍


## TensorRT

* **[Achieving FP32 Accuracy for INT8 Inference Using Quantization Aware Training with NVIDIA TensorRT](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)**
* 