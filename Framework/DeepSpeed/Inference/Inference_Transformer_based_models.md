# [Getting Started with DeepSpeed for Inferencing Transformer based Models](https://www.deepspeed.ai/tutorials/inference-tutorial/)

## [DeepSpeed: Accelerating large-scale model inference and training via system optimizations and compression](https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations-and-compression/)

## [ZeRO-Infinity and DeepSpeed: Unlocking unprecedented model scale for deep learning training](https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/)
* ZeRO-Infinity 和 DeepSpeed：为深度学习训练解锁前所未有的模型规模
* 
* 自去年推出 DeepSpeed 优化库以来，它已经推出了许多用于训练大型 AI 模型的新颖优化——提高了规模、速度、成本和可用性。 随着大型模型在过去一年中迅速发展，DeepSpeed 也是如此。 无论是帮助研究人员创建具有最先进精度的 170 亿参数 Microsoft Turing 自然语言生成 (Turing-NLG)，实现最快的 BERT 训练记录，还是使用单个 GPU 支持 10 倍更大的模型训练，DeepSpeed 凭借大规模模型训练的最新进展，继续应对大规模人工智能的挑战。 现在，DeepSpeed所包含的新型内存优化技术ZeRO（零冗余优化器）正在经历着自身的进一步蜕变。 改进后的 ZeRO-Infinity 提供了超越 GPU 内存壁垒的系统能力，可以训练具有数十万亿参数的模型，比最先进的系统可以支持的数量级大一个数量级。 它还为训练 100 万亿参数模型提供了一条有前途的途径。
* 
* ZeRO-Infinity 概览：ZeRO-Infinity 是一种新颖的深度学习 (DL) 训练技术，用于扩展模型训练，从单个 GPU 到拥有数千个 GPU 的大型超级计算机。 它通过利用系统的全部内存容量，同时利用所有异构内存（GPU、CPU 和 Non-Volatile Memory express 或简称 NVMe）来支持前所未有的模型大小。 在我们的论文“ZeRO-Infinity：打破 GPU 内存墙以实现超大规模深度学习”中了解更多信息。 ZeRO-Infinity 的亮点包括：



## [搞定千亿参数，训练时间只用1/3，微软全新工具催生超级NLP模型](https://picture.iczhiku.com/weixin/message1581455584218.html)
* 为了解决数据并行和模型并行存在的问题，ZeRO 提供了三阶段的优化方法，分别为优化器状态分割、梯度分割、参数分割，三个阶段按顺序实施。
  * 在优化器分割状态：ZeRO 降低了 3/4 的内存，通信量和数据并行相同；
  * 加入梯度分割：降低了 7/8 的内存，通信量和数据并行相同；
  * 加入参数分割：内存减少与数据并行度呈线性关系。例如，在 64 个 GPU 上进行分割的时候，可以将内存降至 1/64。在通信量上有 50% 的提升。



## 思考

### 2023.5.10

我一直听闻说Nemo的速度比Deepspeed的训练速度会更高效，但是一直找不到明确的佐证。从面上看，Tensor并行和pipeline并行是两者共用的，这里应该不会有区别。区别在于后来Nemo引入了Sequence并行、Selective Activate Checkpoint Recompution、Checkpointing Skipping，DeepSpeed用了ZeRO。但就我理解ZeRO和Sequence并行应该是不同层面的东西，会对整体效率造成那么大的影响？
* ZeRO和Sequence并行能同时使用吗？
