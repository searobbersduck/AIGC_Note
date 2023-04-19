## RAFT
* [Welcome to RAFT’s documentation!](https://docs.rapids.ai/api/raft/stable/)
  * RAFT contains fundamental widely-used algorithms and primitives for scientific computing, data science and machine learning. The algorithms are CUDA-accelerated and form building-blocks for rapidly composing analytics.
  * RAFT 包含用于科学计算、数据科学和机器学习的被广泛使用的基本的算法和原语。 这些算法经过 CUDA 加速，形成了可用于快速组合分析的构建块。
* [使用 RAPIDS RAFT 进行机器学习和数据分析的可重用计算模式](https://developer.nvidia.com/zh-cn/blog/reusable-computational-patterns-for-machine-learning-and-data-analytics-with-rapids-raft/)
* [RAFT在Knowhere上的一些评估测试[1]](https://zhuanlan.zhihu.com/p/616244379)
  * RAFT 全名 Reusable Accelerated Functions and Tools，从名字上来看就是工具集合。属于Rapidsai的一个项目。
  * RAFT作为一个加速函数库，实现诸如：稀疏矩阵、空间算法、基础的类聚等等的基础算法。而临近搜索自然也被包含在里面了，目前看RAFT实现的临近算法主要 IVFFlat, IVFPQ, BruteForce(Flat)。
  * Knowhere是向量数据库Milvus的向量引擎。最近主线上已经合并了RAFT索引的支持。

## cuLitho

* [造芯片的“计算光刻”，了解一下](https://www.eet-china.com/news/202303227503.html)
  * 光刻是芯片制造过程中最重要的一个步骤，就像是用“光刀”在晶圆上“雕刻”一样。“雕刻”当然是要“刻”出特定的图案的。这个图案首先要呈现在光掩膜（photomask）上。掩膜板就像是漏字板，激光一照，通过镜头，“漏字板”上的图案也就落到了硅片上。
  * 晶体管、器件、互联线路都需要经过这样的光刻步骤。实操当然比这三两句话的形容要复杂得多，比如现在的芯片上上下下那么多层，不同的层就需要不同的光刻和掩膜板；而且某些层如果器件间距很小，就可能需要多次光刻。
  * 
* [英伟达发布cuLitho，计算光刻提升40倍！阿斯麦台积电站台支持！](https://zhuanlan.zhihu.com/p/616512305)
* [【光刻百科】光学邻近效应修正 Optical Proximity Correction (OPC)](https://picture.iczhiku.com/weixin/message1615564561035.html)
* [计算光刻究竟是个啥？和光刻是什么关系？](https://zhuanlan.zhihu.com/p/617817214)
  * cuLitho可以将下一代芯片计算光刻度提高 40 倍以上，极大降低了光掩膜版开发的时间和成本。
  * 

## TRT/Triton

* [Model Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_analyzer.html)
  * The Triton Model Analyzer is a tool that uses Performance Analyzer to send requests to your model while measuring GPU memory and compute utilization. The Model Analyzer is specifically useful for characterizing the GPU memory requirements for your model under different batching and model instance configurations. Once you have this GPU memory usage information you can more intelligently decide on how to combine multiple models on the same GPU while remaining within the memory capacity of the GPU.
  * Triton 模型分析器是一种工具，它使用性能分析器向您的模型发送请求，同时测量 GPU 内存和计算利用率。 模型分析器特别适用于表征不同批处理和模型实例配置下模型的 GPU 内存需求。 获得此 GPU 内存使用信息后，您可以更智能地决定如何在同一 GPU 上组合多个模型，同时保持在 GPU 的内存容量内。
* [NVIDIA Triton Management Service Demo](https://www.youtube.com/watch?v=Gtko-PpIntk)
* [NVIDIA 发布TensorRT 8.6](https://www.zhihu.com/zvideo/1622980418319360000)


## CV-CUDA

* [CV-CUDA](https://cvcuda.github.io/)
  * NVIDIA CV-CUDA™ 是一个开源项目，用于构建云级(AI) 成像和计算机视觉 (CV) 应用程序。 它使用图形处理单元 (GPU) 加速来帮助开发人员构建高效的预处理和后处理管道。 它可以将吞吐量提高 10 倍以上，同时降低云计算成本。
* [VideoProcessingFramework（VPF）GPU视频流解码方案](https://zhuanlan.zhihu.com/p/615372907)
  * VideoProcessingFramework（VPF）是NVIDIA开源的适用于Python的视频处理框架，可用于硬件加速条件下的视频编解码等处理类任务。同时对于Pytorch比较友好，能够将解析出来的图像数据直接转化成Tensor()的格式。
  * 在ffmpegcv-利用GPU进行视频编解码的文章中，介绍了ffmpegcv库，通过底层编译支持Cuda的ffmpeg对视频进行解码。但是这个方案的局限性在于，只能对视频进行解码，不能解析视频流。解码得到的YUV格式图片在CPU下转化到RGB的色彩空间，没有在GPU上进行全部转化流程。


## Cuopt

NVIDIA cuOpt uses GPU-accelerated logistics solvers relying on heuristics, metaheuristics, and optimization to calculate complex vehicle-routing-problem variants with a wide range of constraints. NVIDIA cuOpt provides a Python interface that relies on NVIDIA CUDA® libraries and RAPIDS™ primitives. Native support for distance and time matrices with asymmetric patterns enables smooth integration with popular map engines. Available on LaunchPad and all major public cloud platforms.

cuOpt有着如下特征：
动态重新路由：重新运行模型并根据变化进行调整，例如倒下的司机、无法操作的车辆、交通/天气中断以及新订单的添加——所有这些都在 SLA 时间限制内。
世界纪录的准确性：在 Gehring & Homberger 基准测试中以 2.98% 的误差差距实现世界纪录的准确性。
无缝扩展：扩展到 1000 个节点以促进计算量大的用例。 NVIDIA cuOpt 的性能优于 SOTA 解决方案，可以解决当今无法实现的创新用例。
实时分析：在 10 秒而不是 20 分钟内路由 1,000 个包裹（快 120 倍），且准确度相同。
在 Launchpad 上免费试用：在 Launchpad 上探索 NVIDIA cuOpt。
省时间：通过动态改道将旅行时间和燃料成本减少 15%。


NVIDIA cuOpt 是一个运筹学优化 API，可帮助您创建复杂的实时车队路线。这些 API 可用于解决具有多个约束的复杂路由问题，并提供动态重新路由、作业调度和机器人路线规划等新功能，同时使用亚秒解算器响应时间。

物流专业人员致力于实时路线优化问题。例如旅行商问题（TSP）、 车辆路径问题（VRP）和 取货和送货问题（PDP）。

机器人公司在机器人部署的规划阶段和连续操作期间都使用 cuOpt。例如，在项目规划阶段，设施的流程布局向系统规划人员通知成功项目 ROI 的吞吐量要求。

22年前，运筹学研究科学家Li和Lim发布了一系列具有挑战性的取货和送货问题（PDP-Pickup and Delivery Problem）,PDP出现在制造、运输零售和物流，甚至救灾领域，PDP是旅行商问题的泛化，同时也是NP-hard问题，这意味着不存在有效算法来找到精确解。随着问题规模的增加，求解时间会呈阶乘增长。NVIDIA cuOpt使用evolution算法和加速计算，每秒分析300亿次动作，打破了世界纪录，并为Li和Lim的挑战找到了合适的解决方案。

AT&T定期派遣3万名技术人员为700个地理区域的1300万客户提供服务，目前，如果在CPU上运行，AT&T的调度优化策略需要一整夜的时间，AT&T希望能够找到一个实时调度接近方案，能不断优化紧急客户需求和整体客户满意度，同时能针对延时和出现的新问题进行调整，借助cuOpt，AT&T可以讲查找解决方案的速度加快100倍，并实时更新调度方案。
