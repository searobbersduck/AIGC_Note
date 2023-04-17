## 

## cuLitho

* [造芯片的“计算光刻”，了解一下](https://www.eet-china.com/news/202303227503.html)
* [英伟达发布cuLitho，计算光刻提升40倍！阿斯麦台积电站台支持！](https://zhuanlan.zhihu.com/p/616512305)

## TRT/Triton

* [Model Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_analyzer.html)
* [NVIDIA Triton Management Service Demo](https://www.youtube.com/watch?v=Gtko-PpIntk)
* [NVIDIA 发布TensorRT 8.6](https://www.zhihu.com/zvideo/1622980418319360000)


## CV-CUDA

* [CV-CUDA](https://cvcuda.github.io/)
* [VideoProcessingFramework（VPF）GPU视频流解码方案](https://zhuanlan.zhihu.com/p/615372907)


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
