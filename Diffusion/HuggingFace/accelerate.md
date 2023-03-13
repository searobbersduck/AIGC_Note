# [accelerate](https://github.com/huggingface/accelerate/)

```

# step 1
accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  This machine
    AWS (Amazon SageMaker)


# step 2
accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  No distributed training
    multi-CPU
    multi-GPU
    TPU


# step 3
accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]:


# step 4
accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 2
----------------------------------------------------------------------------------------------------------------------------------------------------What is the rank of this machine?
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  0
    1


# step 5
accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 2
----------------------------------------------------------------------------------------------------------------------------------------------------What is the rank of this machine?
0
What is the IP address of the machine that will host the main process?


# step 6
accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 2
----------------------------------------------------------------------------------------------------------------------------------------------------What is the rank of this machine?
0
What is the IP address of the machine that will host the main process? 10.19.206.249
What is the port you will use to communicate with the main process? 10001
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: no
What rendezvous backend will you use? ('static', 'c10d', ...):


# step 7
accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 2
----------------------------------------------------------------------------------------------------------------------------------------------------What is the rank of this machine?
0
What is the IP address of the machine that will host the main process? 10.19.206.229
What is the port you will use to communicate with the main process? 10001
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: no
What rendezvous backend will you use? ('static', 'c10d', ...): c10d
Do you wish to optimize your script with torch dynamo?[yes/NO]:


# step 8
accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 2
----------------------------------------------------------------------------------------------------------------------------------------------------What is the rank of this machine?
0
What is the IP address of the machine that will host the main process? 10.19.206.229
What is the port you will use to communicate with the main process? 10001
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: no
What rendezvous backend will you use? ('static', 'c10d', ...): c10d
Do you wish to optimize your script with torch dynamo?[yes/NO]:yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which dynamo backend would you like to use?
Please select a choice using the arrow or number keys, and selecting with enter
    eager
    aot_eager
 ➔  inductor
    nvfuser
    aot_nvfuser
    aot_cudagraphs
    ofi
    fx2trt
    onnxrt
    ipex



# step 9
accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 2
----------------------------------------------------------------------------------------------------------------------------------------------------What is the rank of this machine?
0
What is the IP address of the machine that will host the main process? 10.19.206.229
What is the port you will use to communicate with the main process? 10001
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: no
What rendezvous backend will you use? ('static', 'c10d', ...): c10d
Do you wish to optimize your script with torch dynamo?[yes/NO]:yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which dynamo backend would you like to use?
inductor
Do you want to customize the defaults sent to torch.compile? [yes/NO]:



# step 10
accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 2
----------------------------------------------------------------------------------------------------------------------------------------------------What is the rank of this machine?
0
What is the IP address of the machine that will host the main process? 10.19.206.249
What is the port you will use to communicate with the main process? 10001
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: no
What rendezvous backend will you use? ('static', 'c10d', ...): c10d
Do you wish to optimize your script with torch dynamo?[yes/NO]:yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which dynamo backend would you like to use?
inductor
Do you want to customize the defaults sent to torch.compile? [yes/NO]: yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which mode do you want to use?
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  default
    reduce-overhead
    max-autotune


# step 11
accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 2
----------------------------------------------------------------------------------------------------------------------------------------------------What is the rank of this machine?
0
What is the IP address of the machine that will host the main process? 10.19.206.249
What is the port you will use to communicate with the main process? 10001
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: no
What rendezvous backend will you use? ('static', 'c10d', ...): c10d
Do you wish to optimize your script with torch dynamo?[yes/NO]:yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which dynamo backend would you like to use?
inductor
Do you want to customize the defaults sent to torch.compile? [yes/NO]: yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which mode do you want to use?
default
Do you want the fullgraph mode or it is ok to break model into several subgraphs? [yes/NO]: yes
Do you want to enable dynamic shape tracing? [yes/NO]: yes
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]:



# step 12
accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 2
----------------------------------------------------------------------------------------------------------------------------------------------------What is the rank of this machine?
0
What is the IP address of the machine that will host the main process? 10.19.206.249
What is the port you will use to communicate with the main process? 10001
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: no
What rendezvous backend will you use? ('static', 'c10d', ...): c10d
Do you wish to optimize your script with torch dynamo?[yes/NO]:yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which dynamo backend would you like to use?
inductor
Do you want to customize the defaults sent to torch.compile? [yes/NO]: yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which mode do you want to use?
default
Do you want the fullgraph mode or it is ok to break model into several subgraphs? [yes/NO]: yes
Do you want to enable dynamic shape tracing? [yes/NO]: yes
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO
----------------------------------------------------------------------------------------------------------------------------------------------------What should be your DeepSpeed's ZeRO optimization stage?
Please select a choice using the arrow or number keys, and selecting with enter
    0
    1
    2
 ➔  3



# step 13
accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 2
----------------------------------------------------------------------------------------------------------------------------------------------------What is the rank of this machine?
0
What is the IP address of the machine that will host the main process? 10.19.206.249
What is the port you will use to communicate with the main process? 10001
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: no
What rendezvous backend will you use? ('static', 'c10d', ...): c10d
Do you wish to optimize your script with torch dynamo?[yes/NO]:yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which dynamo backend would you like to use?
inductor
Do you want to customize the defaults sent to torch.compile? [yes/NO]: yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which mode do you want to use?
default
Do you want the fullgraph mode or it is ok to break model into several subgraphs? [yes/NO]: yes
Do you want to enable dynamic shape tracing? [yes/NO]: yes
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO
----------------------------------------------------------------------------------------------------------------------------------------------------What should be your DeepSpeed's ZeRO optimization stage?
3
----------------------------------------------------------------------------------------------------------------------------------------------------Where to offload optimizer states?
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  none
    cpu
    nvme



# step 14
accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 2
----------------------------------------------------------------------------------------------------------------------------------------------------What is the rank of this machine?
0
What is the IP address of the machine that will host the main process? 10.19.206.249
What is the port you will use to communicate with the main process? 10001
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: no
What rendezvous backend will you use? ('static', 'c10d', ...): c10d
Do you wish to optimize your script with torch dynamo?[yes/NO]:yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which dynamo backend would you like to use?
inductor
Do you want to customize the defaults sent to torch.compile? [yes/NO]: yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which mode do you want to use?
default
Do you want the fullgraph mode or it is ok to break model into several subgraphs? [yes/NO]: yes
Do you want to enable dynamic shape tracing? [yes/NO]: yes
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO
----------------------------------------------------------------------------------------------------------------------------------------------------What should be your DeepSpeed's ZeRO optimization stage?
3
----------------------------------------------------------------------------------------------------------------------------------------------------Where to offload optimizer states?
cpu
----------------------------------------------------------------------------------------------------------------------------------------------------Where to offload parameters?
Please select a choice using the arrow or number keys, and selecting with enter
    none
 ➔  cpu
    nvme



# step 15
accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 2
----------------------------------------------------------------------------------------------------------------------------------------------------What is the rank of this machine?
0
What is the IP address of the machine that will host the main process? 10.19.206.249
What is the port you will use to communicate with the main process? 10001
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: no
What rendezvous backend will you use? ('static', 'c10d', ...): c10d
Do you wish to optimize your script with torch dynamo?[yes/NO]:yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which dynamo backend would you like to use?
inductor
Do you want to customize the defaults sent to torch.compile? [yes/NO]: yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which mode do you want to use?
default
Do you want the fullgraph mode or it is ok to break model into several subgraphs? [yes/NO]: yes
Do you want to enable dynamic shape tracing? [yes/NO]: yes
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO
----------------------------------------------------------------------------------------------------------------------------------------------------What should be your DeepSpeed's ZeRO optimization stage?
3
----------------------------------------------------------------------------------------------------------------------------------------------------Where to offload optimizer states?
cpu
----------------------------------------------------------------------------------------------------------------------------------------------------Where to offload parameters?
cpu
How many gradient accumulation steps you're passing in your script? [1]: 2
Do you want to use gradient clipping? [yes/NO]: NO
Do you want to save 16-bit model weights when using ZeRO Stage-3? [yes/NO]: yes
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which Type of launcher do you want to use?
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  pdsh
    standard
    openmpi
    mvapich




# step 16
accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 2
----------------------------------------------------------------------------------------------------------------------------------------------------What is the rank of this machine?
0
What is the IP address of the machine that will host the main process? 10.19.206.249
What is the port you will use to communicate with the main process? 10001
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: no
What rendezvous backend will you use? ('static', 'c10d', ...): c10d
Do you wish to optimize your script with torch dynamo?[yes/NO]:yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which dynamo backend would you like to use?
inductor
Do you want to customize the defaults sent to torch.compile? [yes/NO]: yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which mode do you want to use?
default
Do you want the fullgraph mode or it is ok to break model into several subgraphs? [yes/NO]: yes
Do you want to enable dynamic shape tracing? [yes/NO]: yes
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO
----------------------------------------------------------------------------------------------------------------------------------------------------What should be your DeepSpeed's ZeRO optimization stage?
3
----------------------------------------------------------------------------------------------------------------------------------------------------Where to offload optimizer states?
cpu
----------------------------------------------------------------------------------------------------------------------------------------------------Where to offload parameters?
cpu
How many gradient accumulation steps you're passing in your script? [1]: 2
Do you want to use gradient clipping? [yes/NO]: NO
Do you want to save 16-bit model weights when using ZeRO Stage-3? [yes/NO]: yes
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which Type of launcher do you want to use?
standard
How many GPU(s) should be used for distributed training? [1]:3
----------------------------------------------------------------------------------------------------------------------------------------------------Do you wish to use FP16 or BF16 (mixed precision)?
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  no
    fp16
    bf16
    fp8



# step 17

accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 2
----------------------------------------------------------------------------------------------------------------------------------------------------What is the rank of this machine?
0
What is the IP address of the machine that will host the main process? 10.19.206.249
What is the port you will use to communicate with the main process? 10001
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: no
What rendezvous backend will you use? ('static', 'c10d', ...): c10d
Do you wish to optimize your script with torch dynamo?[yes/NO]:yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which dynamo backend would you like to use?
inductor
Do you want to customize the defaults sent to torch.compile? [yes/NO]: yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which mode do you want to use?
default
Do you want the fullgraph mode or it is ok to break model into several subgraphs? [yes/NO]: yes
Do you want to enable dynamic shape tracing? [yes/NO]: yes
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO
----------------------------------------------------------------------------------------------------------------------------------------------------What should be your DeepSpeed's ZeRO optimization stage?
3
----------------------------------------------------------------------------------------------------------------------------------------------------Where to offload optimizer states?
cpu
----------------------------------------------------------------------------------------------------------------------------------------------------Where to offload parameters?
cpu
How many gradient accumulation steps you're passing in your script? [1]: 2
Do you want to use gradient clipping? [yes/NO]: NO
Do you want to save 16-bit model weights when using ZeRO Stage-3? [yes/NO]: yes
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: yes
----------------------------------------------------------------------------------------------------------------------------------------------------Which Type of launcher do you want to use?
standard
How many GPU(s) should be used for distributed training? [1]:3
----------------------------------------------------------------------------------------------------------------------------------------------------Do you wish to use FP16 or BF16 (mixed precision)?
bf16
accelerate configuration saved at /home/rtx/.cache/huggingface/accelerate/default_config.yaml


```

<br>

## TODO

- [ ] step 6: What rendezvous backend will you use? ('static', 'c10d', ...)
  - [ ] [[源码解析] PyTorch 分布式之弹性训练(1) --- 总体思路](https://www.cnblogs.com/rossiXYZ/p/15718043.html)
  - [ ] [[源码解析] PyTorch 分布式之弹性训练(2)---启动&单节点流程](https://www.cnblogs.com/rossiXYZ/p/15725911.html)
  - [ ] [[源码解析] PyTorch 分布式之弹性训练(3)---代理](https://www.cnblogs.com/rossiXYZ/p/15728861.html)
  - [ ] [[源码解析] PyTorch 分布式之弹性训练(4)---Rendezvous 架构和逻辑](https://cloud.tencent.com/developer/article/1926432)
  - [ ] [云原生的弹性 AI 训练系列之二：PyTorch 1.9.0 弹性分布式训练的设计与实现](https://mp.weixin.qq.com/s/hlOYLKSHFDZWN21AsUn6bg)
  - [ ] [PyTorch Elastic源码阅读](https://zhuanlan.zhihu.com/p/408382623)

- [ ] step 9: Which dynamo backend would you like to use?

- [ ] step 13: Where to offload optimizer states?
  - [x] nvme: 这个选项中，nvme指的是将优化器状态offload到硬盘
    * [什么是 NVMe SSD 技术？](https://www.kingston.com/cn/ssd/what-is-nvme-ssd-technology)
      * NVMe 技术带来出众的存储空间、速度和兼容性。由于 NVMe 利用 PCIe 插槽，它传输的数据量是同等 SATA 产品的 25 倍。
    * [NVMe SSD和普通ssd的区别](https://www.crucial.cn/articles/about-ssd/difference-between-nvme-ssd-and-sata-ssd)
      * 首先要知道什么是NVMe，NVM Express（NVMe），或称非易失性内存主机控制器接口规范(Non-Volatile Memory express),是一个逻辑设备接口规范。
      * NVMe的优势，对比于SATA SSD:
        * 性能有数倍的提升
        * 可大幅降低延迟

- [ ] step 15: Which Type of launcher do you want to use?
  - [ ] [mvapich](https://baike.baidu.com/item/mvapich/10804942)
  - [ ] [并行计算入门：mpich的安装与测试](https://blog.ailemon.net/2018/03/27/parallel-computing-introduction-mpich-install-and-test/)
  - [ ] [MPI(OpenMPI和MPICH（IntelMPI、MVAPICH)）和OpenMP](https://blog.csdn.net/weixin_44004788/article/details/117388559)
  - [ ] [【深度学习】分布式训练常用技术总结](https://my.oschina.net/oneflow/blog/5088758)
    * 内容量有点大，看完再分类
  - [ ] [WRITING DISTRIBUTED APPLICATIONS WITH PYTORCH](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
    * 你要的答案这里都有，细看

