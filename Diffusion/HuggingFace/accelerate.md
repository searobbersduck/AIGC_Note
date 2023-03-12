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

