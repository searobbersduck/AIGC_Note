## Task

- [ ] 前期准备
  - [ ] 熟悉开发过程中涉及到的各个模块以及如何在MLM的基础上进行模型开发；
- [ ] 模型开发
  - [ ] 实现基础模型功能
    - [ ] 子模块功能验证；
      - [ ] vision encoder的实现在Nemo里；
      - [ ] CP/TP的实现在TE里；
    - [ ] 由子模块集成基础模型；
    - [ ] CP/TP引入；
    - [ ] FP8引入；
  - [ ] 模型测试用例
    - [ ] 测试CP
    - [ ] 测试TP
- [ ] Datasets开发
  - [ ] Datasets
  - [ ] 数据预处理模块；
  - [ ] DataLoader
- [ ] Weights Converter
- [ ] 模型E2E功能验证；


<br>

## gitlab配置

```
cd existing_repo
git remote add origin https://gitlab-master.nvidia.com/weidongz/nvidia-dit.git
git branch -M main
git push -uf origin main
```

## 材料

* [DiT Performance](https://docs.google.com/spreadsheets/d/1plPV4y9kT7sARtebr2dgZ-vVLvcp-RrO-n8COvz5gC0/edit#gid=1628491628)
  * seq 16k, torch.compile vs baseline (1.2倍)
  * TODO: TFLOPS计算

### Doc

* [Adding DIT model into Mcore models](https://docs.google.com/document/d/17cQuH9f-UU6hpShxjAH28ylIT5CIqGSDBn44ikknp3A/edit#heading=h.l1o42r1jckse)
* [video foundation model training framwork](https://docs.google.com/document/d/1YutfxSgUKVIwF9MwsYqsabF6uj6MRVLqWN1adlbgpYQ/edit#heading=h.x78rhrd4unxv)

### Table
* [cosmos training benchmarking](https://docs.google.com/spreadsheets/d/1A2NOC3HHqFtw6fHxpKelepExJFQUkm4hBZzONnpHrGw/edit#gid=514599317)
* 


## MOE材料

### PPT

* [2D-TP MoE](https://docs.google.com/presentation/d/1yG6M0DN8BtBBYdo-wvzqIc_8YxJLRBYX/edit#slide=id.g25f8b8fb4d8_0_88)
  * [new 2d-tp-moe design](https://docs.google.com/document/d/1Pz1_szEp9vRu4s1QpMgceqKz63roSe1v/edit)
  * 

### 表格
* [[MoE Perf] Mixtral 8x7B performance tracker](https://docs.google.com/spreadsheets/d/11Wv1uOO5ETri7CE74NSe_joh_f_GsjmKBPeFdQHHZL4/edit#gid=1109706923)
* 

