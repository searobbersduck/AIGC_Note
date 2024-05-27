

## Setup Env

### Run Container

docker images: `https://registry.ngc.nvidia.com/orgs/ea-bignlp/teams/ga-participants/containers/nemofw-training/tags`

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name NF_240301 -p 6022:22 -p 6006:6006 -p 6064:6064 -p 6888:8888 -v /data/weidongz/docker_workspace:/workspace nvcr.io/ea-bignlp/ga-participants/nemofw-training:24.03.01 bash
```


### 

```
sudo docker run --rm -ti --net=host --gpus all -v /gpt3_data:/gpt3_data nvcr.io/nvidian/bignlp-train:24.01-nemofw-nightly-mcore
export PYTHONPATH=/opt/NeMo:/opt/Megatron-LM

NSYS_ARGS="training.model.nsys_profile.enabled=True"
 
PROFILE_LAUNCHER="nsys profile \
    -s none -o /gpt3_data/profiles/2401 -t cuda,nvtx \
    --nic-metrics=true --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop"

${PROFILE_LAUNCHER} python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py \
	${NSYS_ARGS} \
	training=gpt3/5b stages=["training"] numa_mapping.enable=True \
	launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts/ \
	base_results_dir=/gpt3_data/results \
	env_vars.NCCL_NVLS_ENABLE=1 \
	cluster.gpus_per_task=null cluster.gpus_per_node=null cluster.job_name_prefix="nv-test-2401" \
	training.exp_manager.create_checkpoint_callback=False \
	training.run.name="gpt3_5b" \
	training.trainer.log_every_n_steps=1 \
	training.trainer.num_nodes=1 \
	training.trainer.devices=8 \
	cluster_type=interactive \
	++training.cluster_type=BCP \
	training.model.global_batch_size=2048 \
	training.model.micro_batch_size=4 \
	training.model.data.data_impl="mock" \
	training.model.data.data_prefix=[] \
	training.model.tensor_model_parallel_size=1 \
	training.model.pipeline_model_parallel_size=1 \
	training.model.fp8=False \
	training.model.transformer_engine=True \
	training.model.fp8_e4m3=True
```