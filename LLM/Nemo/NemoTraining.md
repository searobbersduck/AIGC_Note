```
root@3906ac915726:/workspace# find / -name config.yaml
/usr/local/lib/python3.8/dist-packages/hydra/test_utils/configs/config.yaml
/usr/local/lib/python3.8/dist-packages/hydra/test_utils/configs/completion_test/config.yaml
/opt/bignlp/bignlp-hp-tool/conf/config.yaml
/opt/bignlp/bignlp-scripts/bignlp/collections/dataprep_scripts/pile_bert_dataprep/conf/config.yaml
/opt/bignlp/bignlp-scripts/bignlp/collections/dataprep_scripts/pile_dataprep/conf/config.yaml
/opt/bignlp/bignlp-scripts/conf/config.yaml
/opt/bignlp/NeMo/examples/asr/conf/config.yaml
```

<br>

**`/opt/bignlp/bignlp-scripts/conf/config.yaml`**
```
defaults:
  - _self_
  - cluster: bcm  # Leave it as bcm even if using bcp. It will be ignored for bcp.
  - data_preparation: gpt3/download_gpt3_pile
  - training: gpt3/5b
  - conversion: gpt3/convert_gpt3
  - fine_tuning: null
  - prompt_learning: null
  - adapter_learning: null
  - ia3_learning: null
  - evaluation: gpt3/evaluate_all
  - export: gpt3/export_gpt3
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null

debug: False

stages:
  - training
  - conversion
  - evaluation
  - export

cluster_type: bcm  # bcm or bcp. If bcm, it must match - cluster above.
bignlp_path: ???  # Path should end with bignlp-scripts
data_dir: ${bignlp_path}/data  # Location to store and read the data.
base_results_dir: ${bignlp_path}/results  # Location to store the results, checkpoints and logs.
container_mounts: # List of additional paths to mount to container. They will be mounted to same path.
  - null
container: nvcr.io/ea-bignlp/bignlp-training:23.01-py3

wandb_api_key_file: null  # File where the w&B api key is stored. Key must be on the first line.

env_vars:
  NCCL_TOPO_FILE: null # Should be a path to an XML file describing the topology
  UCX_IB_PCI_RELAXED_ORDERING: null # Needed to improve Azure performance
  NCCL_IB_PCI_RELAXED_ORDERING: null # Needed to improve Azure performance

```

<br>

```
root@3906ac915726:/opt/bignlp/bignlp-scripts/conf# tree -L 3
.
├── adapter_learning
│   ├── gpt3
│   │   └── squad.yaml
│   └── t5
│       └── squad.yaml
├── cluster
│   └── bcm.yaml
├── config.yaml
├── conversion
│   ├── gpt3
│   │   └── convert_gpt3.yaml
│   ├── mt5
│   │   └── convert_mt5.yaml
│   └── t5
│       └── convert_t5.yaml
├── data_preparation
│   ├── bert
│   │   └── download_bert_pile.yaml
│   ├── generic
│   │   └── custom_dataset.yaml
│   ├── gpt3
│   │   └── download_gpt3_pile.yaml
│   ├── mt5
│   │   └── download_mc4.yaml
│   └── t5
│       └── download_t5_pile.yaml
├── evaluation
│   ├── adapter_gpt3
│   │   └── squad.yaml
│   ├── adapter_t5
│   │   └── squad.yaml
│   ├── gpt3
│   │   ├── evaluate_all.yaml
│   │   └── evaluate_lambada.yaml
│   ├── ia3_gpt3
│   │   └── squad.yaml
│   ├── ia3_t5
│   │   └── squad.yaml
│   ├── mt5
│   │   ├── custom_task.yaml
│   │   └── xquad.yaml
│   ├── prompt_gpt3
│   │   └── squad.yaml
│   ├── prompt_mt5
│   │   └── squad.yaml
│   ├── prompt_t5
│   │   └── squad.yaml
│   └── t5
│       ├── custom_task.yaml
│       └── squad.yaml
├── export
│   ├── gpt3
│   │   └── export_gpt3.yaml
│   ├── mt5
│   │   └── export_mt5.yaml
│   └── t5
│       └── export_t5.yaml
├── fine_tuning
│   ├── mt5
│   │   ├── custom_task.yaml
│   │   └── xquad.yaml
│   └── t5
│       ├── custom_task.yaml
│       └── squad.yaml
├── ia3_learning
│   ├── gpt3
│   │   └── squad.yaml
│   └── t5
│       └── squad.yaml
├── prompt_learning
│   ├── gpt3
│   │   └── squad.yaml
│   ├── mt5
│   │   └── squad.yaml
│   └── t5
│       └── squad.yaml
└── training
    ├── bert
    │   ├── 100b.yaml
    │   ├── 110m.yaml
    │   ├── 20b.yaml
    │   └── 4b.yaml
    ├── gpt3
    │   ├── 126m.yaml
    │   ├── 175b.yaml
    │   ├── 20b.yaml
    │   ├── 40b.yaml
    │   └── 5b.yaml
    ├── mt5
    │   ├── 11b.yaml
    │   ├── 170m.yaml
    │   ├── 23b.yaml
    │   ├── 390m.yaml
    │   └── 3b.yaml
    └── t5
        ├── 11b.yaml
        ├── 220m.yaml
        ├── 23b.yaml
        ├── 3b.yaml
        └── 41b.yaml

```

<br>

**`root@3906ac915726:/opt/bignlp/bignlp-scripts/conf# tree -L 3 training`**
```
root@3906ac915726:/opt/bignlp/bignlp-scripts/conf# tree -L 3 training
training
├── bert
│   ├── 100b.yaml
│   ├── 110m.yaml
│   ├── 20b.yaml
│   └── 4b.yaml
├── gpt3
│   ├── 126m.yaml
│   ├── 175b.yaml
│   ├── 20b.yaml
│   ├── 40b.yaml
│   └── 5b.yaml
├── mt5
│   ├── 11b.yaml
│   ├── 170m.yaml
│   ├── 23b.yaml
│   ├── 390m.yaml
│   └── 3b.yaml
└── t5
    ├── 11b.yaml
    ├── 220m.yaml
    ├── 23b.yaml
    ├── 3b.yaml
    └── 41b.yaml

```

<br>

**`/opt/bignlp/bignlp-scripts/conf/training/gpt3/175b.yaml`**

```

```