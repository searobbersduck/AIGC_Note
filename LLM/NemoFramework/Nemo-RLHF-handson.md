# Nemo RLHF Hands-on

**Ref: &ensp;  [5.16. Reinforcement Learning from Human Feedback](https://github.com/NVIDIA/NeMo-Megatron-Launcher#516-reinforcement-learning-from-human-feedback)**
* NeMo-RLHF supports **only GPT models** and implements the Proximal Policy Optimization (PPO) algorithm.
* Support for **other models** and RL algorithms will be added **in future releases**.

<br>

## Datasets

* **Install dependency**
* ref:  `https://stackoverflow.com/questions/48734119/git-lfs-is-not-a-git-command-unclear`

    ```
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh
    sudo apt-get install git-lfs
    ```

* **REF: [NeMo Framework Reward Modeling](https://gitlab-master.nvidia.com/ai-sae/nemo-llm-playbooks/-/blob/dev-RewardModeling/llm_model_customization/Customization_-_Nemo_Framework_Train_Reward_Model.md)**
    * **[Step 1: Download the 2B GPT model](https://gitlab-master.nvidia.com/ai-sae/nemo-llm-playbooks/-/blob/dev-RewardModeling/llm_model_customization/Customization_-_Nemo_Framework_Train_Reward_Model.md#step-1-download-the-2b-gpt-model)**: 
        ```
        mkdir -p /workspace/data/nemo_rlhf/data
        cd /workspace/data/nemo_rlhf/data
        git  lfs  clone https://huggingface.co/nvidia/GPT-2B-001
        mkdir -p /workspace/data/nemo_rlhf/data/models
        mv GPT-2B-001/GPT-2B-001_bf16_tp1.nemo models/GPT-2B-001_bf16_tp1.nemo
        tar -xvf models/GPT-2B-001_bf16_tp1.nemo

        ls models/GPT-2B-001_bf16_tp1.nemo *.model
        ```
        ![Alt text](image-1.png)

        ![Alt text](image-2.png)

        ![Alt text](image-3.png)
    * **[Step 2: Dataset processing](https://gitlab-master.nvidia.com/ai-sae/nemo-llm-playbooks/-/blob/dev-RewardModeling/llm_model_customization/Customization_-_Nemo_Framework_Train_Reward_Model.md#step-2-dataset-processing)**: 
        ```
        pip install datasets
        ```
        ![Alt text](image-4.png)

        **Task: Create process_anthropic_hh.py with following content for converting Anthropic hh-rlhf dataset to Nemo Framework reward model training jsonl format.**

        **Task: Convert Anthropic hh-rlhf dataset to Nemo Framework reward model training jsonl format**

        ```
        touch process_anthropic_hh.py
        # 填入如下内容并运行
        python process_anthropic_hh.py
        ```

        ```
        """A script to process the Anthropic Dataset"""
        import argparse
        import json
        import warnings
        from pathlib import Path

        from datasets import load_dataset


        def prepare_args():
            parser = argparse.ArgumentParser(description="generate dataset")
            parser.add_argument(
                "--output-dir", type=str, default="./",
            )
            return parser.parse_args()


        START_PROMPT_FORMAT = "User: {body}\n\nAssistant: {response}"
        PROMPT_CONTINUATION_FORMAT = "{text}\n\nUser: {body}\n\nAssistant: {response}"


        def process_hh(split):
            if split == "validation":
                warnings.warn("anthropic HHH has no validation set, so using test set instead")
                split = "test"

            ds = load_dataset("Anthropic/hh-rlhf")[split]

            def convert_string_format(string):
                split_string = string.split("\n\nHuman: ")

                string_to_use = ""
                prompt_string_to_use = ""

                for item in split_string:
                    if len(item) == 0:
                        continue

                    output = item.split("\n\nAssistant: ")

                    if len(output) != 2:
                        return None
                    else:
                        body, response = output

                    if len(string_to_use) == 0:
                        prompt_string_to_use = START_PROMPT_FORMAT.format(body=body, response="")
                        string_to_use = START_PROMPT_FORMAT.format(body=body, response=response)
                    else:
                        prompt_string_to_use = PROMPT_CONTINUATION_FORMAT.format(text=string_to_use, body=body, response="")
                        string_to_use = PROMPT_CONTINUATION_FORMAT.format(text=string_to_use, body=body, response=response)

                # for prompt, remove the space at the end
                return string_to_use, prompt_string_to_use[:-1]

            list_of_dicts = []

            chosen = list(map(convert_string_format, ds["chosen"]))
            rejected = list(map(convert_string_format, ds["rejected"]))

            for c, r in zip(chosen, rejected):
                if c is None or r is None:
                    continue

                chosen_response, chosen_prompt = c
                rejected_response, rejected_prompt = r

                if chosen_prompt != rejected_prompt:
                    continue

                comparison_dict = {
                    "prompt": chosen_prompt,
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                }

                list_of_dicts.append(comparison_dict)

            return list_of_dicts


        def convert_list_of_dict_to_jsonl(list_of_dict):
            return "\n".join(json.dumps(item) for item in list_of_dict)


        def save_dataset(list_of_dict, split, save_dir):
            prompts_to_save = convert_list_of_dict_to_jsonl({"text": item["prompt"]} for item in list_of_dict)

            with open(Path(save_dir) / f"{split}_prompts.jsonl", "w") as f:
                f.write(prompts_to_save)

            comparisons_to_save = []

            for item in list_of_dict:
                comparisons_to_save.append({"text": item["chosen"]})
                comparisons_to_save.append({"text": item["rejected"]})

            comparisons_to_save = convert_list_of_dict_to_jsonl(comparisons_to_save)

            with open(Path(save_dir) / f"{split}_comparisons.jsonl", "w") as f:
                f.write(comparisons_to_save)


        if __name__ == "__main__":
            args = prepare_args()

            for split in ["train", "test"]:
                list_of_dicts = process_hh(split)
                save_dataset(list_of_dicts, split, args.output_dir)

        ```

        ![Alt text](image-5.png)

        **Task: Convert dataset from jsonl to mmap binary format**

    ```
    # cd ${WORK_DIR}
    cd /workspace/data/nemo_rlhf/data

    mkdir datasets
    python3 /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input "train_comparisons.jsonl" \
    --output-prefix "./datasets/hh_comparison_train" \
    --tokenizer-model 2053796188904e679f7e2754a2a1f280_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --tokenizer-library=sentencepiece \
    --json-keys text \
    --dataset-impl mmap \
    --workers 30 \
    --chunk_size=100 \
    --append-eod

    python3 /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input "test_comparisons.jsonl" \
    --output-prefix "./datasets/hh_comparison_test" \
    --tokenizer-model 2053796188904e679f7e2754a2a1f280_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --tokenizer-library=sentencepiece \
    --json-keys text \
    --dataset-impl mmap \
    --workers 30 \
    --chunk_size=100 \
    --append-eod

    ```

    ![Alt text](image-6.png)
    ![Alt text](image-7.png)

    ```
    ls datasets
    ```

    ![Alt text](image-8.png)

    **[Step 3: Customize config file for reward model training](https://gitlab-master.nvidia.com/ai-sae/nemo-llm-playbooks/-/blob/dev-RewardModeling/llm_model_customization/Customization_-_Nemo_Framework_Train_Reward_Model.md#step-3-customize-config-file-for-reward-model-training)**
    
    **Task: Customize the default config file /opt/nemo-rlhf examples/nlp/gpt/conf/training_rm.yaml**
    * It's highly recommended to use 1 epoch for reward model training to avoid overfitting.
    * Make sure "always_save_nemo: True" and "save_nemo_on_train_end: True" in the config file.
    * Add "rampup_batch_size: null" in the config file as following: (This is just a workaround for 23.07 container, should not used in the future release.)