# ChatGLM

找时间详细尝试，很有意义的中文开源预训练模型。

<br>

## BLOG

### [清华系千亿基座对话模型ChatGLM启动内测，开源单卡版模型](https://www.jiqizhixin.com/articles/2023-03-15-3)
* 由清华技术成果转化的公司智谱 AI 基于 GLM-130B 千亿基座模型的 ChatGLM 现已开启邀请制内测。
* 千亿基座模型 GLM-130B, 不同于 BERT、GPT-3 以及 T5 的架构，是一个包含多目标函数的自回归预训练模型。

<br>

### [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
* ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需 6GB 显存）。 ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答。

<br>

### [ChatGLM：千亿基座的对话模型开启内测⸺对应单卡版本开源](https://chatglm.cn/blog)

<br>

### [ChatGLM-6B Chatbot](https://huggingface.co/spaces/wangrongsheng/ChatGLM)

<br><br>

## Code

src/transformers/models/auto/auto_factory.py    from_pretrained

src/transformers/modeling_utils.py      PreTrainedModel     from_pretrained

```
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        kwargs["_from_auto"] = True
        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "resume_download",
            "revision",
            "subfolder",
            "use_auth_token",
        ]
        hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
        if not isinstance(config, PretrainedConfig):
            kwargs_copy = copy.deepcopy(kwargs)
            # ensure not to pollute the config object with torch_dtype="auto" - since it's
            # meaningless in the context of the config object - torch.dtype values are acceptable
            if kwargs_copy.get("torch_dtype", None) == "auto":
                _ = kwargs_copy.pop("torch_dtype")

            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                trust_remote_code=trust_remote_code,
                **hub_kwargs,
                **kwargs_copy,
            )
        if hasattr(config, "auto_map") and cls.__name__ in config.auto_map:
            if not trust_remote_code:
                raise ValueError(
                    f"Loading {pretrained_model_name_or_path} requires you to execute the modeling file in that repo "
                    "on your local machine. Make sure you have read the code there to avoid malicious use, then set "
                    "the option `trust_remote_code=True` to remove this error."
                )
            if hub_kwargs.get("revision", None) is None:
                logger.warning(
                    "Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure "
                    "no malicious code has been contributed in a newer revision."
                )
            class_ref = config.auto_map[cls.__name__]
            module_file, class_name = class_ref.split(".")
            model_class = get_class_from_dynamic_module(
                pretrained_model_name_or_path, module_file + ".py", class_name, **hub_kwargs, **kwargs
            )
            model_class.register_for_auto_class(cls.__name__)
            return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )
```

