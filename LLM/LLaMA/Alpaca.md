# [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)

## BLOG: [Alpaca: A Strong Open-Source Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)
> Instruction-following models such as GPT-3.5 (text-davinci-003), ChatGPT, Claude, and Bing Chat have become increasingly powerful.
>
> We are releasing our findings about an instruction-following language model, dubbed Alpaca, which is fine-tuned from Meta’s LLaMA 7B model. We train the Alpaca model on 52K instruction-following demonstrations generated in the style of self-instruct using text-davinci-003. Alpaca shows many behaviors similar to OpenAI’s text-davinci-003, but is also surprisingly small and easy/cheap to reproduce.
>
> We emphasize that Alpaca is intended only for academic research and any commercial use is prohibited. There are three factors in this decision: First, Alpaca is based on LLaMA, which has a non-commercial license, so we necessarily inherit this decision. Second, the instruction data is based OpenAI’s text-davinci-003, whose terms of use prohibit developing models that compete with OpenAI. Finally, we have not designed adequate safety measures, so Alpaca is not ready to be deployed for general use.
>
> ### Training recipe
>
> Alpaca is a language model fine-tuned using supervised learning from a LLaMA 7B model on 52K instruction-following demonstrations generated from OpenAI’s text-davinci-003.
>
> The figure below illustrates how we obtained the Alpaca model. For the data, we generated instruction-following demonstrations by building upon the self-instruct method. We started with the 175 human-written instruction-output pairs from the self-instruct seed set. We then prompted text-davinci-003 to generate more instructions using the seed set as in-context examples. We improved over the self-instruct method by simplifying the generation pipeline (see details in GitHub) and significantly reduced the cost. Our data generation process results in 52K unique instructions and the corresponding outputs, which costed less than $500 using the OpenAI API. (这里有图示可以参见原文)
>
> Equipped with this instruction-following dataset, we then fine-tuned the LLaMA models using Hugging Face’s training framework, taking advantage of techniques like Fully Sharded Data Parallel and mixed precision training. Fine-tuning a 7B LLaMA model took 3 hours on 8 80GB A100s, which costs less than $100 on most cloud compute providers.
>
> ### Preliminary evaluation
>
* 具体的评估可以参见这篇blog，但我通过[interactive demo](https://crfm.stanford.edu/alpaca/)尝试了一下，基本完全不具备中文能力。基本所有的中文问题都被当成了翻译任务。
* 本来以为这可以作为chatgpt这类任务蒸馏过程的想象，但初步尝试了一下，能力比chatgpt相差甚远。目前看来想象空间并不大。
* 不过能开源，给流程，也算是很值得尊重的工作。虽然还没有提供预训练权重。

> ### Assets released
>
> Demo: An [interactive demo](https://crfm.stanford.edu/alpaca/) for everyone to try out Alpaca.
> 
> Data: [52K demonstrations](https://github.com/tatsu-lab/stanford_alpaca#data-release) used to fine-tune Alpaca.
> 
> Data generation process: the code for [generating the data](https://github.com/tatsu-lab/stanford_alpaca#data-generation-process).
> 
> Hyperparameters: for fine-tuning the model using the Hugging Face API.