# RLHF

## BLOG

### [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)
### [ChatGPT 背后的“功臣”——RLHF 技术详解](https://juejin.cn/post/7188393606544097335)
* OpenAI 推出的 ChatGPT 对话模型掀起了新的 AI 热潮，它面对多种多样的问题对答如流，似乎已经打破了机器和人的边界。这一工作的背后是大型语言模型 (Large Language Model，LLM) 生成领域的新训练范式：RLHF (Reinforcement Learning from Human Feedback) ，即以强化学习方式依据人类反馈优化语言模型。
* LLM 根据人类输入提示 (prompt) 生成多样化文本的能力令人印象深刻。然而，对生成结果的评估是主观和依赖上下文的，例如，我们希望模型生成一个有创意的故事、一段真实的信息性文本，或者是可执行的代码片段，这些结果难以用现有的基于规则的文本生成指标 (如 BLUE 和 ROUGE) 来衡量。除了评估指标，现有的模型通常以预测下一个单词的方式和简单的损失函数 (如交叉熵) 来建模，没有显式地引入人的偏好和主观意见。
* 如果我们 **用生成文本的人工反馈作为性能衡量标准，或者更进一步用该反馈作为损失来优化模型**，那不是更好吗？这就是 RLHF 的思想：使用强化学习的方式直接优化带有人类反馈的语言模型。RLHF 使得在一般文本数据语料库上训练的语言模型能和复杂的人类价值观对齐。



### [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
* 这是InstructGPT对应blog的名字，其实也是RLHF的作用所在。LLM本身已经具备了足够强大的能力。RLHF只是让模型向人类的偏好对齐（alignment）。
> We’ve trained language models that are much better at following user intentions than GPT-3 while also making them more truthful and less toxic, using techniques developed through our alignment research.
>
> 我们训练的语言模型比 GPT-3 更善于遵循用户意图，同时使用通过我们的对齐研究开发的技术使它们更真实、毒性更小。
>
> The OpenAI API is powered by GPT-3 language models which can be coaxed to perform natural language tasks using carefully engineered text prompts. But these models can also generate outputs that are untruthful, toxic, or reflect harmful sentiments. This is in part because GPT-3 is trained to predict the next word on a large dataset of Internet text, rather than to safely perform the language task that the user wants. In other words, these models aren’t aligned with their users.


### [Learning from human preferences](https://openai.com/research/learning-from-human-preferences)
> One step towards building safe AI systems is to remove the need for humans to write goal functions, since using a simple proxy for a complex goal, or getting the complex goal a bit wrong, can lead to undesirable and even dangerous behavior. In collaboration with DeepMind’s safety team, we’ve developed an algorithm which can infer what humans want by being told which of two proposed behaviors is better.
>
> 构建安全 AI 系统的一个步骤是消除人类编写目标函数的需要，因为对复杂目标使用简单代理，或者将复杂目标弄错一点，可能会导致不良甚至危险的行为。 通过与 DeepMind 的安全团队合作，我们开发了一种算法，可以通过告知两种提议的行为中哪一种更好来推断人类的需求。
>
* 与这个主题相关，我们可以思考一下强化学习在LLM中使用时的关联性：
  * 不需要人来编写目标函数；
  * 避免了对复杂目标使用简单代理或者将复杂目标弄错；
  * 可以告知更好的行为。


### [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://www.anthropic.com/index/training-a-helpful-and-harmless-assistant-with-reinforcement-learning-from-human-feedback)


### [InstructGPT 浅析](https://www.qin.news/instructgpt/)
* OpenAI 的 GPT-3，本质上是基于上下文的生成模型。 这意味着当给 GPT-3 某种上下文内容时，它会尝试填充其余的内容。例如，如果给它句子的前半部分，它将尝试扩充推测句子的下半部分，给一个问句，会给出回答，原来的初衷只能用来生成文字，但大道至简，一通百通，既然可以理解文字，那么一切基于自然语言的任务都有了被 GTP-3 涉足的可能性，如果给它上一篇论文的前半部分，它将生成其余的论文，如果给他一段代码的描述，在一定条件下，GPT-3 就能够给出代码的具体内容。

* 虽然今天的大规模预训练语言模型 GPT-3，在广泛的自然语言处理（NLP）任务中取得了惊人的性能，但这些模型经常会生成与用户期望不一致的非预期输出。此外，这些产出也可能带有偏见、虚假或有害、可能造成负面的社会影响的信息。

* 通过 《Training language models to follow instructions with human feedback》论文中的训练方案得到的 InstructGPT，可以很大程度上缓解上述的问题。