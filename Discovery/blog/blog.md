# BLOG

<br>

## [ChatGPT for Drug Development? How Large Language Models Are Changing Drug Discovery](https://vial.com/blog/articles/chatgpt-for-drug-development-how-large-language-models-are-changing-drug-discovery/&https://vial.com/blog/articles/chatgpt-for-drug-development-how-large-language-models-are-changing-drug-discovery/?utm_source=organic)

> The launch of ChatGPT was met with a flurry of excitement and the publication of several studies to determine its strengths and potential applications in healthcare and clinical research, as well as its limitations, risks, and consequences of widespread, unchecked use. The tone appears to be cautious optimism as researchers raise ethical concerns, the risk of artificial hallucinations, and the uncertainty of how ChatGPT will perform in real-world situations, amongst others. In addition, Arif et al. (2023) cite a lack of critical thinking and the inclusion of redundant information as reasons why scientific experts and journals reject ChatGPT.
>
> ChatGPT 的推出引起了一阵兴奋，并发表了几项研究以确定其在医疗保健和临床研究中的优势和潜在应用，以及其局限性、风险和广泛、未经检查的使用的后果。 基调似乎是谨慎乐观的，因为研究人员提出了道德问题、人为幻觉的风险以及 ChatGPT 在现实世界中的表现的不确定性等。 此外，Arif 等人。 (2023) 引用缺乏批判性思维和包含冗余信息作为科学专家和期刊拒绝 ChatGPT 的原因。
>
> **The Application of ChatGPT in Drug Development**
>
> Medical institutions and clinical research organizations (CROs) can transform the landscape of clinical research by leveraging AI algorithms and machine learning. The application of AI models opens up opportunities to analyze massive amounts of unstructured data, expanding data-based research and simplifying the research process. To support drug development, ChatGPT can potentially predict drug-target interactions, speed up the identification of potential drug candidates, and help identify opportunities for drug repurposing.
>
> 医疗机构和临床研究组织 (CRO) 可以利用人工智能算法和机器学习来改变临床研究的格局。 人工智能模型的应用为分析大量非结构化数据、扩展基于数据的研究和简化研究过程提供了机会。 为了支持药物开发，ChatGPT 可以潜在地预测药物与靶标的相互作用，加快潜在候选药物的识别，并帮助确定药物再利用的机会。
>
> **Limitations and Risks of Using ChatGPT for Drug Discovery**
> 
>
> ChatGPT’s output can be incorrect or biased, e.g., citing non-existent references or perpetuating sexist stereotypes
>
> ChatGPT 的输出可能不正确或有偏见，例如，引用不存在的参考资料或延续性别歧视的刻板印象
> 
> erroneous ChatGPT outputs used to train future iterations of the model will be amplified
> 
> 用于训练模型未来迭代的错误 ChatGPT 输出将被放大
> 
> inaccuracies in ChatGPT outputs could fuel the spread of misinformation.
> 
> ChatGPT 输出的不准确性可能助长错误信息的传播。
> 
> risk of introducing errors and plagiarized content into publications which in the long run may negatively impact research and health policy decisions 
> 
> 将错误和剽窃内容引入出版物的风险，从长远来看可能会对研究和卫生政策决策产生负面影响
> 
> users can circumvent OpenAI guardrails set up to minimize these risks.
> 
> 用户可以绕过为将这些风险降至最低而设置的 OpenAI 护栏。
>
* ### The Role of GPT-4 in Drug Discovery
> White discovered that in the GPT-4 technical report, it seemed to be able to propose molecules that could be a potential new drug.
> 
> White 发现，在 GPT-4 技术报告中，它似乎能够提出可能成为潜在新药的分子。


<br>

## [The Role of GPT-4 in Drug Discovery](https://vial.com/blog/articles/the-role-of-gpt-4-in-drug-discovery/?utm_source=organic)

To start, we’re trying to fill a list of plausible compounds that could lead to new drugs based on research papers. This is one small step in drug discovery. There are many others! Let’s start with an example of proposing a new drug for psoriasis by targeting a known protein TYK2.

To begin the process, I made tools for GPT-4 to use, instructing it to rely on these tools when working with molecules directly. First, GPT-4 uses one of these tools to conduct literature searches on the target protein, TYK2. It then parses the literature review, which is itself constructed from GPT-3.5-turbo, to identify drugs that have been studied in relation to TYK2. At times, GPT-4 may not know which drugs are small molecules and which are antibodies, so it uses another tool to differentiate between the two.

Once GPT-4 has identified a list of potential drugs, it determines which of them are patented using yet another tool. GPT-4 can then propose modifications to these compounds in an effort to create novel compounds that may be effective in treating psoriasis. However, it is important to note that these modifications are simplistic and do not reflect the true complexity of drug discovery. In many cases, a real medicinal chemist would have to conduct much more extensive modifications to develop a viable drug candidate.

After proposing modifications to the identified compounds, GPT-4 checks to see if the modified compounds are novel. Novel compounds are those that are not present in the SureChEMBL database, which is an approximation of a real patent search. If GPT-4 determines that a compound is novel, it may propose it for further study. However, it is important to note that just because a compound is novel does not mean it will be effective in treating psoriasis. Many other factors must be considered, such as toxicity and side effects.

Finally, GPT-4 determines which of the proposed compounds are not purchasable and must be synthesized. GPT-4 may then propose an email for synthesis to be sent to a lab. This is where the proposed compounds begin the path toward becoming a viable drug candidate.


首先，我们试图根据研究论文填写一份可能导致新药的合理化合物清单。 这是药物发现的一小步。 还有很多！ 让我们从一个针对已知蛋白质 TYK2 提出治疗牛皮癣新药的例子开始。

为了开始这个过程，我制作了供 GPT-4 使用的工具，指示它在直接处理分子时依赖这些工具。 首先，GPT-4 使用其中一种工具对目标蛋白 TYK2 进行文献检索。 然后，它解析本身由 GPT-3.5-turbo 构建的文献综述，以识别已研究的与 TYK2 相关的药物。 有时，GPT-4 可能不知道哪些药物是小分子，哪些是抗体，因此它使用另一种工具来区分两者。

一旦 GPT-4 确定了一份潜在药物清单，它就会使用另一种工具确定其中哪些药物获得了专利。 然后 GPT-4 可以提出对这些化合物的修改，以努力创造可能有效治疗牛皮癣的新化合物。 然而，重要的是要注意这些修改是简单化的，并不能反映药物发现的真正复杂性。 在许多情况下，真正的药物化学家必须进行更广泛的修改才能开发出可行的候选药物。

在对已识别的化合物提出修改建议后，GPT-4 检查修改后的化合物是否新颖。 新化合物是那些不存在于 SureChEMBL 数据库中的化合物，它是真实专利搜索的近似值。 如果 GPT-4 确定一种化合物是新颖的，它可能会建议它进行进一步研究。 然而，重要的是要注意，仅仅因为一种化合物是新的并不意味着它能有效治疗牛皮癣。 必须考虑许多其他因素，例如毒性和副作用。

最后，GPT-4 确定哪些建议的化合物不可购买且必须合成。 GPT-4 然后可能会建议将合成电子邮件发送到实验室。 这是拟议的化合物开始成为可行的候选药物的道路。

### The Impact of GPT-4 on Drug Discovery

While GPT-4’s ability to propose new compounds is impressive, it is important to note that this is just one small step in the complex process of drug discovery. The compounds that GPT-4 proposes must be created and tested to determine if they are effective in treating disease. This requires extensive testing and experimentation through clinical trials, which cannot be fully automated. However, with the help of contract research organizations (CROs) like Vial, the clinical trial process can be expedited through technology that streamlines operations.

So what will the impact be on drug discovery? Unknown. GPT-4’s ability to propose new compounds does open the door to automating more parts of the drug discovery process which could lead to faster and more efficient drug discovery, as well as the discovery of new drugs that may not have been identified through traditional methods. The example above certainly hints at this, but ultimately shows that GPT-4 will not dramatically change drug discovery just yet. It is important to remember that GPT-4 has potential but is not a substitute for the expertise of medicinal chemists and other experts in the field.

虽然 GPT-4 提出新化合物的能力令人印象深刻，但需要注意的是，这只是复杂的药物发现过程中的一小步。 必须创建和测试 GPT-4 提出的化合物，以确定它们是否能有效治疗疾病。 这需要通过临床试验进行广泛的测试和实验，而这不能完全自动化。 然而，在 Vial 等合同研究组织 (CRO) 的帮助下，可以通过简化操作的技术加快临床试验过程。

那么这会对药物发现产生什么影响呢？ 未知。 GPT-4 提出新化合物的能力确实为自动化药物发现过程的更多部分打开了大门，这可能会导致更快、更有效的药物发现，以及可能无法通过传统方法识别的新药的发现。 上面的例子当然暗示了这一点，但最终表明 GPT-4 目前还不会显着改变药物发现。 重要的是要记住，GPT-4 具有潜力，但不能替代药物化学家和该领域其他专家的专业知识。