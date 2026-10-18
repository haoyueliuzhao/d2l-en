# 第 15 章 自然语言处理：预训练重点解析

[返回解析目录](README.md) · [原章目录](../../chapter_natural-language-processing-pretraining/index.md) · [术语对照表](../GLOSSARY.zh-CN.md)

## 本章定位与先修

本章回答“没有逐句人工标注时，怎样学到可迁移的文本表示”。先修是嵌入层（embedding layer）、交叉熵（cross-entropy）、小批量训练，以及第 11 章 Transformer 编码器。学习主线从每个词对应固定向量，推进到同一个词随上下文变化的表示；自监督学习（self-supervised learning）的标签由原始文本构造，并不表示训练没有目标。

应先掌握 word2vec 的预测任务，再读数据流水线；BERT 部分则按输入、预训练任务、批数据、损失四步阅读。小语料上的教学训练只用于观察机制，不能据此认定得到通用语言理解能力。

## 核心机制与公式

跳元模型（skip-gram）给中心词 $c$ 预测上下文词 $o$：

$$
p(o\mid c)=\frac{\exp(\mathbf u_o^\top\mathbf v_c)}{\sum_{w\in V}\exp(\mathbf u_w^\top\mathbf v_c)}.
$$

$V$ 是词表，$\mathbf v$ 与 $\mathbf u$ 是两套可训练向量。分母遍历词表，是近似训练要解决的主要成本。负采样（negative sampling）将一次正词对与 $K$ 个噪声词对组成二分类目标：

$$
-\log\sigma(\mathbf u_o^\top\mathbf v_c)
-\sum_{k=1}^{K}\log\sigma(-\mathbf u_{n_k}^\top\mathbf v_c).
$$

它是改变训练目标的有效近似，不能将负样本集合上的 sigmoid 数值当作完整词表上的归一化概率。BERT 则输出形状为“批量 × 序列长度 × 隐藏维度”的上下文表示，联合训练掩码语言建模（masked language modeling, MLM）和下一句预测（next sentence prediction, NSP）：本书实现将二者损失相加，但 MLM 只在被选中的预测位置计分。

## 按教材顺序逐节解析

### 15.1 词嵌入：word2vec

[原文与代码](../../chapter_natural-language-processing-pretraining/word2vec.md)。独热编码（one-hot encoding）不能直接表达词之间的相似程度，词嵌入把词映射到可学习的稠密空间。跳元模型由中心词预测上下文；连续词袋模型（continuous bag of words, CBOW）反过来用上下文的汇总表示预测中心词。窗口内词的条件独立是目标分解所用的建模假设，不是语言的真实规律。

### 15.2 近似训练

[原文与代码](../../chapter_natural-language-processing-pretraining/approx-training.md)。负采样通过区分真实词对和噪声词对，避免每次更新计算全词表 softmax；分层 softmax（hierarchical softmax）把词预测转成树上路径的一串二分类。后者计算量与路径长度有关，树的结构会影响成本；两种方法不可简单理解为“省略几个分母项”。

### 15.3 词嵌入预训练数据集

[原文与代码](../../chapter_natural-language-processing-pretraining/word-embedding-dataset.md)。PTB 文本先建词表，再对高频词做下采样（subsampling），以减少大量功能词对训练的支配。窗口大小随机化产生不同长度上下文，噪声分布使用词频的 $0.75$ 次幂；最后统一填充（padding），用掩码（mask）区分有效词对和补齐位置。数据预处理在这里直接决定训练目标的统计分布。

### 15.4 预训练 word2vec

[原文与代码](../../chapter_natural-language-processing-pretraining/word2vec-pretraining.md)。中心词查表得到一条向量，上下文及负样本查表得到多条向量，批量矩阵乘法一次产生全部二分类分数。两套嵌入都要更新；掩码后的损失还应按有效项数归一化，避免上下文更短的样本仅因 padding 而显得损失更小。余弦近邻是表示的定性检查，不是通用语义能力评分。

### 15.5 全局向量：GloVe

[原文与代码](../../chapter_natural-language-processing-pretraining/glove.md)。GloVe 从全局共现计数出发，让词向量内积加偏置拟合对数共现次数，并对不同频率赋权。对数使很大的计数不至于完全主导平方误差；权重函数也让极低频统计的影响受到控制。理解共现概率比值如何区分语义，比背诵损失形式更重要。

### 15.6 子词嵌入

[原文与代码](../../chapter_natural-language-processing-pretraining/subword-embedding.md)。fastText 用字符 n-gram 的向量组成词表示，使有相似词形的词共享参数；字节对编码（byte pair encoding, BPE）则反复合并高频符号对，学习分词单元。前者重点在表示组合，后者重点在词元（token）切分，不能视作同一个算法。子词能缓解未登录词问题，但不保证任何生僻词的语义都准确。

### 15.7 词相似度与类比

[原文与代码](../../chapter_natural-language-processing-pretraining/similarity-analogy.md)。通过预训练向量和余弦相似度（cosine similarity）检索近邻，再用 $\mathbf b-\mathbf a+\mathbf c$ 做类比检索。检索要排除查询词本身，且区分语义相似与语境相关；向量关系也可能包含语料偏见。几个漂亮的类比例子只能说明这些例子的现象。

### 15.8 BERT

[原文与代码](../../chapter_natural-language-processing-pretraining/bert.md)。输入是词元、片段、位置三种嵌入之和，通过双向 Transformer 编码器获得上下文表示。MLM 根据左右上下文恢复指定词元，NSP 根据 `<cls>` 表示判断句对关系；“双向”不等于自回归生成时可以访问未来标签。`MaskLM` 从指定位置抽取隐藏状态，`NextSentencePred` 使用整段聚合位置，二者读取的信息粒度不同。

### 15.9 BERT 预训练数据集

[原文与代码](../../chapter_natural-language-processing-pretraining/bert-dataset.md)。WikiText 的段落用于构造相邻或随机句对，随后抽取约 15% 的词元作为 MLM 目标；选中位置再按 80% 掩码、10% 随机词、10% 原词处理。标签保留替换前的真实词元。词元 ID、片段 ID、有效长度、预测位置、MLM 权重与标签、NSP 标签必须一一对应。

### 15.10 预训练 BERT

[原文与代码](../../chapter_natural-language-processing-pretraining/bert-pretraining.md)。训练循环合并 MLM 与 NSP 损失，使用较小网络演示端到端预训练，并展示同一词在不同句子中的表示。观察上下文向量变化可以验证计算路径，但向量不同本身不能证明模型正确理解多义词。这里的小规模预训练与下一章加载已有模型微调，是两条不同的实验路线。

## 代码阅读路线

先追踪 `load_data_ptb → batchify → skip_gram → SigmoidBCELoss`，写下各个张量的形状和有效长度。再追踪 `get_tokens_and_segments → _get_mlm_data_from_tokens → _pad_bert_inputs → BERTModel → _get_batch_loss_bert`；重点检查哪些位置进入注意力、哪些位置进入 MLM 损失。上述函数均在对应小节源码中，适合与 PyTorch 标签页逐行对照。

## 易错点与自测

- **为何 word2vec 有两套嵌入？** 答案线索：中心词与上下文词承担不同角色，二分类内积连接两套参数；训练后选哪套表示是后续使用选择。
- **MLM 为什么不能把 padding 当预测目标？** 答案线索：padding 是批处理产生的人工位置；监督它会改变目标且污染损失归一化。
- **BERT 是词向量表吗？** 答案线索：输入查表是静态的，输出经过注意力后依赖整段有效上下文。
- **提高负样本数就一定提高效果吗？** 答案线索：计算成本、噪声分布、语料规模都会改变收益，需实验判断。

## 建议实验（未执行）

1. 固定 PTB 子集、随机种子与总更新次数，比较负样本数 2 和 5；记录每步耗时、有效词对数、损失定义以及固定查询词的近邻。训练目标随负样本数变化，不能只比较原始损失大小判断优劣。
2. 用同一 BERT 参数，对包含同一词的两个句子提取该词输出；同时查看输入嵌入和编码后向量的余弦相似度。此实验用于确认上下文依赖，不作为语义理解能力结论。
