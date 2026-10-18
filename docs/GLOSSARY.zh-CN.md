# D2L 重点学术词汇中英对照表

本表整理 310 条深度学习重点术语，按 18 个教材主题分类，覆盖基础数学、神经网络、计算机视觉、自然语言处理、推荐系统、生成对抗网络等。第三列仅补充常见缩写或容易混淆之处。本表可独立用作学习参考，教材正文未因本表而改动。

机器可读词表：[glossary.tsv](glossary.tsv)。在原有 282 条术语基础上补充 28 条评估指标、推荐系统、生成对抗网络与范数术语。同义名称与英文拼写变体分别保留，因此 310 条记录不等于 310 个互不重叠的概念。缩写取决于语境；“—”表示此处无需额外说明。

阅读时可使用“推荐中文（英文术语）”形式，例如“随机梯度下降（stochastic gradient descent）”。

## 学习范式、数据与评估

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| deep learning | 深度学习 | DL |
| machine learning | 机器学习 | ML |
| supervised learning | 监督学习 | 训练时使用标注信息。 |
| unsupervised learning | 无监督学习 | — |
| self-supervised learning | 自监督学习 | 监督信号由数据本身构造。 |
| semi-supervised learning | 半监督学习 | 同时利用有标注和无标注数据。 |
| representation learning | 表示学习 | — |
| transfer learning | 迁移学习 | — |
| multitask learning | 多任务学习 | MTL |
| training set | 训练集 | 用于拟合模型参数。 |
| validation set | 验证集 | 用于模型选择与调参。 |
| test set | 测试集 | 用于独立评估。 |
| generalization | 泛化 | — |
| overfitting | 过拟合 | — |
| underfitting | 欠拟合 | — |
| inductive bias | 归纳偏置 | — |
| empirical risk | 经验风险 | 基于有限样本计算。 |
| expected risk | 期望风险 | 相对于数据分布取期望。 |
| loss function | 损失函数 | 描述预测与目标之间的代价。 |
| objective function | 目标函数 | 优化对象，可包含损失与正则项。 |
| regularization | 正则化 | — |
| cross-validation | 交叉验证 | 常见写法：k-fold CV（k折交叉验证）。 |
| data augmentation | 数据增强 | — |

## 线性代数与张量运算

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| scalar | 标量 | — |
| vector | 向量 | — |
| matrix | 矩阵 | — |
| tensor | 张量 | 本书中通常用来表示多维数组。 |
| transpose | 转置 | — |
| symmetric matrix | 对称矩阵 | — |
| identity matrix | 单位矩阵 | — |
| diagonal matrix | 对角矩阵 | — |
| matrix multiplication | 矩阵乘法 | — |
| Hadamard product | 哈达玛积 | 逐元素乘积；不同于矩阵乘法。 |
| dot product | 点积 | 两个向量的点积结果为标量。 |
| inner product | 内积 | 实向量的标准内积即点积。 |
| outer product | 外积 | 两个向量的外积可形成矩阵；不同于向量叉积（cross product）。 |
| linear combination | 线性组合 | — |
| linear independence | 线性无关 | — |
| linear transformation | 线性变换 | — |
| orthogonality | 正交性 | — |
| orthonormal basis | 标准正交基 | — |
| vector space | 向量空间 | — |
| eigenvalue | 特征值 | — |
| eigenvector | 特征向量 | — |
| eigendecomposition | 特征分解 | — |
| singular value decomposition | 奇异值分解 | SVD；奇异值与特征值需区分。 |
| positive definite | 正定 | PD |
| positive semidefinite | 半正定 | PSD |
| Frobenius norm | 弗罗贝尼乌斯范数 | — |
| Euclidean norm | 欧几里得范数 | 即向量的 L2 范数。 |
| broadcasting | 广播机制 | 按形状规则扩展数组间的运算。 |
| norm | 范数 | 向量大小的度量；不是向量的维数。 |

## 微积分与自动微分

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| partial derivative | 偏导数 | — |
| gradient | 梯度 | — |
| Jacobian | 雅可比矩阵 | 此处指一阶偏导数组成的矩阵。 |
| Hessian | 海森矩阵 | 此处指二阶偏导数组成的矩阵。 |
| chain rule | 链式法则 | — |
| automatic differentiation | 自动微分 | AD；不同于数值差分。 |
| computational graph | 计算图 | — |
| backpropagation | 反向传播 | BP；利用链式法则反向传播梯度。 |
| Taylor expansion | 泰勒展开 | — |

## 概率、统计与信息论

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| probability distribution | 概率分布 | — |
| random variable | 随机变量 | — |
| conditional probability | 条件概率 | — |
| joint probability | 联合概率 | — |
| marginal probability | 边缘概率 | — |
| Bayes' theorem | 贝叶斯定理 | 与 Bayes' rule 指同一概率法则。 |
| Bayes' rule | 贝叶斯法则 | Bayes' theorem 的另一常见称呼。 |
| independent and identically distributed | 独立同分布 | i.i.d. / IID |
| conditional independence | 条件独立 | — |
| expectation | 期望 | — |
| variance | 方差 | — |
| covariance | 协方差 | — |
| standard deviation | 标准差 | — |
| Gaussian distribution | 高斯分布 | 与 normal distribution 同义。 |
| normal distribution | 正态分布 | 与 Gaussian distribution 同义。 |
| Bernoulli distribution | 伯努利分布 | — |
| categorical distribution | 类别分布 | 描述一次试验的类别结果。 |
| multinomial distribution | 多项分布 | 描述多次试验中各类别的计数。 |
| uniform distribution | 均匀分布 | — |
| probability density function | 概率密度函数 | PDF；密度值不等同于点概率。 |
| probability mass function | 概率质量函数 | PMF |
| cumulative distribution function | 累积分布函数 | CDF |
| likelihood | 似然 | 固定观测数据后，作为参数的函数。 |
| maximum likelihood estimation | 极大似然估计 | MLE |
| maximum a posteriori estimation | 最大后验估计 | MAP；使用先验信息。 |
| prior distribution | 先验分布 | 观测当前数据之前的分布。 |
| posterior distribution | 后验分布 | 根据观测数据更新后的分布。 |
| central limit theorem | 中心极限定理 | CLT |
| law of large numbers | 大数定律 | LLN |
| confidence interval | 置信区间 | CI；与贝叶斯可信区间区分。 |
| hypothesis testing | 假设检验 | — |
| entropy | 熵 | — |
| cross-entropy | 交叉熵 | — |
| Kullback-Leibler divergence | 库尔贝克–莱布勒散度 | KL；通常不对称，不是距离度量。 |
| KL divergence | KL散度 | Kullback-Leibler divergence 的缩写形式。 |
| mutual information | 互信息 | MI |

## 回归、分类与训练迭代

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| linear regression | 线性回归 | — |
| logistic regression | 逻辑回归 | 通常用于分类，名称中仍使用“回归”。 |
| softmax regression | softmax回归 | 常用于多分类。 |
| multiclass classification | 多分类 | — |
| binary classification | 二分类 | — |
| one-hot encoding | 独热编码 | — |
| mean squared error | 均方误差 | MSE |
| negative log-likelihood | 负对数似然 | NLL |
| minibatch | 小批量 | 一次参数更新使用的一小组样本。 |
| batch size | 批量大小 | 一个批次包含的样本数。 |
| epoch | 训练轮次 | 通常指完整遍历训练集一次；不同于一次参数更新。 |

## 评估指标与统计偏差

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| accuracy | 准确率 | 正确预测数占全部样本的比例；不同于 precision。 |
| precision | 精确率 | 亦称“查准率”；预测为正的样本中实际为正的比例。 |
| recall | 召回率 | 亦称“查全率”；实际为正的样本中被识别为正的比例。 |
| F1 score | F1分数 | 精确率与召回率的调和平均；需注意多分类平均方式。 |
| confusion matrix | 混淆矩阵 | 按真实类别与预测类别汇总计数。 |
| root mean squared error | 均方根误差 | RMSE；均方误差的平方根。 |
| receiver operating characteristic | 受试者工作特征 | ROC；常指真阳性率对假阳性率的曲线。 |
| area under the curve | 曲线下面积 | AUC；应指明 ROC-AUC 或 PR-AUC 等具体曲线。 |
| bias | 偏差 | 统计语境译“偏差”；模型加性参数中常译“偏置”。 |

## 神经网络与数值稳定性

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| multilayer perceptron | 多层感知机 | MLP |
| feed-forward neural network | 前馈神经网络 | FFNN / FNN |
| hidden layer | 隐藏层 | — |
| hidden unit | 隐藏单元 | — |
| activation function | 激活函数 | — |
| rectified linear unit | 修正线性单元 | ReLU |
| sigmoid function | sigmoid函数 | — |
| hyperbolic tangent | 双曲正切 | tanh |
| universal approximation theorem | 通用近似定理 | — |
| dropout | 暂退法 | 亦称“随机失活”或“丢弃法”。 |
| weight initialization | 权重初始化 | — |
| Xavier initialization | Xavier初始化 | 亦称 Glorot 初始化。 |
| vanishing gradient | 梯度消失 | 梯度随传播变得过小。 |
| exploding gradient | 梯度爆炸 | 梯度随传播变得过大。 |
| gradient clipping | 梯度裁剪 | 用于限制梯度大小。 |
| numerical stability | 数值稳定性 | — |

## 梯度优化与正则化

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| learning rate | 学习率 | 常记作 η 或 α。 |
| stochastic gradient descent | 随机梯度下降 | SGD |
| minibatch stochastic gradient descent | 小批量随机梯度下降 | 实践中也常简称 SGD。 |
| gradient descent | 梯度下降 | GD |
| weight decay | 权重衰减 | 具体实现与 L2 正则化是否等价取决于优化器。 |
| convex optimization | 凸优化 | — |
| convex function | 凸函数 | — |
| local minimum | 局部极小值 | 仅相对于局部邻域而言。 |
| global minimum | 全局最小值 | 相对于整个定义域而言。 |
| saddle point | 鞍点 | — |
| momentum | 动量 | — |
| Nesterov momentum | Nesterov动量 | 常与 NAG（Nesterov加速梯度）相关联。 |
| adaptive learning rate | 自适应学习率 | — |
| learning rate scheduler | 学习率调度器 | — |
| learning rate decay | 学习率衰减 | — |
| convergence rate | 收敛速率 | — |
| preconditioning | 预条件化 | — |

## 卷积网络与计算机视觉

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| convolutional neural network | 卷积神经网络 | CNN |
| convolutional layer | 卷积层 | — |
| convolution kernel | 卷积核 | — |
| cross-correlation | 互相关 | 与数学定义中的卷积区分。 |
| receptive field | 感受野 | 某个输出所依赖的输入区域。 |
| translation invariance | 平移不变性 | 输入平移后，输出保持不变。 |
| translation equivariance | 平移等变性 | 输入平移时，输出发生相应平移。 |
| parameter sharing | 参数共享 | — |
| padding | 填充 | — |
| stride | 步幅 | — |
| feature map | 特征图 | — |
| pooling | 汇聚 | 亦称“池化”。 |
| max pooling | 最大汇聚 | 亦称“最大池化”。 |
| average pooling | 平均汇聚 | 亦称“平均池化”。 |
| global average pooling | 全局平均汇聚 | GAP；亦称“全局平均池化”。 |
| batch normalization | 批量规范化 | BN；亦称“批量归一化”。 |
| layer normalization | 层规范化 | LN；亦称“层归一化”。 |
| residual connection | 残差连接 | — |
| residual network | 残差网络 | ResNet |
| densely connected network | 稠密连接网络 | DenseNet |
| inception block | Inception块 | — |
| fully connected layer | 全连接层 | FC层 |
| network in network | 网络中的网络 | NiN |
| object detection | 目标检测 | — |
| semantic segmentation | 语义分割 | — |
| bounding box | 边界框 | 常写作 bbox。 |
| anchor box | 锚框 | — |
| intersection over union | 交并比 | IoU |
| non-maximum suppression | 非极大值抑制 | NMS |
| region of interest | 感兴趣区域 | RoI / ROI |
| region proposal | 区域提议 | 候选区域；不同于已确定的检测结果。 |
| multiscale detection | 多尺度检测 | — |
| transposed convolution | 转置卷积 | 不是一般意义上的卷积逆运算。 |
| fully convolutional network | 全卷积网络 | FCN |
| neural style transfer | 神经风格迁移 | NST |
| fine-tuning | 微调 | 在预训练模型基础上继续训练。 |

## 生成对抗网络

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| generative adversarial network | 生成对抗网络 | GAN |
| generator | 生成器 | GAN 中生成样本的网络。 |
| discriminator | 判别器 | GAN 中区分真实样本与生成样本的网络。 |
| adversarial training | 对抗训练 | GAN 中指双方对抗优化；与对抗样本鲁棒性训练语境区分。 |
| mode collapse | 模式崩溃 | 亦称“模式坍塌”；生成样本缺乏多样性。 |
| deep convolutional generative adversarial network | 深度卷积生成对抗网络 | DCGAN |
| minimax game | 极小极大博弈 | GAN 常用的博弈表述。 |

## 循环网络与序列建模

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| recurrent neural network | 循环神经网络 | RNN |
| hidden state | 隐藏状态 | — |
| sequence model | 序列模型 | — |
| language model | 语言模型 | LM |
| autoregressive model | 自回归模型 | AR |
| Markov assumption | 马尔可夫假设 | — |
| latent variable | 潜变量 | — |
| backpropagation through time | 通过时间反向传播 | BPTT |
| truncated backpropagation through time | 截断的通过时间反向传播 | TBPTT |
| long short-term memory | 长短期记忆网络 | LSTM |
| gated recurrent unit | 门控循环单元 | GRU |
| reset gate | 重置门 | — |
| update gate | 更新门 | — |
| forget gate | 遗忘门 | — |
| input gate | 输入门 | — |
| output gate | 输出门 | — |
| memory cell | 记忆元 | — |
| bidirectional recurrent neural network | 双向循环神经网络 | BiRNN |
| sequence-to-sequence learning | 序列到序列学习 | seq2seq |
| encoder-decoder | 编码器–解码器 | 连字符写法；与下一条为同一术语。 |
| encoder–decoder | 编码器–解码器 | 短横线写法；保留原词表中的拼写变体。 |
| encoder | 编码器 | — |
| decoder | 解码器 | — |
| teacher forcing | 教师强制 | — |
| beam search | 束搜索 | — |
| perplexity | 困惑度 | PPL |

## 注意力与 Transformer

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| attention mechanism | 注意力机制 | — |
| attention pooling | 注意力汇聚 | — |
| attention score | 注意力评分 | 归一化之前的评分。 |
| attention weight | 注意力权重 | 常由评分经 softmax 归一化得到。 |
| additive attention | 加性注意力 | — |
| scaled dot-product attention | 缩放点积注意力 | 常用缩放因子为 1/√d。 |
| multi-head attention | 多头注意力 | MHA |
| self-attention | 自注意力 | 查询、键和值来自同一序列表示。 |
| cross-attention | 交叉注意力 | 查询与键/值来自不同序列表示。 |
| positional encoding | 位置编码 | — |
| positionwise feed-forward network | 逐位置前馈网络 | FFN；逐位置应用相同网络。 |
| query vector | 查询向量 | Q |
| key vector | 键向量 | K |
| value vector | 值向量 | V |
| masked softmax | 掩蔽softmax | — |
| vision transformer | 视觉Transformer | ViT |

## 自然语言处理与预训练

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| natural language processing | 自然语言处理 | NLP |
| word embedding | 词嵌入 | — |
| distributed representation | 分布式表示 | 指信息分布于多个表示维度，不是分布式计算。 |
| skip-gram | 跳元模型 | 利用中心词预测上下文词。 |
| continuous bag-of-words | 连续词袋模型 | CBOW；利用上下文词预测中心词。 |
| negative sampling | 负采样 | — |
| hierarchical softmax | 层次softmax | — |
| subword embedding | 子词嵌入 | — |
| byte pair encoding | 字节对编码 | BPE |
| masked language modeling | 掩蔽语言建模 | MLM |
| next sentence prediction | 下一句预测 | NSP |
| pretraining | 预训练 | — |
| sentiment analysis | 情感分析 | — |
| natural language inference | 自然语言推断 | NLI |
| text classification | 文本分类 | — |
| machine translation | 机器翻译 | MT |

## 推荐系统

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| recommendation system | 推荐系统 | 亦称 recommender system。 |
| collaborative filtering | 协同过滤 | CF |
| matrix factorization | 矩阵分解 | MF；推荐系统中常用于学习用户与物品的因子表示。 |
| latent factor model | 隐因子模型 | LFM |
| implicit feedback | 隐式反馈 | 如点击、浏览或购买行为；未观察到行为不必然表示不喜欢。 |
| explicit feedback | 显式反馈 | 如用户主动给出的评分。 |
| ranking loss | 排序损失 | 面向候选项相对次序的损失。 |
| Bayesian personalized ranking | 贝叶斯个性化排序 | BPR；常用于成对排序。 |
| factorization machine | 因子分解机 | FM |
| click-through rate | 点击率 | CTR |
| autoencoder | 自编码器 | AE；通常通过重构输入学习表示。 |

## 计算性能与并行训练

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| computational complexity | 计算复杂度 | — |
| asynchronous computation | 异步计算 | — |
| parallel computation | 并行计算 | — |
| data parallelism | 数据并行 | DP；不同设备处理不同数据批次。 |
| model parallelism | 模型并行 | 模型的不同部分分配到不同设备。 |
| parameter server | 参数服务器 | PS |
| all-reduce | 全规约 | 亦写作 allreduce；常用于同步聚合梯度。 |
| bandwidth | 带宽 | 单位时间可传输的数据量。 |
| latency | 延迟 | 一次操作或请求的响应延迟。 |
| throughput | 吞吐量 | — |
| mixed precision | 混合精度 | 混合使用不同数值精度。 |
| vectorization | 向量化 | — |
| just-in-time compilation | 即时编译 | JIT |

## 强化学习

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| reinforcement learning | 强化学习 | RL |
| Markov decision process | 马尔可夫决策过程 | MDP |
| value function | 价值函数 | 通常记作 V；衡量状态的预期回报。 |
| action-value function | 动作价值函数 | 通常记作 Q；考虑给定状态和动作。 |
| Bellman equation | 贝尔曼方程 | — |
| discount factor | 折扣因子 | 通常记作 γ。 |
| optimal policy | 最优策略 | — |
| value iteration | 价值迭代 | — |
| policy iteration | 策略迭代 | — |
| Q-learning | Q学习 | 基于动作价值函数的强化学习算法。 |
| temporal difference | 时序差分 | TD |
| exploration-exploitation tradeoff | 探索与利用的权衡 | 在探索未知与利用已有知识之间取舍。 |

## 高斯过程与不确定性

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| Gaussian process | 高斯过程 | GP |
| kernel function | 核函数 | 在GP中用于指定函数值之间的协方差。 |
| covariance function | 协方差函数 | 在GP语境下也称核函数。 |
| radial basis function | 径向基函数 | RBF |
| squared exponential kernel | 平方指数核 | 常称 SE 核；也常称 RBF 核。 |
| posterior predictive distribution | 后验预测分布 | 基于已观测数据对新观测或函数值作出的预测分布。 |
| uncertainty quantification | 不确定性量化 | UQ |
| marginal likelihood | 边缘似然 | 对潜变量或参数积分后得到；积分对象取决于模型。 |

## 超参数优化

| 英文术语 | 推荐中文 | 常见缩写 / 必要辨析 |
| --- | --- | --- |
| hyperparameter | 超参数 | 与训练过程中拟合的模型参数区分。 |
| hyperparameter optimization | 超参数优化 | HPO |
| hyperparameter tuning | 超参数调优 | 强调选择、调整超参数的过程。 |
| search space | 搜索空间 | — |
| random search | 随机搜索 | — |
| grid search | 网格搜索 | — |
| Bayesian optimization | 贝叶斯优化 | BO |
| acquisition function | 采集函数 | 用于选择下一次评估的位置。 |
| surrogate model | 代理模型 | 近似目标函数，通常用于降低评估成本。 |
| early stopping | 早停 | 亦称“提前停止”。 |
| successive halving | 逐次减半 | SH |
| asynchronous successive halving | 异步逐次减半 | ASHA |
| multi-fidelity optimization | 多保真度优化 | — |
| expected improvement | 期望改进 | EI |
| upper confidence bound | 置信上界 | UCB |
