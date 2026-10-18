# 第 14 章：计算机视觉重点解析

[学习导览](README.md) · [本章原文](../../chapter_computer-vision/index.md) · [术语对照表](../GLOSSARY.zh-CN.md)

本章将分类网络扩展到迁移学习、检测、分割和风格迁移。先修是 CNN、多尺度特征、损失函数、训练/评估模式和基本图像坐标。学习目标是说清每项任务的预测对象、标注形式、损失与评价指标；不要因为都使用 CNN 就沿用同一套输出和指标。

## 核心关系与公式

边界框（bounding box）可用角点 $(x_1,y_1,x_2,y_2)$，也可用中心与宽高 $(c_x,c_y,w,h)$ 表示，其中 $c_x=(x_1+x_2)/2$、$w=x_2-x_1$，纵向同理。代码约定的坐标顺序和归一化尺度必须一致。

两框的交并比（intersection over union，IoU）为

$$\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}.$$

面积应非负，并集为零等退化情况需单独处理。IoU 衡量重叠程度，不是类别预测的置信度。

转置卷积（transposed convolution）单个空间轴的常见输出尺寸为

$$H_{out}=(H_{in}-1)S-2P+D(K-1)+O+1,$$

$S$ 为步幅、$P$ 为填充、$D$ 为膨胀率、$K$ 为核大小、$O$ 为 `output_padding`。恢复形状时可能需要选择合法的 $O$；即使输出形状与原输入相同，也不等于恢复了原输入数值。转置的是卷积的线性映射，不是求逆。

## 按教材顺序解析

### 14.1 图像增广

[原文与代码](../../chapter_computer-vision/image-augmentation.md)。图像增广（image augmentation）通过翻转、裁剪、颜色扰动等扩展训练时的输入变化。变换需保留任务标签语义，例如随意翻转文字未必合理。随机训练增强与通常确定性的验证预处理要区分；检测与分割还需同步变换坐标或标签图。

### 14.2 微调

[原文与代码](../../chapter_computer-vision/fine-tuning.md)。迁移学习（transfer learning）复用源任务的表示，微调（fine-tuning）在目标数据上继续调整模型。教材用热狗识别展示替换输出头，并为预训练层和新头设置不同学习率。预处理应匹配预训练模型，目标数据和源数据差异会影响迁移效果，不能把“预训练”视为无条件收益。

### 14.3 目标检测与边界框

[原文与代码](../../chapter_computer-vision/bounding-box.md)。目标检测（object detection）同时预测“是什么”和“在哪里”，一张图可能包含多个对象。先掌握两种框表示的互转与绘图，再进入模型。图像张量的高宽顺序与坐标的横纵顺序不同，是常见错误来源。

### 14.4 锚框

[原文与代码](../../chapter_computer-vision/anchor.md)。锚框（anchor box）为不同位置、尺度和宽高比提供候选参照。训练时为锚框分配真实框，产生类别与偏移标签；背景锚框通常不承担目标框回归。预测时解码偏移，再通过非极大值抑制（non-maximum suppression，NMS）删去高度重叠的低分候选。分配阈值与 NMS 阈值服务于不同阶段，不能混用。

### 14.5 多尺度目标检测

[原文与代码](../../chapter_computer-vision/multiscale-object-detection.md)。不同分辨率特征图提供不同密度的锚框中心，深层特征通常对应更大的感受野。小物体的空间细节与大物体的语义上下文需要不同尺度信息。多尺度不是只在同一位置机械增加框数量，还要看预测特征与输入区域的对应关系。

### 14.6 目标检测数据集

[原文与代码](../../chapter_computer-vision/object-detection-dataset.md)。香蕉检测数据集用于展示图像与“类别加坐标”标注如何一起加载。检查每个标签列的意义、框坐标是否归一化，并先绘制少量样本核对位置。单对象教学数据的训练流程不能直接代表复杂场景的完整评估难度。

### 14.7 单发多框检测 SSD

[原文与代码](../../chapter_computer-vision/ssd.md)。单发多框检测（single shot multibox detection，SSD）在多个特征尺度上同时预测每个锚框的类别与偏移。若每位置有 $a$ 个锚框、前景类别数为 $c$，类别头通常输出 $a(c+1)$ 个通道，回归头输出 $4a$ 个通道；多出的类别是背景。教材 `TinySSD` 使用分类损失与掩码后的 L1 回归损失，是教学实现；分类准确率和框误差不等同于完整检测指标 mAP。

### 14.8 区域卷积网络 R-CNN 系列

[原文与演示](../../chapter_computer-vision/rcnn.md)。R-CNN 对候选区域分别提取特征，Fast R-CNN 共享整图特征并引入兴趣区域汇聚（RoI pooling），Faster R-CNN 进一步用区域提议网络（region proposal network）生成候选。Mask R-CNN 增加实例掩码分支。关注各代方案减少的重复计算与增加的预测目标，本节没有提供这些模型的完整训练复现实验。

### 14.9 语义分割与数据集

[原文与代码](../../chapter_computer-vision/semantic-segmentation-and-dataset.md)。语义分割（semantic segmentation）为每个像素分配类别，实例分割（instance segmentation）还区分同类对象个体。VOC 标签颜色是类别编码，不能当作连续 RGB 回归目标。教材对图像与标签进行相同随机裁剪以保留对齐；若自行缩放类别标签，应采用不会混出新类别的离散标签处理，例如最近邻插值。

### 14.10 转置卷积

[原文与代码](../../chapter_computer-vision/transposed-conv.md)。可将普通卷积展成稀疏矩阵乘法，转置卷积使用该矩阵的转置把输入贡献分发到输出位置。重叠位置的贡献相加；核、步幅和填充决定输出形状。它可以学习上采样，但并非所有配置都会扩大空间尺寸，更不能称为“卷积的精确逆”。

### 14.11 全卷积网络 FCN

[原文与代码](../../chapter_computer-vision/fcn.md)。全卷积网络（fully convolutional network，FCN）先提取空间特征，以 $1\times1$ 卷积把通道变为类别数，再上采样到目标分辨率。教材用双线性插值形式初始化转置卷积，之后允许学习其权重。输出一般为 `(批量, 类别, 高, 宽)`，标签为 `(批量, 高, 宽)`；按类别维取最大值才得到像素类别图。

### 14.12 神经风格迁移

[原文与代码](../../chapter_computer-vision/neural-style.md)。神经风格迁移（neural style transfer）把合成图像本身作为优化对象，使用预训练特征提取网络构建内容与风格约束。内容损失匹配特征表示，Gram 矩阵描述通道响应的相关性，全变差（total variation）项鼓励局部平滑。风格损失不是逐像素复制风格图，也没有唯一客观最优的视觉效果；各损失权重改变的是目标定义。

### 14.13 Kaggle CIFAR-10 图像分类

[原文与代码](../../chapter_computer-vision/kaggle-cifar10.md)。这节把目录整理、训练/验证切分、增强、训练和预测提交串成完整工作流。演示模式使用缩小数据以展示流程，其分数不能代表完整竞赛设置。提交文件要保留样本 ID 与类别的正确对应，验证集用于选择配置，未知标签测试集不能用于直接调参。

### 14.14 Kaggle 犬种识别

[原文与代码](../../chapter_computer-vision/kaggle-dog.md)。较大且尺寸不一的图像需要重新考虑裁剪和增强；教材使用预训练网络与新的分类头完成迁移。当前 PyTorch 示例冻结 `features` 参数，并将原模型 1000 维输出送入新头，不是自动截取倒数第二层。冻结梯度与切换评估模式不同，BN 的运行状态、dropout 行为仍需单独检查。

## 代码阅读路线

检测主线沿 [框表示转换](../../chapter_computer-vision/bounding-box.md) → [锚框的 `multibox_prior`、`assign_anchor_to_bbox`、`multibox_target`、`nms`](../../chapter_computer-vision/anchor.md) → [`TinySSD` 与 `calc_loss`](../../chapter_computer-vision/ssd.md)。特别核对拼接后锚框顺序与类别/偏移预测顺序一致。分割主线沿 [VOC 的 `voc_label_indices`、`voc_rand_crop`](../../chapter_computer-vision/semantic-segmentation-and-dataset.md) → [转置卷积的 `trans_conv`](../../chapter_computer-vision/transposed-conv.md) → [FCN 的 `bilinear_kernel` 与 `loss`](../../chapter_computer-vision/fcn.md)。风格迁移则看 [`gram`、`compute_loss`、`SynthesizedImage`](../../chapter_computer-vision/neural-style.md) 的优化变量。

## 易错点与自测

- **NMS 能修复坐标偏移预测错误吗？** 不能，它主要筛掉重复候选；定位本身需由模型与解码规则保证。
- **检测准确率很高就说明框预测好了吗？** 不一定，背景锚框比例很高时分类正确率也可能很高，应同时检查前景、定位与适当检测指标。
- **分割标签做双线性插值有什么问题？** 类别索引不具备连续数值意义，会产生不存在的标签或混淆边界。
- **转置卷积得到原尺寸是否等于恢复原图？** 不等于；形状匹配与信息可逆是不同问题。
- **风格迁移时更新的参数是什么？** 本节主要更新合成图像像素；预训练网络提供损失所用的特征表示。

## 小实验建议（本次未执行）

1. 对固定检测预测，改变 NMS 的 IoU 阈值，记录保留框数、重复框与漏掉相邻对象的案例。固定模型输出以隔离后处理影响，不将框变少自动当作效果更好。
2. 对一张图像和分割标签做同步裁剪，叠加显示以检查像素对应关系，再手算一个小尺寸转置卷积的输出。此实验验证数据与算子理解，不产生语义分割准确率结论。
