# D2L 英文教材：GPU 学习环境与本地网站部署

本次在 `/root/workspace/d2l-en` 部署上游 `d2l-ai/d2l-en` 的教材源码，同时提供 JupyterLab 与静态教材网站。上游提交固定为 `23d7a5aecceee57d1292c56e90cce307f183bb0a`，教材声明版本为 `1.0.3`。记录日期：2026-10-18（Asia/Shanghai）。

## 部署范围

- **学习环境**：独立 Python 3.12.13，现代 PyTorch/CUDA 12.8、torchvision、JupyterLab；默认内核为 `D2L PyTorch (CUDA 12.8)`。
- **教材网站**：由本次检出的 Markdown 源码生成，保留 PyTorch、MXNet、JAX、TensorFlow 代码标签页。构建过程关闭 notebook 执行，因此没有本次训练生成的输出；原稿自带的插图仍保留。
- **可运行笔记本**：从同一份源码生成的 PyTorch `.ipynb` 放在 `notebooks/pytorch/`。用户在 JupyterLab 中运行单元格后才会获得执行输出。
- **部署边界**：学习内核安装 PyTorch。网站展示其他框架代码，不代表已安装或验证 MXNet、TensorFlow、JAX 环境。未执行全书训练，也未构建 PDF。

服务器 GPU 是 NVIDIA RTX PRO 6000 Blackwell Server Edition。本次采用支持此 GPU 的现代 PyTorch，未沿用上游安装指南中的 PyTorch 2.0。`setup.py` 的 Python 和基础库依赖范围随部署更新；精确安装版本记录在 `constraints.txt`。这属于本次部署选择，不表示上游已认证这些新版依赖，也不代表全书所有章节均兼容。

## 访问与日常使用

在服务器上执行：

```bash
cd /root/workspace/d2l-en
.venv/bin/python deployment/server.py status
.venv/bin/python deployment/server.py url
```

第二条服务管理命令输出带 token 的 JupyterLab 地址。token 是执行服务器代码的访问凭据，不应提交仓库或公开分享。网站地址为 `http://127.0.0.1:8080/`，JupyterLab 使用 `http://127.0.0.1:8888/`。

在自己的电脑上通过 SSH 转发访问，将 `USER@SERVER` 替换为实际 SSH 登录目标；如服务器 SSH 端口不同，加 `-p` 参数：

```bash
ssh -N -L 8888:127.0.0.1:8888 -L 8080:127.0.0.1:8080 USER@SERVER
```

保持该 SSH 连接运行，在本机浏览器访问上述地址。JupyterLab 请使用 `server.py url` 输出的完整 token 链接。两个服务默认仅监听服务器 IPv4 回环地址；这些 localhost 链接不是公网发布地址，本次未配置域名、TLS 或公网反向代理。

服务管理：

```bash
.venv/bin/python deployment/server.py start
.venv/bin/python deployment/server.py status
.venv/bin/python deployment/server.py stop
```

管理脚本按 PID、进程启动时间与命令识别本项目进程。当前环境 PID 1 是 `sshd`，使用独立后台会话运行，未配置机器重启后的自动启动。重新启动机器后需再次运行 `start`。

终端中使用学习环境：

```bash
source deployment/activate.sh
python -c 'import torch; print(torch.cuda.get_device_name(0))'
```

默认限制 OpenMP/MKL 为 8 个线程，避免单个 notebook 占用全部逻辑 CPU。CUDA 会与服务器现有任务共享，是否有足够显存取决于运行当时的其他任务。大型章节的数据集在首次运行时按原教材逻辑下载，未预先下载全书数据。

## 从源码重新安装

需要 Linux x86_64、可用的 `uv`、NVIDIA GPU 驱动和访问 Python/GitHub/PyTorch 软件源的网络。CUDA 用户态依赖随 wheel 安装，无需为此覆盖系统 CUDA toolkit。

```bash
cd /root/workspace/d2l-en
bash deployment/install.sh
```

`install.sh` 创建隔离 `.venv/`，先从 PyTorch 官方 CUDA 12.8 wheel 源安装 GPU 框架，再按 `constraints.txt` 安装本地可编辑 `d2l` 与 JupyterLab，最后注册项目环境内的内核。约束文件是精确版本快照，不包含所有制品的哈希。它针对本次 Linux x86_64/Python 3.12 部署，不应视为通用跨平台锁文件。

教材构建的依赖、命令、输出及已知限制见 [BOOK.zh-CN.md](BOOK.zh-CN.md)。网站构建完成后生成可编辑学习副本并启动服务：

```bash
.venv/bin/python deployment/prepare_notebooks.py
.venv/bin/python deployment/server.py start
```

`prepare_notebooks.py` 仅复制缺失文件，为新笔记本设置项目内核，保留已有练习文件；重新构建网站不会覆盖练习。构建环境 `.venv-book/` 独立于 `.venv/`，防止旧版文档工具依赖约束影响现代 GPU 学习环境。

## 必要验收与结果边界

验收入口：

```bash
.venv/bin/python deployment/validate.py
```

该脚本验证本地 `d2l.torch` 导入和合成数据函数，使用真实 GPU 执行 100 步小型线性回归与 torchvision CUDA NMS，并通过项目 Jupyter 内核执行 `ndarray.ipynb`、`autograd.ipynb` 两篇教材笔记本。输入笔记本保留原样，执行副本保存在忽略目录 `.deployment-validation/`。结构化结果写入 [validation.json](validation.json)。

GPU 验收使用 1,024 条合成样本、固定随机种子 42、2 个输入特征、100 步 SGD。断言最终均方误差小于 `1e-6` 且相对初值下降至少 10,000 倍。此结果只验证基础 GPU 运算、梯度、优化器及 CUDA 扩展可用，不是教材学习效果评估、GPU 性能基准或长期稳定性测试。

### 本次实测结果

| 项目 | 结果 |
| --- | --- |
| Python / PyTorch / torchvision | `3.12.13` / `2.11.0+cu128` / `0.26.0+cu128` |
| GPU 架构 | compute capability `12.0`；PyTorch 构建包含 `sm_120` |
| GPU 合成回归 MSE | `36.7904663` → `1.2763e-12`，100 步通过 |
| torchvision CUDA NMS | 通过 |
| 数据操作 notebook | 24 个代码单元执行通过 |
| 自动微分 notebook | 13 个代码单元执行通过 |
| Python 依赖一致性 | `uv pip check` 检查 128 个包，通过 |
| Jupyter 未认证 API | 返回 HTTP 403 |
| Jupyter 已认证 API、内核及 notebook 文件 | 返回 HTTP 200，包含 `d2l-pytorch` 内核 |
| 教材网站 HTTP 检查 | 首页、章节、CSS、图片和 notebook ZIP 均返回 200 |
| 网站 / PyTorch notebook 数量 | 194 个 HTML 文件 / 183 个有效 notebook |

验收时间和原始浮点结果见 `validation.json`。本次没有用这两个基础 notebook 推断全书运行成功；其余章节可能需要额外库、数据或针对新版 API 的修订。

## 文件与版本控制

| 路径 | 用途 | 是否提交 |
| --- | --- | --- |
| `deployment/` | 安装、构建、服务管理、验收脚本及文档 | 是，日志除外 |
| `setup.py` | 现代学习环境依赖范围 | 是 |
| `.venv/`、`.venv-book/` | 独立学习/构建环境 | 否 |
| `_build/` | 生成的网站、构建中间文件及日志 | 否 |
| `notebooks/` | 本地生成并供用户练习的 notebooks | 否 |
| `.d2l-runtime/` | token、PID、服务日志 | 否 |
| `.deployment-validation/` | 实际执行 notebook 副本与验收原始材料 | 否 |

运行产生的 notebook 修改不会自动备份到 Git。重要练习可以另存至自己管理的目录，按需纳入版本控制；不要把 token 或内核连接凭据提交。

上游与参考资料：[d2l-en 源码](https://github.com/d2l-ai/d2l-en/tree/23d7a5aecceee57d1292c56e90cce307f183bb0a)、[上游安装说明](https://d2l.ai/chapter_installation/index.html)、[PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)、[d2l-book 构建工具](https://github.com/d2l-ai/d2l-book)。保留上游教材与示例代码的许可文件。

本次部署源码推送目标为 [haoyueliuzhao/d2l-en](https://github.com/haoyueliuzhao/d2l-en)。`origin` 指向该仓库，`upstream` 保留官方 `d2l-ai/d2l-en`。生成的网站、练习文件、运行环境和凭据留在本机，远端保存可重建部署的脚本、依赖约束和文档。
