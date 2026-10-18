# D2L 英文书静态网站构建说明

## 交付范围

本部署从仓库当前工作区生成完整多框架阅读网站，并导出 PyTorch Jupyter Notebook。网站使用上游 `d2lbook` 解析章节、数学公式、参考文献和框架选项卡，保留官方首页布局、封面、作者、机构及静态地图。构建不会执行书中的训练代码，也不下载训练数据集。

学习环境与网站构建环境分开：学习环境由仓库部署脚本管理；`.venv-book` 只负责文档构建，不安装本项目的训练依赖或四套深度学习框架。这样可以使用现代 PyTorch 学习环境，同时保留上游现有文档工具链。

输出位置：

| 路径 | 内容 |
| --- | --- |
| `_build/html/` | 可直接由静态 HTTP 服务托管的网站 |
| `_build/html/d2l-pytorch-notebooks.zip` | 当前源码导出的 PyTorch Notebook、图片和辅助库 |
| `_build/notebooks/pytorch/` | Notebook 生成目录，可重新生成，不用于保存个人练习 |
| `_build/book-build-report.json` | 源码提交、构建时间、输出数量及执行范围 |
| `_build/logs/book-build.log` | Python 构建器日志 |
| `_build/logs/sphinx.log` | Sphinx 渲染日志 |
| `_build/logs/book-console.log` | 本次完整控制台输出，含上游依赖告警 |

供用户编辑的 Notebook 位于根目录 `notebooks/pytorch/`。构建成功后运行 `deployment/prepare_notebooks.py`，只补充该目录缺失的文件并设置 `d2l-pytorch` 内核，保留已有个人练习；不要用整目录复制覆盖用户文件。

## 安装与构建

在仓库根目录运行。此次验证使用 Python 3.10.12 构建文档；学习环境的 Python 版本不受此限制。

```bash
uv venv --python python3.10 .venv-book
uv pip install --python .venv-book/bin/python -r deployment/requirements-book.in
mkdir -p _build/logs
.venv-book/bin/python deployment/build_book.py --jobs 16 > _build/logs/book-console.log 2>&1
.venv/bin/python deployment/prepare_notebooks.py
```

`--jobs` 控制 RST 转换与 Sphinx 并行工作进程，默认最多 16 个；四种框架的源文件转换最多同时运行 4 个进程。转换过程不启动训练内核。Pandoc 由 `pypandoc-binary` 随环境安装，脚本自动将其目录加入子进程 PATH，无需系统安装 LaTeX、Pandoc 或 SVG 转 PDF 程序。

`requirements-book.in` 固定上游 `d2l-book` 提交 `2322e565deda1605451bc3976be3f21c731223cc`；`constraints-book.txt` 固定本次使用的构建依赖。对网站脚本或上游 `config.ini` 的修改会触发重新转换章节；输入章节不变时利用 `_build` 中的转换缓存。

本地临时预览示例：

```bash
.venv-book/bin/python -m http.server 8080 --bind 127.0.0.1 --directory _build/html
```

长期运行请使用总部署说明中的服务管理脚本。静态网站可以放在站点根路径或 URL 子目录；Notebook 下载链接会按页面目录转换为相对链接。这里只生成 HTML 和 Notebook，未生成 PDF，因此导航栏不包含不存在的本地 PDF 链接。

## 必要的构建修订

1. 不改写上游 `config.ini`：在构建脚本内关闭 `eval_notebook`，只选择 `index.md` 与章节 Markdown，避免把部署说明当书籍章节。
2. 移除官方 Analytics 配置及自动论坛嵌入，保留讨论区普通链接；保留框架切换所需的空统计回调，避免关闭统计后按钮报错。
3. 保留官方首页内容，移除第三方社交脚本和已注释的旧 Google Maps 代码，继续使用仓库自带静态地图。首页明确说明示例未在本次构建中执行。
4. 导航和 API 源码链接指向实际存在的上游提交 `23d7a5aecceee57d1292c56e90cce307f183bb0a`，便于准确定位教材原始源码。
5. 为 Pandoc 3 输出添加局部表格修复，将被当作标题的 D2L 表格标签恢复成 Sphinx 目标和正确标题，修复表格交叉引用。
6. API 附录从当前 `d2l/torch.py`、`mxnet.py`、`jax.py` 和 `tensorflow.py` 静态提取定义、签名和文档字符串。同名类内方法使用源码最后一次定义；无需导入或运行训练框架。导入的别名、动态注入成员可能不在此附录中，应以源码和实际运行时为准。
7. `d2l-notedown` 仍使用 `pkg_resources`，构建环境固定 `setuptools<81` 的已验证版本；上游 Sphinx 文献扩展依赖旧 pybtex 内部接口，因此固定 `pybtex==0.24.0`。这些约束仅作用于独立文档环境。

## 本次实际验证结果

2026-10-18（北京时间）基于源码提交 `23d7a5aecceee57d1292c56e90cce307f183bb0a` 构建成功。构建器读取的是该提交基础上的当前工作区，提交号不代表工作区一定没有部署修改。

- 生成 192 个内容页面，连同搜索、索引共 194 个 HTML 文件。
- 生成 183 个非空 PyTorch Notebook；少数章节在上游没有 PyTorch 分支，未生成空 Notebook。
- 最终 Sphinx 输出为 `build succeeded.`，没有 Sphinx `WARNING:` 或 `ERROR:`。
- 检查了首页、搜索入口与索引、数据操作、自动微分及 API 附录的输出存在且非空。
- 检查生成 JavaScript 中不存在官方 Google Analytics 地址和跟踪 ID，HTML 不包含未替换的导航占位符。
- 抽查首页、数据操作和自动微分三页引用的本地图片、脚本与样式，缺失数为 0。
- 构建输出约 207 MiB，PyTorch Notebook 目录约 49 MiB；体积会随源码和打包内容变化。

已验证主要版本：`d2lbook 1.0.3`（上面固定 Git 提交）、`Sphinx 5.3.0`、`sphinxcontrib-bibtex 2.4.2`、`pybtex 0.24.0`、`mxtheme 0.3.17`、`nbconvert 7.17.2`、`nbformat 5.11.1`、`d2l-notedown 2.1.0`、`pypandoc-binary 1.17`（Pandoc 3.9）、`setuptools 80.10.2`。

上游 Notebook 合并时会产生重复 cell ID 告警，`nbformat` 会为重复项重新分配 ID；这类信息记录在控制台日志中，不是训练结果或 Sphinx 渲染失败。`pkg_resources` 的弃用提醒也会出现在日志中。

## 结果边界

本次仅验证内容转换、静态渲染和必要文件完整性，没有执行全书训练实验。网页包含源码、公式、插图和说明，但没有重新计算代码输出或训练曲线；正文中引用的历史实验结果来自上游作者，不能视为本服务器上的实验结果。Notebook 能否完整运行还取决于对应章节依赖、数据下载、GPU 和代码与当前框架 API 的兼容性，学习环境的运行验证见总部署/实验说明。

网站保留 MathJax CDN 等主题外部资源及上游外链，因此不承诺完全断网可用。未进行全网外链巡检，也没有尝试生成 PDF 或重跑全部章节。PyTorch 不支持的上游专属章节可在完整网站阅读；Notebook 目录中的少数目录链接可能仍指向这些未导出的专属章节。
