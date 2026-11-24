# 大模型分布式推理学习笔记

跟着 TokenSpeed 源码学习单机与多机推理通信。从双卡 AllReduce 开始，逐步理解线性层、Attention 和 MoE 如何划分计算、交换数据，以及保证结果正确。

完整教程保存在 [tokenspeed_learning](tokenspeed_learning/README.md)，包含讲解、LaTeX 公式、PNG/SVG 配图、练习代码和原始运行日志。

## 章节

| 章节 | 内容 |
| --- | --- |
| [第一课](tokenspeed_learning/01_allreduce.md) | 为什么需要 AllReduce，双卡数值例子 |
| [第二课](tokenspeed_learning/02_collectives.md) | AllGather、ReduceScatter、AllToAll |
| [第三课](tokenspeed_learning/03_single_and_multi_node.md) | 单机与多机通信，启动方式和网络 |
| [第四课](tokenspeed_learning/04_row_column_tp.md) | Row Parallel、Column Parallel 与 TP |
| [第五课](tokenspeed_learning/05_tp_dp_pp_ep.md) | TP、DP、PP、EP 的分组与源码 |
| [第六课](tokenspeed_learning/06_attention_parallel_softmax.md) | QKV 切分、LSE 合并与 online softmax |
| [第七课](tokenspeed_learning/07_moe_parallel.md) | MoE 从 AllGather 到 AllToAll，再引入 SP |
| [第八章](tokenspeed_learning/08_attention_parallel_optimization.md) | Attention 通信优化，四 token 双卡流水线，wo_a / wo_b |

## 获取源码

```bash
git clone --recurse-submodules https://github.com/yyq0210/LLM-Distribued-reasoning.git
cd LLM-Distribued-reasoning
```

TokenSpeed 以子模块引用，固定于 `8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613`。教程中的源码链接也固定到该版本，避免上游变动使解释失效。已普通克隆的仓库可运行 `git submodule update --init --recursive` 获取源码。

## 运行范围

第一、二、四课保留原机器上的双卡 NCCL/线性层正确性运行记录；其余明确标注为 CPU 公式计算、布局示意或未经多机验证的模板。完整模型推理性能与多机服务没有在这套教程中完成实测。

无需 GPU 的部分可以在仓库根目录运行：

```bash
python3 tokenspeed_learning/06_softmax_merge.py
python3 tokenspeed_learning/moe_layout_dimensions.py
python3 tokenspeed_learning/08_attention_calculations.py
```

配图重绘需要 Pillow，字体及许可证已放在 `tokenspeed_learning/assets/fonts/`。

GPU 练习需要另行准备兼容的 PyTorch/CUDA/TokenSpeed 环境。仓库没有上传原机器的虚拟环境、CUDA 工具目录和下载包。`env.sh` 记录的是原环境路径；教程及日志里的 `/home/hadoop-scale-llm/dasc` 是原运行位置，其他机器需替换为自己的路径。具体依赖状态和已知冲突见[教程入口](tokenspeed_learning/README.md)。安装尝试脚本作为记录保留，不应当作已经验证通过的通用安装器。

## 同步后续学习笔记

原始学习目录与发布仓库分开存放时，可在本仓库运行：

```bash
python3 scripts/sync_learning.py /path/to/tokenspeed_learning
```

同步会排除安装包、CUDA 工具和 Python 缓存，并把教程中的本地 TokenSpeed 链接转换为固定版本的 GitHub 链接；不会修改原始学习目录，也不会自动推送。

## 第三方材料

TokenSpeed 使用其上游许可证；DeepSeek 参考实现的来源、哈希和许可证保存在 `tokenspeed_learning/references/deepseek_v41_flash/`；配图字体使用 SIL Open Font License。引用配置与代码用于对照阅读，不包含模型权重。
