# 跟着 TokenSpeed 学大模型推理通信

这份教程从一个两张 GPU 共同完成的矩阵乘法开始，解释大模型为什么需要通信、每张卡发送什么，以及单机和多机运行有什么区别。先把具体数字算明白，再对应到 TokenSpeed 的线性层和通信后端。

**当前可直接学习和运行的是单机双卡通信练习。** 下文记录的结果来自本机真实运行，不是模拟输出。完整模型服务仍有一项依赖冲突，尚未验证模型加载和生成；多机部分目前是启动模板。

## 学习顺序

| 顺序 | 文档 | 学完后能够解释 |
| --- | --- | --- |
| 1 | [第一课：为什么需要 AllReduce](01_allreduce.md) | 为什么每张卡算完后，还要交换并合并结果；rank 是什么 |
| 2 | [第二课：四种通信操作](02_collectives.md) | 相加、拼接、求和后分片、按目标交换有什么区别 |
| 3 | [第三课：从单机到多机](03_single_and_multi_node.md) | NCCL、NVLink、网卡各自负责什么；跨机器如何启动 |
| 4 | [第四课：Row、Column 与 TP](04_row_column_tp.md) | 如何按输入或输出特征分片；为什么 column 和 row 可以衔接 |
| 5 | [第五课：TP、DP、PP、EP](05_tp_dp_pp_ep.md) | 按张量、请求、层、专家分工有什么区别；一个 rank 为什么属于多个组 |
| 6 | [第六课：QKV 并行与 Softmax 合并](06_attention_parallel_softmax.md) | 按 head 和按历史 token 切分有何区别；何时需要 LSE 合并 |
| 7 | [第七课：MoE 并行与通信](07_moe_parallel.md) | 四卡如何组合 DP/TP/EP；专家分发、共享分支与 SP 的维度和通信量 |
| 8 | [第八章：Attention 并行优化](08_attention_parallel_optimization.md) | 用 DeepSeek-V4.1-Flash 理解 AllGather 后移、跨 DP 重分配、O 权重分片与 QCP/KVP/DCP |

每课都包含代码入口和解释。第一、二、四课附上**双卡运行输出和逐项解读**；第三课是多机模板，第五课附上真实 Mapping 类在 CPU 上的分组计算输出，第六课附上 CPU softmax 合并公式演示。

第四、五课已经加入并行策略的源码讲解。后续可以继续学习真实模型中的调度、通信耗时和计算通信重叠；完整模型的多种并行配置目前还没有完成服务实验。

第七课按**基础 AllGather → 路由 AllToAll → 再加入 SP** 三个版本逐步讲解，包含四张配图、张量形状和 CPU 通信量计算记录，并说明共享专家如何衔接。这些是教学计算，不是四卡 MoE 性能实测。

第八章以官方 **DeepSeek-V4.1-Flash** 配置为例，先用 **4 个 token、2 张 GPU** 从输入走到输出，再展开各项优化。明确区分 AG 交换 latent 与随后生成 heads，并拆解 `wo_a`、`wo_b` 的分组投影及跨卡求和。附六张算子与逐 GPU 数据流图、LaTeX 推导和 CPU 数值验证日志。分别说明教学重排方案与 TokenSpeed 固定版本已支持的路径，没有将理论计算量作为 GPU 性能结果。

## 先运行第一课

```bash
cd /home/hadoop-scale-llm/dasc
bash tokenspeed_learning/run_lesson_01.sh
```

脚本会自动选择独立学习环境，启动两个 Python 进程，各控制一张 GPU。成功时最后一行是：

```text
PASS: api=tokenspeed，两张卡都得到完整答案 [4.0, 6.0]
```

第二课的命令是：

```bash
cd /home/hadoop-scale-llm/dasc
bash tokenspeed_learning/run_lesson_02.sh
```

成功时最后一行是：

```text
PASS: 三种通信操作的实际结果均正确
```

无需重新运行安装脚本。`attempt_full_install.sh` 是之前完整安装尝试的记录，并非课程入口。

## 已验证的结果

验证日期：2026 年 11 月 23 日。

| 项目 | 实际结果 | 记录 |
| --- | --- | --- |
| TokenSpeed AllReduce | 两个 rank 均得到 `[4, 6]`，数值断言通过 | [完整日志](logs/01_tokenspeed.log) |
| TokenSpeed AllGather | 两个 rank 均得到 `[1, 2, 3, 4]` | [完整日志](logs/02_tokenspeed.log) |
| TokenSpeed ReduceScatter | rank 0 得到 `[11, 22]`，rank 1 得到 `[33, 44]` | 同上 |
| TokenSpeed AllToAll | rank 0 得到 `[0, 10]`，rank 1 得到 `[1, 11]` | 同上 |
| 真实 ColumnParallelLinear 与 RowParallelLinear | column 输出两份特征分片；row 的 `[4, 2]` 和 `[1, 6]` 合并为 `[5, 8]`，与完整计算一致 | [双卡日志](logs/04_parallel_linear.log) |
| Mapping 的 8-rank 分组计算 | 输出各 rank 的 attention TP/DP、PP、MoE EP 组；仅 CPU 计算，没有启动 8 卡服务 | [分组日志](logs/05_mapping.log) |
| `tokenspeed env` | 命令执行通过 | [环境日志](logs/tokenspeed_env.log) |
| `tokenspeed serve --help` | 帮助命令执行通过，不代表模型服务启动成功 | [帮助输出](logs/tokenspeed_serve_help.log) |

练习使用 TokenSpeed 的 `NcclBackend`，设置 `use_pynccl=False`，由它调用 PyTorch 的 NCCL 进程组。代码会检查实际数值，打印 `PASS` 前必须通过断言。

这些是**通信正确性验证**，没有测量通信带宽或模型吞吐。验证时本机原有 GPU 压力进程仍在运行，因此也不能用这些运行的耗时判断性能。

## 环境与安装范围

| 项目 | 本次使用的配置 |
| --- | --- |
| 源码 | [lightseekorg/tokenspeed](https://github.com/lightseekorg/tokenspeed) |
| 固定版本 | `8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613` |
| 源码目录 | `/home/hadoop-scale-llm/dasc/tokenspeed` |
| Python 环境 | `/home/hadoop-scale-llm/dasc/tokenspeed/.venv-cu129` |
| Python | 3.11.18 |
| PyTorch | 2.14.0+cu126 |
| NCCL | 2.29.3 |
| GPU | 2 × NVIDIA H20-3e |
| GPU 拓扑 | 安装时 `nvidia-smi topo -m` 显示两卡之间为 `NV18` |
| 本地编译工具 | CUDA Toolkit 12.9，nvcc 12.9.86 |

环境名称中的 `cu129` 对应本地编译工具配置，**不意味着 PyTorch 的 CUDA 运行时也是 12.9**。这里的 PyTorch 是 cu126，CUDA runtime 包为 12.6.77。详细信息保存在 [installation_status.json](installation_status.json)。

完整依赖检查仍报告：

```text
tokenspeed-mooncake 0.3.13.post20261119 has requirement nvidia-cuda-runtime-cu12<13,>=12.9, but you have nvidia-cuda-runtime-cu12 12.6.77.
```

这是 Mooncake 与当前 PyTorch 环境的 CUDA runtime 要求冲突，见 [检查日志](logs/pip_check.log)。当前课程不使用 Mooncake。模型推理、Mooncake 传输、MoE 专家通信和跨机器运行均尚未验证。原有研究环境、系统 CUDA 和驱动没有被修改。

## 文件入口

| 文件 | 用途 |
| --- | --- |
| [01_allreduce.py](01_allreduce.py) | 两张卡完成一个线性层的数值例子，含中文注释 |
| [02_collectives.py](02_collectives.py) | 三种其他通信操作的数值例子 |
| [run_lesson_01.sh](run_lesson_01.sh) | 第一课启动脚本 |
| [run_lesson_02.sh](run_lesson_02.sh) | 第二课启动脚本 |
| [04_parallel_linear.py](04_parallel_linear.py) | 实例化真实 TokenSpeed column/row 层，对照开关与完整计算 |
| [run_lesson_04.sh](run_lesson_04.sh) | 第四课双卡启动脚本 |
| [05_mapping.py](05_mapping.py) | 在 CPU 上查看 TP/DP/PP/EP 进程分组 |
| [run_two_nodes.sh](run_two_nodes.sh) | 两台机器各一张 GPU 的启动模板，未跨机实测 |
| [env.sh](env.sh) | 选择独立学习环境和本地 CUDA 工具 |

第一次学习建议只读第一课，先能够手算 `[1, 2] + [3, 4] = [4, 6]`，再理解通信代码为什么放在矩阵乘法之后。
