# 第二课 用真实输出区分四种通信操作

上一课的 AllReduce 把多个部分结果相加，并把完整结果交给每个参与者。本课再看三种操作：AllGather、ReduceScatter、AllToAll。关键是看清楚**输出的每个数字来自哪里**。

代码：[02_collectives.py](02_collectives.py)。完整日志：[02_tokenspeed.log](logs/02_tokenspeed.log)。[返回教程入口](README.md)。

## 1 运行命令与完整数值输出

```bash
cd /home/hadoop-scale-llm/dasc
bash tokenspeed_learning/run_lesson_02.sh
```

以下是 2026 年 11 月 23 日在本机两张 H20-3e 上实际运行的数值输出：

```text
AllGather: rank 0: [1.0, 2.0] -> [1.0, 2.0, 3.0, 4.0]
AllGather: rank 1: [3.0, 4.0] -> [1.0, 2.0, 3.0, 4.0]
ReduceScatter: rank 0: [1.0, 2.0, 3.0, 4.0] -> [11.0, 22.0]
ReduceScatter: rank 1: [10.0, 20.0, 30.0, 40.0] -> [33.0, 44.0]
AllToAll: rank 0: [0.0, 1.0] -> [0.0, 10.0]
AllToAll: rank 1: [10.0, 11.0] -> [1.0, 11.0]
PASS: 三种通信操作的实际结果均正确
```

原始日志中还有 PyTorch 的 `FutureWarning`，提示 TokenSpeed 当前调用的 `all_gather_into_tensor` 和 `reduce_scatter_tensor` 接口将弃用。上面为方便阅读省略了警告，保留了全部数值结果。此次各项数值断言均通过，警告内容可在完整日志中查看。

## 2 AllGather 把分片拼起来

代码：

```python
gathered = backend.all_gather(x, group)
```

| 参与者 | 输入 | 输出 |
| --- | --- | --- |
| rank 0 | `[1, 2]` | `[1, 2, 3, 4]` |
| rank 1 | `[3, 4]` | `[1, 2, 3, 4]` |

先放 rank 0 的分片，再放 rank 1 的分片。**没有做加法**，每个进程的输出都包含全部分片，长度从 2 变成 4。

当一份完整数据被分散保存在多张卡上，而后续计算需要每张卡都拿到全部数据时，就可能需要这种操作。它是否适合某一层，取决于具体的数据切分方式。

## 3 ReduceScatter 先求和再分片

代码：

```python
reduced = backend.reduce_scatter(x, group)
```

先从数学上看逐元素求和：

```text
rank 0 的输入  [ 1,  2,  3,  4]
rank 1 的输入  [10, 20, 30, 40]
逐元素求和     [11, 22, 33, 44]
```

然后把结果分成两块：

| 参与者 | 最终输出 |
| --- | --- |
| rank 0 | `[11, 22]` |
| rank 1 | `[33, 44]` |

“先求和再分片”描述的是操作的数学含义，不要求底层真的先在每张卡上产生一个完整求和结果。

它和 AllReduce 的差别是：**AllReduce 后每个进程都有完整结果，ReduceScatter 后每个进程只持有结果的一片。** 如果下一步可以继续在分片上计算，就不一定要立即把完整结果复制到每张卡。

对这个等长分片例子，把 ReduceScatter 的两个输出再做 AllGather，就能得到 `[11, 22, 33, 44]`，在数学上等价于直接对原输入做 SUM AllReduce。不能据此认定两种实际调用方式性能一定相同。

## 4 AllToAll 按接收者交换数据

代码：

```python
backend.all_to_all_single(exchanged, x, group)
```

本例每个进程把输入分成两块：第一块发给 rank 0，第二块发给 rank 1。

| 发送者 | 发给 rank 0 | 发给 rank 1 |
| --- | --- | --- |
| rank 0 | `[0]` | `[1]` |
| rank 1 | `[10]` | `[11]` |

接收端按来源 rank 排列收到的数据：

```text
rank 0 收到：来自 rank 0 的 [0]，来自 rank 1 的 [10] → [0, 10]
rank 1 收到：来自 rank 0 的 [1]，来自 rank 1 的 [11] → [1, 11]
```

这里既没有求和，也没有让每个人得到所有输入。**不同数据有不同目的地。**

MoE（混合专家）模型中，把不同 token 的数据送到对应专家所在的设备，是理解这种交换模式的一个例子。真实专家通信可能使用专用后端、变长消息和重叠调度；本课没有运行完整 MoE，也没有验证 DeepEP。

## 5 放在一起对照

| 操作 | 对数据做什么 | 每个进程最后拿到什么 |
| --- | --- | --- |
| AllReduce SUM | 各进程对应位置相加 | 完整求和结果 |
| AllGather | 按 rank 拼接各个分片 | 全部分片 |
| ReduceScatter SUM | 对应位置相加，并划分结果 | 求和结果的一片 |
| AllToAll | 按目标 rank 发送对应分块 | 各来源发给自己的块 |

所有这些都是集体通信：组内进程需要协调参加，并按兼容的顺序、形状和数据类型调用。当前例子刻意采用两卡和等长小张量，方便看清操作本身。

## 6 回到源码

四种操作的本课实现都能在 [NcclBackend](https://github.com/lightseekorg/tokenspeed/blob/8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613/python/tokenspeed/runtime/distributed/comm_backend/nccl.py) 中找到：

```text
all_reduce         → torch.distributed.all_reduce
all_gather         → all_gather_single → torch.distributed.all_gather_into_tensor
reduce_scatter     → torch.distributed.reduce_scatter_tensor
all_to_all_single  → torch.distributed.all_to_all_single
```

这是 `use_pynccl=False` 配置下的调用关系。学习时先理解输入输出，再看函数如何分配输出张量、获取通信组并调用底层接口。

继续阅读：[第三课：从单机到多机](03_single_and_multi_node.md)。
