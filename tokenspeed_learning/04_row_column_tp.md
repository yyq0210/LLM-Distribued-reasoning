# 第四课 从源码理解 Column Parallel 和 Row Parallel

**TP 把同一层的计算分给多张卡。Column parallel 和 row parallel 是线性层实现 TP 的两种切分方式。** 本课先弄清输入、输出和权重的形状，再读 TokenSpeed 的类，最后对照双卡实测结果。

源码版本固定为 `8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613`。[返回教程入口](README.md)。

## 1 先统一矩阵的写法

第一课为了方便手算使用 $y = Wx$，其中 $x$ 是列向量。TokenSpeed 的线性层类注释使用另一种等价写法：

$$
Y = XA + b
$$

这里每个 token 是 `X` 的一行：

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| $X$ | $T\times H_{\mathrm{in}}$ | $T$ 个 token，每个 token 有 $H_{\mathrm{in}}$ 个输入特征 |
| $A$ | $H_{\mathrm{in}}\times H_{\mathrm{out}}$ | 数学公式中的权重矩阵 |
| $Y$ | $T\times H_{\mathrm{out}}$ | 每个 token 的输出特征 |
| $b$ | $H_{\mathrm{out}}$ | 可选的偏置 |

普通 prefill 中，`T` 可以是这一批实际处理的多个输入 token；普通 decode 中，每个活跃请求通常贡献一个 token。这里的 `T` 不一定等于请求数量。

**row/column 的名字是针对公式里的 $A$。** PyTorch 的实际 `weight` 通常存成 $H_{\mathrm{out}}\times H_{\mathrm{in}}$，也就是 $A^{\mathsf T}$。

这点在 [dense/unquant.py](https://github.com/lightseekorg/tokenspeed/blob/8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613/python/tokenspeed/runtime/layers/dense/unquant.py) 的 `UnquantizedLinearMethod.create_weights` 中能直接看到：分配顺序是输出维度在前、输入维度在后。所以不要仅凭 `weight.shape` 的第一维或第二维，猜它是 row 还是 column parallel。

## 2 Column parallel 切输出特征

把 $A$ 沿列切分：

$$
A = \begin{bmatrix}A_0 & A_1\end{bmatrix},
\qquad
Y = \begin{bmatrix}XA_0 & XA_1\end{bmatrix}
$$

两张卡都拿完整的输入特征，分别计算不同的输出特征：

```text
GPU 0：完整 X × A 的前半列 → Y 的前半特征
GPU 1：完整 X × A 的后半列 → Y 的后半特征
```

以 $H_{\mathrm{in}}=4$、$H_{\mathrm{out}}=6$、$\mathrm{TP}=2$ 为例：

| 项目 | 每张 GPU 上的形状 |
| --- | --- |
| 输入 $X$ | $T\times 4$ |
| 本地数学权重 $A_i$ | $4\times 3$ |
| 实际存储的 `weight` | $3\times 4$ |
| 本地输出 $Y_i$ | $T\times 3$ |

每张卡得到的是**一部分输出特征的完整数值**。两边算的是不同的特征，不能直接把它们相加。

如果后续需要每张卡持有全部输出，可以 AllGather，拼成 `[T, 6]`；如果下一层可以消费分片，就保持 `[T, 3]`，省去这次拼接。

### 对照 ColumnParallelLinear 源码

阅读 [linear.py](https://github.com/lightseekorg/tokenspeed/blob/8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613/python/tokenspeed/runtime/layers/linear.py) 的 `ColumnParallelLinear`，当前版本约第 359 行开始。

初始化时：

```python
self.output_size_per_partition = divide(self.output_size, self.tp_size)
```

输出维度被 TP 数量均分。`forward` 约第 503 行，省略特殊分支后是：

```python
output_parallel = self.quant_method.apply(self, input_, bias)
if self.gather_output and self.tp_size > 1:
    output = all_gather(output_parallel, self.tp_group, dim=-1)
else:
    output = output_parallel
```

`dim=-1` 表示沿最后一维，也就是特征维度拼接；没有把不同 token 的批次拼起来。`gather_output=False` 是该类的默认配置，意味着输出保持分片，不是少算了一部分答案。

这里 `quant_method` 是执行线性运算的统一接口，名字包含 quant 也不意味着一定量化；本课采用的是未量化 FP32 路径。

## 3 Row parallel 切输入特征

把数学权重 $B$ 沿行切开，并把输入特征按同样方式切开：

$$
B = \begin{bmatrix}B_0 \\ B_1\end{bmatrix},
\qquad
Z = \begin{bmatrix}Z_0 & Z_1\end{bmatrix},
\qquad
Y = Z_0B_0 + Z_1B_1
$$

每张卡只拿一部分输入，算出对**全部输出位置**的部分贡献。

以 $H_{\mathrm{in}}=6$、$H_{\mathrm{out}}=2$、$\mathrm{TP}=2$ 为例：

| 项目 | 每张 GPU 上的形状 |
| --- | --- |
| 本地输入 $Z_i$ | $T\times 3$ |
| 本地数学权重 $B_i$ | $3\times 2$ |
| 实际存储的 `weight` | $2\times 3$ |
| 本地部分结果 | $T\times 2$ |
| 求和后的完整输出 | $T\times 2$ |

注意本地结果的形状已经是 `[T, 2]`，但数值还不完整。不能只凭形状判断一个张量是否已经包含所有卡的贡献。

### 对照 RowParallelLinear 源码

同一份 [linear.py](https://github.com/lightseekorg/tokenspeed/blob/8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613/python/tokenspeed/runtime/layers/linear.py) 中，`RowParallelLinear` 约第 1142 行开始：

```python
self.input_size_per_partition = divide(input_size, self.tp_size)
```

`forward` 约第 1301 行，先决定输入是否已经分片：

```python
if self.input_is_parallel:
    input_parallel = input_
else:
    splitted_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
    input_parallel = splitted_input[self.tp_rank].contiguous()
```

`input_is_parallel=True` 表示上游已经给了本卡对应的输入分片。False 路径是在本地从完整输入取一片，这个切片操作本身不是跨 GPU 的 Scatter 通信。

本地矩阵乘法完成后：

```python
if self.reduce_results and self.tp_size > 1:
    output = all_reduce(output_parallel, self.tp_group)
else:
    output = output_parallel
```

AllReduce SUM 合并相同输出位置上的部分贡献。`reduce_results=False` 则返回部分结果，后续必须按模型的数据流正确处理，不能直接当完整输出使用。

还有一个细节：该普通 forward 路径只让 TP rank 0 在矩阵乘法中加 bias。否则两张卡都先加同一个 bias，再 AllReduce，就会把 bias 加两遍。`skip_bias_add=True` 时则把 bias 返回给外部处理。

## 4 为什么通常把 Column 和 Row 接起来

考虑一个教学用的两层网络：

$$
Y = \operatorname{ReLU}(XA)B
$$

第一层按输出切分，第二层恰好按输入切分：

```text
                  Column                  Row
GPU 0：X → XA0 → ReLU → Z0 → Z0B0 ─┐
                                   ├→ AllReduce SUM → 完整 Y
GPU 1：X → XA1 → ReLU → Z1 → Z1B1 ─┘
```

ReLU 是逐元素操作，各张卡可以独立处理自己的特征分片。第一层的分片可以直接交给第二层，因此这里**不需要在两层之间 AllGather**。

不过，不能把 ReLU 随意放在第二层部分结果上，再做求和：通常 $\operatorname{ReLU}(P_0+P_1)\ne\operatorname{ReLU}(P_0)+\operatorname{ReLU}(P_1)$。需要知道张量是“完整特征的分片”，还是“尚未合并的部分和”。

真实 Llama 的 MLP 使用门控激活，结构更接近：

```text
X → gate_proj 和 up_proj 的 column 分片
  → 本地 SiLU(gate) × up
  → down_proj 的 row 分片
  → 合并部分结果
```

阅读 [mlp.py](https://github.com/lightseekorg/tokenspeed/blob/8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613/python/tokenspeed/runtime/layers/mlp.py) 的 `GatedMLP.tensor_parallel`，约第 135 行：它明确创建 `MergedColumnParallelLinear` 作为 `gate_up_proj`，创建 `RowParallelLinear` 作为 `down_proj`。Merged 表示把 gate/up 两个投影组织在一起；各 rank 持有相互对应的 gate 和 up 分片，不能把它理解成一张卡只算 gate、另一张卡只算 up。

## 5 真实 TokenSpeed 层的双卡运行结果

运行：

```bash
cd /home/hadoop-scale-llm/dasc
bash tokenspeed_learning/run_lesson_04.sh
```

代码：[04_parallel_linear.py](04_parallel_linear.py)。这个练习实际实例化 TokenSpeed 两个线性层类，使用它们的 `forward`；全局通信后端明确配置为 `NcclBackend(use_pynccl=False)`。没有加载完整大模型。

输入和两层数学权重为：

$$
X = \begin{bmatrix}1 & 2 & 3 & 4\end{bmatrix}
$$

$$
A = \begin{bmatrix}
1 & 0 & 0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0
\end{bmatrix},
\qquad
B = \begin{bmatrix}
1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1
\end{bmatrix}
$$

$$
\begin{aligned}
XA &= \begin{bmatrix}1 & 2 & 3 & 4 & 1 & 2\end{bmatrix}, \\
\operatorname{ReLU}(XA)B
&= \begin{bmatrix}1+3+1 & 2+4+2\end{bmatrix}
 = \begin{bmatrix}5 & 8\end{bmatrix}.
\end{aligned}
$$

2026 年 11 月 23 日实测数值输出如下。原始 [日志](logs/04_parallel_linear.log) 还包含与第二课相同的接口弃用警告，下面仅省略该警告。

```text
rank=0
  input X: [[1.0, 2.0, 3.0, 4.0]]
  column weight [out_local, in]: [3, 4]
  column gather_output=False: [[1.0, 2.0, 3.0]]
  column gather_output=True:  [[1.0, 2.0, 3.0, 4.0, 1.0, 2.0]]
  row weight [out, in_local]: [2, 3]
  row reduce_results=False: [[4.0, 2.0]]
  row reduce_results=True:  [[5.0, 8.0]]
  dense reference: [[5.0, 8.0]]
rank=1
  input X: [[1.0, 2.0, 3.0, 4.0]]
  column weight [out_local, in]: [3, 4]
  column gather_output=False: [[4.0, 1.0, 2.0]]
  column gather_output=True:  [[1.0, 2.0, 3.0, 4.0, 1.0, 2.0]]
  row weight [out, in_local]: [2, 3]
  row reduce_results=False: [[1.0, 6.0]]
  row reduce_results=True:  [[5.0, 8.0]]
  dense reference: [[5.0, 8.0]]
PASS: actual TokenSpeed ColumnParallelLinear + RowParallelLinear = dense reference
```

手算两张卡的 row 部分结果：

$$
\begin{aligned}
P_0 &= \begin{bmatrix}1 & 2 & 3\end{bmatrix}B_0
     = \begin{bmatrix}1+3 & 2\end{bmatrix}
     = \begin{bmatrix}4 & 2\end{bmatrix}, \\
P_1 &= \begin{bmatrix}4 & 1 & 2\end{bmatrix}B_1
     = \begin{bmatrix}1 & 4+2\end{bmatrix}
     = \begin{bmatrix}1 & 6\end{bmatrix}, \\
Y &= P_0+P_1 = \begin{bmatrix}5 & 8\end{bmatrix}.
\end{aligned}
$$

脚本中的 gather 和不 gather、reduce 和不 reduce 各用一个独立实例，目的是展示开关区别。正式的示范链直接把不 gather 的本地特征送给 row 层，没有消费完整的中间结果。教学参考矩阵在每卡保留一份用于断言，因此这不是显存占用或速度测试。

## 6 为什么模型里有 reduce_results=False

在 [llama.py](https://github.com/lightseekorg/tokenspeed/blob/8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613/python/tokenspeed/runtime/models/llama.py) 的 `LlamaAttention` 中，可以看到：

- `qkv_proj` 使用 `QKVParallelLinear`，它继承 column parallel 的基本结构，针对 Q/K/V 的分片做了专门处理。
- `o_proj` 使用 `RowParallelLinear`，但设置了 `reduce_results=False`。

这不意味着省掉了必要的合并。在 [base/decoder_layer.py](https://github.com/lightseekorg/tokenspeed/blob/8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613/python/tokenspeed/runtime/models/base/decoder_layer.py) 中，attention 之后会调用 `comm_manager.post_attn_reduce_norm`；MLP 周围也有对应的通信步骤。

再进入 [comm_manager.py](https://github.com/lightseekorg/tokenspeed/blob/8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613/python/tokenspeed/runtime/distributed/comm_manager.py)：

| 方法 | 本课关心的处理 |
| --- | --- |
| `post_attn_comm` | 根据布局做 AllReduce 或 token ReduceScatter |
| `pre_dense_comm` | 布局需要时，把不同 token 行 AllGather 到 dense 组 |
| `post_dense_comm` | 根据布局合并并复制结果，或合并后按 token 行分散 |

实际还可以把通信与残差、归一化融合。这里强调的是语义：**row 层产生的部分和仍需要被正确合并，只是合并不一定写在线性层自己的 forward 里。**

还有两个阅读注意点。第一，GQA/MQA 的 KV head 数量可能比 TP 数量少，这时 QKV 中某些 KV 权重会复制，不能假定每个张量都严格除以 TP。第二，列并行里的 feature AllGather 和通信管理器中的 token AllGather 拼的是不同维度，阅读时要同时看张量形状。

## 7 TP 与这两个类是什么关系

TP（Tensor Parallelism，张量并行）是一种把同一层的张量和计算拆到多个进程的策略。Row/column parallel 是线性层的两种实现方式，不是两种额外需要相乘的 GPU 数量。

比如 TP=8，通常表示某个 TP 组中的 8 个进程合作算同一批 token 的一层。它不表示同时处理 8 份互相独立的请求，也不表示一个算 row、另一个算 column。

读完这一课，应当能回答：当前卡的张量是完整数据、特征分片，还是部分和？下一步需要拼接、求和，还是可以继续本地计算？

## 8 层内并行与层间并行要分开理解

“层间”可能指两件事：同一 TP 组如何衔接相邻算子，或者不同 GPU 负责不同层段的 PP。它们不能混为一谈。

| 问题 | 分工方式 | 依赖如何满足 |
| --- | --- | --- |
| 一个线性层内部怎么分工 | TP：column 切输出特征，row 切输入特征 | 分别生成特征分片，或合并部分和 |
| 相邻线性层怎么衔接 | 同一 TP 组按相容的布局连接 | column 输出可直接作为 row 输入；必要时通信转换布局 |
| 不同层放在不同设备上 | PP：每个阶段拥有一段层 | 前一阶段发送边界激活，后一阶段接着计算 |

### 相邻算子衔接仍然有先后依赖

对于 $Y=\phi(XA)B$，同一个输入必须先完成所需的第一层计算，再执行第二层。Column→逐元素激活→Row 的好处是中间特征分片可直接沿本卡继续流动，不需要先拼回完整张量；它并没有消除网络的先后依赖。

在普通复制输入的 TP 路径中，row 层末尾将部分和 AllReduce，得到下一部分网络所需的完整 hidden states。采用 token 分片布局时，也可能使用 ReduceScatter，在后续需要完整 token 行集合时再 AllGather。残差和归一化同样必须遵守当前布局，不能把尚未合并的部分和直接当完整值使用。

这也是经典 [Megatron 的 MLP 与 attention 张量并行](https://research.nvidia.com/labs/adlr/MegatronLM/) 中 column/row 配对的目的：保留局部计算链，减少中间通信。

### PP 如何让不同层段并行工作

以 32 层、PP=2 为概念示例：阶段 0 持有层 0–15，阶段 1 持有层 16–31。对同一批输入 A，阶段 1 必须等待阶段 0 的结果；多个独立微批 A、B、C 才能填满流水线：

| 理想化时隙 | 阶段 0：层 0–15 | 阶段 1：层 16–31 |
| --- | --- | --- |
| 1 | A | 等待 |
| 2 | B | A |
| 3 | C | B |
| 4 | 后续微批或等待 | C |

边界通常传输 hidden states 和实现需要的 residual 等状态，各阶段为自己的层维护 KV cache。每一层有自己的 Q/K/V 权重，下一阶段根据收到的 hidden states 生成自己各层的 Q/K/V，并不是沿流水线传递同一份 Q/K/V 来代替后续层计算。

该表是一般 PP 的教学调度，不是本机测试结果，也不代表同一请求的连续 decode token 可以独立流水执行；后一个 token 通常依赖前一个 token 的生成结果。本地 TokenSpeed 固定版本的 [server_args.py](https://github.com/lightseekorg/tokenspeed/blob/8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613/python/tokenspeed/runtime/utils/server_args.py) 中，正式 PP 路径要求 `disaggregation_mode=prefill`，属于 prefill chunk pipeline，不能据此宣称已支持或验证完整 decode 流水线。

TP 与 PP 可以组合。例如 TP=2、PP=2 时，GPU 0/1 合作计算前 16 层，GPU 2/3 合作计算后 16 层。TP 负责每个阶段内的一层怎么拆，PP 负责哪些层属于哪个阶段。

## 9 Q K V 如何套用线性层并行

Q/K/V 投影是三个线性变换，可以融合执行。以普通 head TP、GQA、无 bias 为例，设输入维度 4,096，Q head 数 32，KV head 数 8，每个 head 维度 128，TP=4；以下是形状推导，没有新增四卡模型测试。

| 投影 | 完整数学权重形状 | 每卡数学权重形状 | 每卡生成的张量 |
| --- | --- | --- | --- |
| $W_Q$ | $4096\times4096$ | $4096\times1024$ | $T\times8\times128$ |
| $W_K$ | $4096\times1024$ | $4096\times256$ | $T\times2\times128$ |
| $W_V$ | $4096\times1024$ | $4096\times256$ | $T\times2\times128$ |

各卡输入均为同一份 $X\in\mathbb{R}^{T\times4096}$。采用连续 GQA 分组时，GPU 0 负责 Q head 0–7、KV head 0–1；GPU 1 负责 Q head 8–15、KV head 2–3；其他卡依次类推。每个 Q head 只访问对应的 KV head。

每张卡都保存自己负责的 Q、K、V 权重块，通常融合为一次本地矩阵乘法：

$$
\begin{bmatrix}Q_r&K_r&V_r\end{bmatrix}
=X\begin{bmatrix}W_{Q,r}&W_{K,r}&W_{V,r}\end{bmatrix}
$$

本地融合输出宽度为 $1024+256+256=1536$。源码中的 `qkv.split(...)` 将这 1,536 个特征拆成 Q、K、V 三段，不涉及跨卡发送。**不是 GPU 0 只算 Q、GPU 1 只算 K、GPU 2 只算 V。** 权重加载器分别选择 Q/K/V 的本地 head，不能把全局融合矩阵不分段地随意均分。

在普通 head TP 中，每张卡保存本地 KV head 的完整可见历史，因此本地 attention 可独立计算，无需跨卡 LSE 合并。Prefill 会生成多个新 token 的 Q/K/V；普通单步 decode 每个活跃请求通常只生成当前 token 的 Q/K/V，再读取历史 KV cache，不会重新投影全部历史 token。

Attention 后，每卡产生 $O_r\in\mathbb{R}^{T\times1024}$，直接送给 row-parallel 输出投影：

$$
P_r=O_rW_{O,r},\qquad
W_{O,r}\in\mathbb{R}^{1024\times4096},\qquad
Y=\sum_{r=0}^{3}P_r
$$

每个 $P_r$ 都是 $T\times4096$，但只是完整输出的部分和。

```text
同一份完整输入 X
  ├─ GPU 0：本地 QKV 投影 → 本地 heads 的 attention → 本地输出投影 P₀ ─┐
  ├─ GPU 1：本地 QKV 投影 → 本地 heads 的 attention → 本地输出投影 P₁ ─┤
  ├─ GPU 2：本地 QKV 投影 → 本地 heads 的 attention → 本地输出投影 P₂ ─┤
  └─ GPU 3：本地 QKV 投影 → 本地 heads 的 attention → 本地输出投影 P₃ ─┘
                                  合并部分和 → 完整 Y
```

若 KV head 数少于 TP 数，本地 TokenSpeed 会复制所需的 KV head，而 Q head 仍分片；具体例子见[第六课](06_attention_parallel_softmax.md)。把同一 head 的历史 token 再分布到多卡属于另一条切分维度，需要按实际 KV 布局处理归一化与通信。

对应源码入口：`QKVParallelLinear` 负责 QKV 权重及 head 分片，`LlamaAttention.forward` 连接 QKV→attention→row 输出投影；外层 `CommManager` 处理尚未完成的归约。不要把 `reduce_results=False` 误解为完全没有通信。

继续阅读：[第五课：TP、DP、PP、EP 与进程分组](05_tp_dp_pp_ep.md)、[第六课：Attention 并行与 Softmax 合并](06_attention_parallel_softmax.md)。
