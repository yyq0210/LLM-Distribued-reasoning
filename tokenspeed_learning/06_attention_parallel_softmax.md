# 第六课 Attention 如何切分 Q K V 与合并 Softmax

**是否需要跨卡合并 softmax，取决于同一个 query、同一个 head 的 key 集合有没有被切开。** 普通按 head 的 TP 各自完成完整 softmax；若历史 K/V 分布在多张卡，每张卡只算部分 key，就需要恢复全局归一化。

本课先讲标准 softmax attention，再对应本地 TokenSpeed 提交 `8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613` 的代码。数学演示已在 CPU 上执行；本课没有新增多卡 attention kernel 或完整模型服务测试。[返回教程入口](README.md)。

## 1 Softmax 到底对哪个维度计算

先省略 batch 维度。对一个请求，常见逻辑形状是：

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| $Q$ | $T_q\times H_q\times d$ | query token 数、query head 数、每个 head 的维度 |
| $K$ | $T_k\times H_{kv}\times d$ | key token 数、KV head 数、head 维度 |
| $V$ | $T_k\times H_{kv}\times d_v$ | 对应的 value；维度 $d_v$ 不必总与 $d$ 相同 |

对 query 位置 $t$、query head $h$，设它使用 KV head $g(h)$：

$$
s_{t,h,j}=\frac{q_{t,h}k_{j,g(h)}^{\mathsf T}}{\sqrt d}+M_{t,j}
$$

$$
o_{t,h}=\sum_{j=1}^{T_k}
\frac{\exp(s_{t,h,j})}{\sum_{u=1}^{T_k}\exp(s_{t,h,u})}
v_{j,g(h)}
$$

$M_{t,j}$ 表示 causal mask 等约束，被屏蔽的位置可以看作分数为负无穷。**分母在 key 的 token 维度上求和，没有跨 head 或跨 query 求和。**

Prefill 通常有很多 query 行；普通单步 decode 中，一个请求通常只有当前 token 的一个 query 行，但会访问该请求的历史 KV。这不改变 softmax 的归一化方向。

## 2 最常见的 TP 按 head 切分

以普通 MHA 为例：8 个 Q/K/V head，TP=2。两张卡分别负责 4 个完整 head：

```text
同一批输入 hidden states X
    │
    ├─ rank 0：Q/K/V heads 0–3 → 每个 head 看完整的可见历史 → O0
    │
    └─ rank 1：Q/K/V heads 4–7 → 每个 head 看完整的可见历史 → O1
```

普通 head TP 会把每个 head 对应的 Q/K/V 投影权重分配给负责它的 GPU。这些 GPU 读取同一批 token 的完整输入特征，各自乘自己的权重，直接生成负责的 head 的 Q/K/V。这里的“输出 head”指投影结果中的 attention head，不是模型最后的词表输出头。

### 每个 head 的 Q K V 是否独立

在普通 MHA 中，第 $h$ 个 head 有自己的投影参数块：

$$
Q^{(h)}=XW_Q^{(h)},\qquad
K^{(h)}=XW_K^{(h)},\qquad
V^{(h)}=XW_V^{(h)}
$$

每个 head 使用自己的 $Q^{(h)},K^{(h)},V^{(h)}$ 完成 attention，不需要其他 head 的 attention 输出。但这些 head 读取同一个 $X$，最后还会由输出投影混合，所以这里说的是**计算可以分开**，不能理解为统计独立，也不是各自只读取输入向量的一小段。

这些参数不一定以许多小矩阵分别存储。实现通常把所有 head 的参数打包为大矩阵，甚至把 Q/K/V 三个投影融合为一个 GEMM；逻辑上仍可以按 head 区分。GQA/MQA 的多个 Q head 会共享 K/V，见下一节。

### 两头的具体矩阵例子

令输入维度为 4，head 数为 2，每个 head 的维度为 2。数学上的 $W_Q$ 形状是 $4\times4$，可以按列分成两个 $4\times2$ 的块：

$$
X=\begin{bmatrix}1&2&3&4\end{bmatrix},\qquad
W_Q=\begin{bmatrix}
1&0&1&0\\
0&1&1&0\\
1&0&0&1\\
0&1&0&1
\end{bmatrix}
=\begin{bmatrix}W_Q^{(0)}&W_Q^{(1)}\end{bmatrix}
$$

$$
W_Q^{(0)}=\begin{bmatrix}1&0\\0&1\\1&0\\0&1\end{bmatrix},\qquad
W_Q^{(1)}=\begin{bmatrix}1&0\\1&0\\0&1\\0&1\end{bmatrix}
$$

完整计算为：

$$
Q=XW_Q=\begin{bmatrix}4&6&3&7\end{bmatrix}
$$

TP=2 时可以直接这样算：

| GPU | 本地输入 | 本地 Q 投影权重 | 直接算出的 Q |
| --- | --- | --- | --- |
| 0 | 完整 $X=[1,2,3,4]$ | $W_Q^{(0)}$，形状 $4\times2$ | $Q^{(0)}=[4,6]$ |
| 1 | 同一个完整 $X$ | $W_Q^{(1)}$，形状 $4\times2$ | $Q^{(1)}=[3,7]$ |

K 和 V 使用各自的投影矩阵，按同样的 head 划分方式处理。这就是“加载权重时已分片，各卡直接生成自己的 Q/K/V”：GPU 0 从一开始就只持有自己需要的投影参数，而不是先算出四个 Q 特征再丢掉两个。

输入 $X$ 怎样到达每张 GPU，取决于上一层的数据布局。普通 TP 中它可能已经复制，也可能需要从上游分片收集。本段只说明 Q/K/V 投影本身，不能据此认为整个 attention 前后没有通信。

rank 0 不需要 rank 1 的 head 就能计算自己的 attention。每个本地 head 都拿到自己所需的全部可见 key，softmax 分母也完整，所以**不需要跨 GPU 的 softmax 修正**。如果使用 FlashAttention，本地 kernel 内仍可能进行分块归一化；那是另一层次的计算组织。

Attention 输出还需要通过输出投影。按 head 拼接的数学输出可以直接接 row-parallel 投影：

$$
Y=\begin{bmatrix}O_0&O_1\end{bmatrix}
\begin{bmatrix}W_{O,0}\\W_{O,1}\end{bmatrix}
=O_0W_{O,0}+O_1W_{O,1}
$$

因此通常各卡先算自己的输出投影贡献，再 AllReduce，或由外层管理器按布局合并。**这里相加的是输出投影的部分结果，不是把不同 head 的 softmax 概率相加。**

### 对照 TokenSpeed 的普通 Llama 路径

[linear.py](https://github.com/lightseekorg/tokenspeed/blob/8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613/python/tokenspeed/runtime/layers/linear.py) 中的 `QKVParallelLinear` 继承 `ColumnParallelLinear`，类注释明确说按 head 维度并行。它使用 `gather_output=False`。

[llama.py](https://github.com/lightseekorg/tokenspeed/blob/8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613/python/tokenspeed/runtime/models/llama.py) 的 `LlamaAttention.forward` 主要执行：

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
attn_output = self._attn(positions, q, k, v, ctx)
output, _ = self.o_proj(attn_output)
```

其中 `qkv.split(...)` 是**在本卡把融合投影的结果拆成 Q、K、V 三段**，它本身没有跨卡发送数据。“TP head 划分已体现在权重和本地 head 数量里”可以具体展开成：

1. 建层时计算本地 head 数，例如 `num_heads = total_num_heads / tp_size`；两头、TP=2 时每卡一个 head。
2. 加载器按当前 rank 分别选取 Q/K/V 对应 head 的权重。PyTorch 的权重存为转置布局，数学上的输出列通常对应实际权重的输出行。
3. 本例各卡逻辑上的融合权重为 $[W_Q^{(r)},W_K^{(r)},W_V^{(r)}]$，形状 $4\times6$；实际存储形状为 $6\times4$。
4. 对 $T$ 个 token，本地投影输出形状为 $T\times6$，拆成三个 $T\times2$ 的张量后，各自可解释为 $T\times1\times2$：本卡一个 head，每个 head 两维。

所以 `split` 是识别本地结果中的 Q/K/V 段；负责哪个全局 head 已由权重选择确定。融合权重加载时会分别处理 q/k/v 的位置与 head 分片，不能简单把全局融合矩阵不分段地从中间切成两半。

`o_proj` 是 row-parallel；该模型把必要的结果归约交给外层通信管理器，具体衔接见第四课。

## 3 GQA 和 MQA 如何切 K V

MHA 通常每个 Q head 对应自己的 KV head。GQA 中一组 Q head 共享一个 KV head；MQA 则所有 Q head 共享一个 KV head。因此不能假定 Q、K、V 总是具有相同的 head 数，或者全部严格均分。

例如：8 个 Q head、2 个 KV head、TP=4，采用连续的标准 GQA 分组：

| Rank | Q head | 本地需要的 KV head |
| --- | --- | --- |
| 0 | 0、1 | KV 0 |
| 1 | 2、3 | KV 0 的副本 |
| 2 | 4、5 | KV 1 |
| 3 | 6、7 | KV 1 的副本 |

Q head 仍分片；不足以分给每个 rank 一个不同 KV head 时，KV 投影与相应 cache 会复制给需要它们的 rank。共享或复制的是 K/V，各 Q head 仍分别计算自己的 softmax。

`QKVParallelLinear.__init__` 中对应逻辑是：

```python
if tp_size >= self.total_num_kv_heads:
    self.num_kv_heads = 1
    self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
else:
    self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
    self.num_kv_head_replicas = 1
```

[models/utils.py](https://github.com/lightseekorg/tokenspeed/blob/8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613/python/tokenspeed/runtime/models/utils.py) 的 `validate_attention_partition` 会检查 Q head 与 TP、KV head 与 TP 的整除关系。表中例子是布局解释，没有新增四卡实测。

## 4 历史 K V 按 token 切开时需要合并归一化

长上下文下，可能让多卡分别保存同一个 head 的不同历史位置：

```text
                         同一个 query q
                         /           \
rank 0：历史 key 集合 S0 的 K/V       rank 1：历史 key 集合 S1 的 K/V
        ↓                                  ↓
      O0、LSE0                           O1、LSE1
                         \           /
                       合并 → 完整 attention 输出
```

每张卡必须得到相应 query；具体实现可以交换 Q，也可以通过其他布局安排让 query 已在本地。历史分片也不必是连续前后两段，TokenSpeed 的 KVP 可以按页交错分配。

现在两张卡处理的是**同一个 query、同一个 head 的不同 key**。各卡自己的 softmax 分母只包含本地 key，所以不能直接平均局部输出。

固定一个 query 和一个 head，把下标 $t,h$ 省略。分数 $s_j$ 已包含缩放及适用的 mask。假设至少有一个有效 key，将有效 key 分成互不重叠的集合 $S_0,S_1$。先分别定义分子和分母：

$$
N_r=\sum_{j\in S_r}\exp(s_j)v_j,\qquad
Z_r=\sum_{j\in S_r}\exp(s_j)
$$

$Z_r$ 是标量，$N_r$ 一般是向量；局部输出 $O_r=N_r/Z_r$。完整 attention 的推导是：

$$
\begin{aligned}
O
&=\frac{\sum_{j\in S_0\cup S_1}\exp(s_j)v_j}
        {\sum_{j\in S_0\cup S_1}\exp(s_j)}\\
&=\frac{N_0+N_1}{Z_0+Z_1}\\
&=\frac{Z_0O_0+Z_1O_1}{Z_0+Z_1}.
\end{aligned}
$$

第三步只是把 $N_r=Z_rO_r$ 代回去。对任意数量的非空分片，同样定义：

$$
Z_r=\sum_{j\in S_r}\exp(s_j),
\qquad
O_r=\frac{\sum_{j\in S_r}\exp(s_j)v_j}{Z_r}
$$

如果各 $S_r$ 互不重复、合起来覆盖所需的全部 key，那么：

$$
O=\frac{\sum_r Z_rO_r}{\sum_r Z_r}
$$

这就是恢复完整 softmax 的关键：**局部输出按它对应的归一化分母加权合并。**

## 5 为什么实现中传 LSE

直接计算很大的指数容易溢出，因此通常保存：

$$
L_r=\operatorname{LSE}_r=\log Z_r
$$

令 $m=\max_r L_r$，合并权重和结果为：

$$
\alpha_r=\frac{\exp(L_r-m)}{\sum_u\exp(L_u-m)},
\qquad
O=\sum_r\alpha_rO_r
$$

推导方法是将上一节的 $Z_r$ 换成 $\exp(L_r)$，再把分子、分母同时除以 $\exp(m)$。这是同一个分式，不会改变结果。

这里的 $m$ 是**分片 LSE 的最大值**。每个分片内部计算 attention 时，还会针对本地 score 使用自己的稳定化最大值。下面分别记为 $m_r$，避免把两种最大值混为一谈。

这个合并可以看成“在分片的 LSE 上再做一次 softmax，得到每个分片的权重”。它代数上等价于在全部 key 上做一次 softmax，不是凭经验修正，也不是按每卡 token 数平均。FlashInfer 的 [Attention States and Recursive Attention](https://docs.flashinfer.ai/tutorials/recursive_attention.html) 用局部输出与 LSE 描述可合并的 attention 状态。

### 与 Online Softmax 的关系

Online softmax 在扫描分数时维护当前最大值和相对于它的指数和。最大值变化时，需要把已有的指数和换算到新的基准；这就是重标定因子的来源。[Online normalizer calculation for softmax](https://arxiv.org/html/1805.02867v1)

对一个 key 分块 $S_r$，保存以下状态：

$$
m_r=\max_{j\in S_r}s_j,\qquad
\ell_r=\sum_{j\in S_r}\exp(s_j-m_r),\qquad
u_r=\sum_{j\in S_r}\exp(s_j-m_r)v_j
$$

$m_r,\ell_r$ 是标量；$u_r$ 是尚未归一化的加权 value 向量。两种状态表示之间的关系为：

$$
O_r=\frac{u_r}{\ell_r},\qquad
L_r=m_r+\log\ell_r
$$

合并两个非空块时，选共同基准 $m'=\max(m_0,m_1)$。因为：

$$
\exp(s_j-m')=\exp(s_j-m_r)\exp(m_r-m'),\qquad j\in S_r
$$

所以令 $a=\exp(m_0-m')$、$b=\exp(m_1-m')$，就得到：

$$
\boxed{
\ell'=a\ell_0+b\ell_1,\qquad
u'=au_0+bu_1,\qquad
O'=\frac{u'}{\ell'}
}
$$

当 block 0 代表“已经处理的全部 key”、block 1 代表“刚读到的新块”时，这就是逐块 online attention 的更新。如果新块只有一个元素 $s,v$，它的状态是 $(s,1,v)$，因而：

$$
\begin{aligned}
m'&=\max(m,s),\\
\ell'&=e^{m-m'}\ell+e^{s-m'},\\
u'&=e^{m-m'}u+e^{s-m'}v.
\end{aligned}
$$

原始 online softmax 的归一化器对应前两项；attention 额外维护 $u$，让最终结果不必通过物化全部 softmax 概率再乘 V 获得。[FlashAttention §3.1](https://arxiv.org/html/2205.14135v2#S3.SS1) 使用分块计算及归一化重标定来完成 attention。

如果实现保存的是已归一化的 $O$，而不是 $u$，更新式必须写为：

$$
O'=\frac{a\ell_0O_0+b\ell_1O_1}{a\ell_0+b\ell_1}
$$

不能遗漏 $\ell_0,\ell_1$，只用 $a,b$ 去平均两个 $O$。局部最大值只说明分数的尺度，不能替代整个分片的指数和。

分布式合并与 online 更新的对应关系如下：

| 执行方式 | 分块结果在哪里产生 | 如何合并 |
| --- | --- | --- |
| 单个计算任务逐块扫描 | 旧状态加上新读到的 key 块 | 每次更新 $(m,\ell,u)$ |
| 多 GPU 各算历史的一部分 | 各 GPU 得到局部状态 | 通信后合并 $(m,\ell,u)$ 或 $(L,O)$ |
| KV 块在设备间轮流传递 | 每个 query owner 陆续看到不同 KV 块 | 本地按块更新 online 状态 |

从代数上看，都是合并不相交 key 集合的统计量；实际通信方式、内存访问和调度不同。精确算术中可以改变分组和合并顺序，浮点实现仍可能有舍入差异。Online 指沿输入逐步更新，不等同于“大模型在线服务”，也不要求必须跨 GPU。

以上公式按非空、具有有效分数的状态推导。空状态通常以 $\ell=0,u=0$ 表示，首次更新及全 mask 情况需要单独处理，避免直接计算负无穷减负无穷。

### TokenSpeed 的代码在哪里

[dcp/comm.py](https://github.com/lightseekorg/tokenspeed/blob/8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613/python/tokenspeed/runtime/layers/attention/dcp/comm.py) 的 `combine_attention_partials` 接收：

```python
local_output  # 局部历史产生的 attention 输出
local_lse     # 每个 query/head 的自然对数 LSE，FP32
```

根据路径，它会把局部输出和 LSE 发给对应 head 的 owner 再合并，或收集 LSE、加权本地输出后使用 AllReduce/ReduceScatter。

在 [DCP 的 Triton 实现](https://github.com/lightseekorg/tokenspeed/blob/8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv4/_triton/dcp.py) 的 `_dcp_weight_kernel` 和 `_combine_dcp_attention_partials` 中，可以直接找到减最大值、指数权重和 FP32 加权累加。实现还处理空分片等边界情况。

这说明“用哪个 collective”与“怎样恢复数学结果”是两个问题：AllReduce 本身只会合并数据，softmax 的正确权重需要在相应计算中明确生成。

## 6 一个可手算的反例

设四个 key 的指数分数和标量 value 为：

$$
\exp(s)=\begin{bmatrix}1&3&2&4\end{bmatrix},
\qquad
V=\begin{bmatrix}10&20&30&40\end{bmatrix}
$$

前两个 key 放分片 0，后两个放分片 1：

$$
\begin{aligned}
Z_0&=1+3=4,& O_0&=\frac{1\times10+3\times20}{4}=17.5,\\
Z_1&=2+4=6,& O_1&=\frac{2\times30+4\times40}{6}\approx36.6667.
\end{aligned}
$$

直接平均得到约 $27.0833$，这是错误结果。正确合并是：

$$
O=\frac{4O_0+6O_1}{10}=29
$$

完整计算也得到：

$$
O=\frac{1\times10+3\times20+2\times30+4\times40}{1+3+2+4}=29
$$

运行 [06_softmax_merge.py](06_softmax_merge.py)：

```bash
cd /home/hadoop-scale-llm/dasc
python3 tokenspeed_learning/06_softmax_merge.py
```

2026 年 11 月 24 日的 CPU 公式演示输出，原始记录见 [日志](logs/06_softmax_merge.log)：

```text
CPU formula demonstration; no GPU or distributed communication
shard 0: local_output=17.500000, LSE=1.386294, merge_weight=0.400000
shard 1: local_output=36.666667, LSE=1.791759, merge_weight=0.600000
naive average (wrong): 27.083333
LSE-weighted merge: 29.000000
full attention:    29.000000
PASS: LSE-weighted merge equals full attention within 1e-12
```

这验证了示例的合并公式，不等同于验证 TokenSpeed 的 GPU DCP kernel。

## 7 按 query 切分是否也要 LSE 合并

### 按 token 切分叫 CP 还是 SP

需要结合框架的定义。把**同一个请求的长序列**沿 token 维度拆到多卡，是广义的序列维度并行；不同资料可能称为 context parallelism（CP）或 sequence parallelism（SP）。拆不同请求通常属于 DP，不能只因为数据里有 token 就叫 CP。

在 Megatron 的术语中，CP 覆盖输入和各层激活的序列维度切分，并处理 attention 跨分片读取 KV 的需求；传统 SP 主要与 TP 配合，切分 LayerNorm、Dropout 等区域的激活，减少重复存储，并不自动意味着 attention 的历史 KV 在 softmax 时分布于不同卡。[Megatron CP 说明](https://docs.nvidia.com/megatron-core/developer-guide/0.17.0/user-guide/features/context_parallel.html)

所以判断是否需要合并 softmax，应检查真实数据流：每个 query 是否在一次完整的计算中看到全部有效 key，还是产生多个局部归一化结果。CP/SP 的名字本身不足以决定。

### 三个场景对照：LSE 合并、query 分片与 online softmax

**不能把区别理解成“LSE 用于多卡，online softmax 用于单卡”。** 判断 LSE 合并是否必要，要固定同一个 query、同一个 head，检查它需要关注的 key 是否分散到了多个局部计算中。

**场景一：同一个 query，分别计算部分 KV。** 假设一个 query 可以关注 1,000 个历史 token，两卡各保存一半：

```text
                       同一个 query q、同一个 head
                         /                   \
GPU 0：KV token 0–499                           GPU 1：KV token 500–999
          ↓                                              ↓
     局部 O₀、LSE₀                                  局部 O₁、LSE₁
                         \                   /
                            LSE 加权合并
                                  ↓
                       q 的完整 attention 输出
```

每张 GPU 只看到部分 KV，其局部 softmax 分母不完整。因此需要按第五节的 LSE 权重合并，不能直接平均局部输出。这里合并的是**同一输出行的不同贡献**。实际也可以同时处理一批 query，但合并关系仍逐 query、逐 head 成立。

**场景二：不同 query，各自看到完整的可见 KV。** 如果 rank 0 算 query 行 0–99，rank 1 算 query 行 100–199，并且每一行都能访问完整的可见 K/V：

```text
GPU 0：Q token 0–99    × 完整可见 KV → attention 输出行 0–99
GPU 1：Q token 100–199 × 完整可见 KV → attention 输出行 100–199
```

两边算的是不同的 softmax 行，每一行的分母都完整，**不需要跨卡 LSE 合并**。按位置组织输出行即可，也不一定需要立即把所有行收集到同一张卡。这里的“完整可见”仍遵守 causal mask 等约束，不代表允许读取未来 token。

**场景三：固定 query 块，逐块读取 KV，使用 online softmax。** 在单卡 FlashAttention 的典型计算中，固定一个 Q 块，依次计算不同 KV 块对应的 score 子矩阵：

$$
S_b=\frac{Q_{\mathrm{block}}K_b^{\mathsf T}}{\sqrt d}+M_b
$$

```text
读取 KV 块 0 → 算 S₀ → 初始化每个 query 的最大值、指数和、加权 V 和
读取 KV 块 1 → 算 S₁ → 重标定旧状态，再加入新块的贡献
读取 KV 块 2 → 算 S₂ → 继续更新
……
处理全部可见 KV → 得到该 Q 块的完整 attention 输出
```

这就是“分块计算 S”的含义：不必存下完整的 score 矩阵或 softmax 概率矩阵，而是逐块维护第五节的 $(m,\ell,u)$。仅把 Q 分成不同的行，还不足以说明使用了 online softmax；关键在于同一行沿 key 维度逐块更新归一化状态。

Online softmax 也可用于多卡：例如 KV 块在设备间轮流传递，query 所在的卡每收到一个新块，就更新一次本地累计状态。

| 方式 | 核心过程 | 是否限定单卡或多卡 |
| --- | --- | --- |
| LSE 合并局部输出 | 各 KV 块分别得到 $O_r,L_r$，再合并同一 query/head 的贡献 | 不限定；也可合并单卡内不同计算任务的部分结果 |
| Online softmax | 保存累计状态，新 KV 块到来时立即更新 | 不限定；KV 块可以来自本卡存储，也可以来自其他卡 |

**两者可以同时使用：各 GPU 内部用 online softmax 算自己的局部 attention，再在 GPU 之间用 LSE 合并局部结果。** 它们使用相同的归一化重标定原理，区别在于分块结果如何产生和何时合并。

TokenSpeed 本地 [parallelism.md](https://github.com/lightseekorg/tokenspeed/blob/8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613/docs/serving/parallelism.md) 区分三种术语：KVP 是 KV 历史页的存储分片；DCP 是 decode 时对这些历史分片计算并合并局部结果；QCP 是 prefill 的 query 行分片，其描述的路径访问收集后的完整历史，因此不做跨分片 LSE 合并。后端与模型的适用范围不同，不应把某一种配置视为所有模型的通用路径。

| 切分对象 | 本地计算缺少什么 | 如何得到正确结果 |
| --- | --- | --- |
| 不同 head，且每个本地 head 有完整可见 KV | 不缺本 head 的 softmax 分母 | 独立 attention；随后正确处理输出投影 |
| 不同 query 行，且有完整可见 KV | 不缺该 query 行的 softmax 分母 | 独立计算行；后续按需要重排输出 |
| 同一 query/head 的不同 KV token | 缺其他分片的归一化贡献 | 合并局部输出与 LSE，或等价地合并稳定化分子/分母 |
| 同一个 head 内的点积维度 $d$ | 本地 score 只是点积的部分和 | 先求和得到完整 score，再做 softmax |

最后一种情况下 $QK^{\mathsf T}=\sum_r Q_rK_r^{\mathsf T}$。不能先对每个点积的部分和做 softmax，再用前述 LSE 公式拼接；前述公式要求分的是不同 key 集合。

## 8 如何确保计算结果正确

正确性首先来自上述代数分解，同时实现必须满足对应条件：Q head 与 KV head 对应关系正确，分片覆盖正确，RoPE 使用正确的全局位置，causal mask 不能把未来 token 纳入，同一轮读取的 KV 必须与请求和历史一致。

对于 KV 分片合并，每个有效 key 应恰好计入一次；LSE 的对数底、softmax scale 和 score 处理必须一致。某一分片没有有效 key 时，其归一化质量应为零，需要避免负无穷相减导致 NaN。滑动窗口、attention sink 等机制也要按各自语义处理。

数学等价不意味着不同 GPU 数量下逐 bit 相同。浮点求和顺序、低精度、kernel 选择会带来数值差异；实际验证通常把相同输入下的输出与完整参考实现按适当误差容限比较。

单机或多机主要改变通信通路，不改变这里的分解公式。本课讨论的是 **softmax attention**；Kimi-Linear 中 KDA 等线性注意力使用递归状态，不能直接照搬这一套 softmax 分母合并解释。
