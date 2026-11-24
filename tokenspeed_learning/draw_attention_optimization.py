#!/usr/bin/env python3
"""Chapter 8 vector diagrams, with matching PNG previews."""
from draw_moe_parallel import Canvas, INK, MUTED, BLUE, TEAL, PURPLE, ORANGE

GOLD, GREEN, RED = '#FFF0CC', '#E6F4DC', '#FBE1E1'


def node(c,x,y,w,h,title,detail='',fill='#FFFFFF',color=INK):
    c.box(x,y,w,h,fill,color)
    c.text(x+w/2,y+(h/2 if not detail else h*.30),title,27,color,True)
    if detail:c.text(x+w/2,y+h*.75,detail,23,MUTED,True)


def architecture():
    c=Canvas(1760,'DeepSeek V4.1 Flash：Attention 的算子与维度')
    c.text(50,112,'单层文本主干 · T=2048 · H=5120 · 64 heads × 512 · 图中先不切 TP',27,MUTED)
    node(c,510,155,580,80,'mHC 交给 Attention 的输入 X','[T,H] = [2048,5120]')
    c.arrow(800,240,800,273)
    node(c,510,280,580,75,'Attention 输入 RMSNorm','逐 token；完整 H 维', '#F1F5F9')
    for x in (310,800,1290):c.arrow(800,360,x,410)
    node(c,70,420,480,95,'① Q 下投影  wq_a','W: [5120,1280]',GOLD)
    node(c,570,420,460,95,'② 本层 SWA KV  wkv','W: [5120,512]',GOLD)
    node(c,1060,420,470,95,'③ Global KV 来源','仅 source layers 产生；其他层复用',GOLD)
    c.text(310,550,'[2048,1280]',27,BLUE,True)
    c.text(800,550,'[2048,512]',27,TEAL,True)
    c.arrow(310,572,310,605);c.arrow(800,572,800,605);c.arrow(1290,522,1290,605)
    node(c,70,615,480,82,'q_norm','RMSNorm over 1280', '#F1F5F9')
    node(c,570,615,460,82,'kv_norm → RoPE','共享 KV 向量；末 64 维旋转','#F1F5F9')
    node(c,1060,615,470,82,'FlatKV / Compressor','保持请求、原始位置和跨层来源','#F1F5F9')
    c.arrow(310,705,310,755);c.arrow(800,705,800,755);c.arrow(1290,705,1290,755)
    node(c,70,765,480,95,'④ Q 上投影  wq_b → RoPE','W: [1280,32768]',GREEN)
    node(c,570,765,460,95,'SWA cache / 当前 KV','窗口 128；K/V 共用存储表示',GREEN)
    node(c,1060,765,470,95,'Indexer / 复用 Top-k','选择 global positions；最多 512',GREEN)
    c.text(310,900,'Q [2048,64,512]',27,BLUE,True)
    c.text(800,900,'KV [可见行数,512]',26,TEAL,True)
    c.text(1290,900,'global KV + 选择索引',26,PURPLE,True)
    for x in (310,800,1290):c.arrow(x,930,x,1005)
    node(c,70,1015,1460,100,'⑤ Sparse Attention：每个 query 只读本请求允许的位置','SWA + 被选中的 global KV；causal mask + attention sink',RED)
    c.arrow(800,1123,800,1170)
    node(c,300,1180,1000,90,'O [2048,64,512] → inverse RoPE → 按 8 组整理','每组 8 heads × 512 = 4096；得到 [2048,8,4096]')
    c.arrow(800,1278,800,1310)
    node(c,300,1320,1000,95,'⑥ 分组低秩 wo_a：各组 4096 → 1024','W: [8,4096,1024]；输出 [2048,8,1024]','#EDE6F8',PURPLE)
    c.arrow(800,1423,800,1455)
    node(c,300,1465,1000,95,'⑦ Flatten → wo_b：8192 → 5120','W: [8192,5120]；输出 [2048,5120]','#EDE6F8',PURPLE)
    c.text(800,1610,'返回 mHC / 后续 MoE；并行时 wo_b 的局部贡献还要求和',28,INK,True)
    c.text(50,1670,'权重按数学 XW 方向标注，PyTorch 存储通常转置。global 分支仅示意依赖，非完整 CED 图。',23,MUTED)
    c.text(50,1715,'KV 选择还依赖 X、q latent；图中省略这两条侧向线。量化尺度、Engram、视觉及 DSpark 未展开。',23,MUTED)
    c.save('08_v41_attention_flow')


def gather():
    c=Canvas(1310,'优化一：把 AllGather 移到逐 token 小投影之后')
    c.text(50,110,'两卡 SP2 示意 · T=2048 · 两边使用相同权重 · 起点已经完成 SUM / ReduceScatter',26,MUTED)
    for left,title,col in [(50,'原始：先收齐，再重复算',BLUE),(825,'优化：先各算一半，再收齐',TEAL)]:
        c.box(left,160,725,965,'#FFFFFF',col)
        c.text(left+362,203,title,30,col,True)
        for r in range(2):
            x=left+22+r*350
            c.text(x+165,260,f'GPU {r}',29,col,True)
            node(c,x,295,330,80,f'X{r} [1024,5120]','1024 个不重复 token')
        c.text(left+362,410,'GPU 0 / GPU 1 是同一个 TP 组',24,MUTED,True)
    for r in range(2):
        x=72+r*350
        node(c,x,520,330,80,'X [2048,5120]','完整 2048 行')
        c.arrow(x+165,608,x+165,665)
        node(c,x,675,330,110,'Norm → Q down','每卡算 2048 行',GOLD)
        c.arrow(x+165,793,x+165,860)
        node(c,x,870,330,100,'q latent + SWA KV','[2048,1280+512]',GREEN)
        x=847+r*350
        node(c,x,520,330,110,'Norm → Q down','每卡只算 1024 行',GOLD)
        c.arrow(x+165,638,x+165,680)
        node(c,x,690,330,100,'q latent + SWA KV','[1024,1280+512]',GREEN)
        node(c,x,900,330,80,'拼成完整小张量','[2048,1280+512]')
    c.arrow(237,475,587,475,BLUE);c.arrow(587,490,237,490,BLUE)
    c.text(412,451,'AllGather 5120 维',25,BLUE,True)
    c.arrow(1012,851,1362,851,TEAL);c.arrow(1362,866,1012,866,TEAL)
    c.text(1187,822,'AllGather 1792 维',25,TEAL,True)
    for x in (412,1187):
        c.arrow(x,985,x,1020)
        c.text(x,1060,'本卡 Q heads 投影 → Attention',26,INK,True)
    c.text(50,1170,'同样是完整 token 行，逐行算子与拼接可交换；Norm 不能放到 SUM 之前。',28,INK)
    c.text(50,1220,'1792 = Q latent 1280 + 本层 SWA KV 512；这里不包含 global compressor / indexer 的通信。',24,MUTED)
    c.text(50,1265,'每卡算术量减少，不保证延迟同比例下降；还要计小 GEMM、通信启动和缓冲开销。',24,MUTED)
    c.save('08_move_allgather')


def balance():
    c=Canvas(1500,'优化二：跨 DP 均衡线性层，Attention 仍回到请求 owner')
    c.text(50,110,'4 GPU · Attention DP2 × TP2 · 请求 A=3072 tokens，B=1024 tokens · 教学重排方案',26,MUTED)
    xs=[235,610,985,1360]
    for left,txt in [(60,'DP0 / TP组 (0,1)：请求 A'),(820,'DP1 / TP组 (2,3)：请求 B')]:
        c.box(left,155,720,245,'#EAF1FF',BLUE);c.text(left+360,195,txt,27,BLUE,True)
    for r,x in enumerate(xs):
        c.text(x,250,f'GPU {r}',29,BLUE,True)
        c.text(x,303,f'{"A" if r<2 else "B"} 全行 × 本地 32 heads',23,INK,True)
        c.text(x,355,'原始线性层：'+('3072' if r<2 else '1024')+' 行',24,MUTED,True)
    c.text(800,447,'① 对唯一 token 做跨 DP 重分配，逐 token 计算均衡',29,TEAL,True)
    for r,x in enumerate(xs):
        c.arrow(x,471,x,485,TEAL)
        node(c,x-165,495,330,120,f'{"A"+str(r) if r<3 else "B"}：1024 行','Norm / Q down / 本地 KV',GOLD)
    c.text(800,655,'② AllGather 小 latent → 按 request_id 选回 A 或 B（不混合请求语义）',28,BLUE,True)
    for r,x in enumerate(xs):
        c.arrow(x,680,x,705,BLUE)
        node(c,x-165,715,330,135,('A [3072,32,512]' if r<2 else 'B [1024,32,512]'),'Q up → 本地 Attention',RED)
    c.text(800,894,'此处 KV 仍在原 DP；A 的 Attention 依旧可能更慢',29,ORANGE,True)
    c.text(800,944,'③ 按 (request_id, token_id, head_id) 做 AllToAll，拼成完整 heads',27,PURPLE,True)
    for r,x in enumerate(xs):
        c.arrow(x,992,x,1010,PURPLE)
        node(c,x-165,1020,330,120,f'{"A"+str(r) if r<3 else "B"} [1024,64,512]','新 owner：完整 heads',GREEN)
        c.arrow(x,1148,x,1185,TEAL)
        node(c,x-165,1195,330,100,'完整 wo_a + wo_b','SP：1024 行 → [1024,5120]','#EDE6F8',PURPLE)
    c.arrow(xs[0],470,xs[3],470,TEAL)
    c.arrow(xs[3],680,xs[0],680,BLUE)
    c.arrow(xs[0],980,xs[2],980,PURPLE)
    c.arrow(xs[1],992,xs[2],992,PURPLE)
    c.text(800,1340,'紫色横箭头示例：GPU0/1 的 A2 两份 head 拼到 GPU2；其余传输路径省略。',26,INK,True)
    c.text(50,1400,'A0/A1/A2 各 1024 行；最终 inverse permutation 恢复请求顺序。残差、位置和 mask 同步重排。',24,MUTED)
    c.text(50,1452,'此图不是 TokenSpeed V4.1 的现成配置；源码中的 model-wide QCP 是另一条完整路径。',24,MUTED)
    c.save('08_dp_linear_balance')


def context():
    c=Canvas(1390,'优化三：QCP 切 query 工作；KVP + DCP 切历史 KV')
    c.text(50,110,'固定同一请求、相同 query head；左侧按 query 行分工，右侧对同一行合并 KV 分片结果',26,MUTED)
    for left,title,col in [(50,'QCP：不同 Q，完整可见 KV',BLUE),(825,'DCP：相同 Q，互斥 KV 分片',PURPLE)]:
        c.box(left,160,725,660,'#FFFFFF',col);c.text(left+362,205,title,28,col,True)
        for r in range(2):
            x=left+22+r*350
            c.text(x+165,265,f'GPU {r}',30,col,True)
            node(c,x,310,330,90,((('Q 位置 0,1,6,7' if r==0 else 'Q 位置 2,3,4,5')) if left==50 else '相同的 Q：位置 7'),'每行带原始 position')
            c.arrow(x+165,407,x+165,455,col)
            node(c,x,465,330,100,('本请求全部可见 KV' if left==50 else f'本地 KV pages {r}'),'causal / sparse mask',GREEN)
            c.arrow(x+165,573,x+165,615,col)
            node(c,x,625,330,95,('本地 Q 行的完整 O' if left==50 else '局部 O_r + LSE_r'),('每行的完整归一化域' if left==50 else '仅本地 KV 归一化'),RED)
        c.text(left+362,772,('按 token ID 放回；无需 LSE 合并' if left==50 else '按 LSE 权重合并同一个 Q 的结果'),25,col,True)
    c.box(50,860,1500,320,'#FFFFFF',TEAL)
    c.text(800,905,'KVP4 示例：一个请求的 8 个逻辑 page，page p 属于 GPU (p mod 4)',28,TEAL,True)
    for r in range(4):
        x=65+370*r
        node(c,x,955,350,100,f'GPU {r}：page {r}, {r+4}','自己的本地物理 slots')
    c.text(800,1100,'查询表保存 owner + local slot；请求、逻辑位置和 prefix 身份保持一致',28,INK,True)
    c.text(800,1145,'V4.1：global KV / index-K 可分片；SWA、compressor tails 等仍有复制部分',25,MUTED,True)
    c.text(50,1225,'DCP 的同一贡献只计一次：复制的 SWA 指定一个 owner，attention sink 也只能计一次。',26,INK)
    c.text(50,1280,'分布式 Top-k 仍要选全局的目标集合；不能每卡随意少选一些，然后指望 LSE 恢复遗漏。',25,INK)
    c.text(50,1335,'CP 会增加数据交换；更大 KV 容量与更低 decode 延迟是两个不同目标。',25,MUTED)
    c.save('08_qcp_dcp_pages')


def walkthrough():
    c=Canvas(1690,'走一遍：4 个 token，在 2 张 GPU 上完成一次 Attention')
    c.text(50,111,'V4.1 维度 · 一个请求 · 无历史前缀 · 选纯 SWA 层 · 教学 SP → head TP → SP 布局',25,MUTED)
    c.text(385,182,'GPU 0：负责 heads 0–31',30,BLUE,True)
    c.text(1215,182,'GPU 1：负责 heads 32–63',30,TEAL,True)
    for r,x in enumerate((100,930)):
        col=BLUE if r==0 else TEAL
        pair='t0、t1' if r==0 else 't2、t3'
        node(c,x,225,570,100,'① 已归约的完整 token 行',pair+'：X [2,5120]','#FFFFFF',col)
        c.arrow(x+285,333,x+285,375,col)
        node(c,x,385,570,130,'② 本地 Norm + Q down + SWA KV','Cq [2,1280]；KV [2,512]',GOLD,col)
        node(c,x,650,570,115,'③ 收齐 4 个 latent，尚未生成 Q heads','Cq [4,1280]；KV [4,512]','#FFFFFF',col)
        c.arrow(x+285,773,x+285,825,col)
        node(c,x,835,570,140,'④ Q up → 本地 heads 的 Attention','Q / O [4,32,512]；读本请求可见 KV',RED,col)
        c.arrow(x+285,983,x+285,1030,col)
        node(c,x,1040,570,150,'⑤ 4 组 wo_a → 本地 wo_b','[4,4096] → P'+str(r)+' [4,5120] 部分和','#EDE6F8',col)
        node(c,x,1340,570,115,'⑥ ReduceScatter 后的完整输出',pair+'：Y [2,5120]','#FFFFFF',col)
    c.text(800,558,'AllGather：每卡把自己的 2 行小 latent 发给对方',28,INK,True)
    c.arrow(385,590,1215,590,BLUE)
    c.arrow(1215,612,385,612,TEAL)
    for x in (385,1215):c.arrow(x,620,x,644)
    c.text(800,1230,'ReduceScatter：相同 token 的 P0 + P1，再按 token 分给 owner',27,INK,True)
    c.arrow(385,1265,1215,1265,BLUE)
    c.arrow(1215,1287,385,1287,TEAL)
    for x in (385,1215):c.arrow(x,1300,x,1334)
    c.text(800,1510,'追踪 t3：GPU1 做小投影 → 两卡各算一半 heads → 两份输出贡献汇到 GPU1',27,INK,True)
    c.text(50,1575,'①② 按 token 分工（SP）；④⑤ 按 heads / 权重分工（TP）；⑥ 回到按 token 分工。',26,MUTED)
    c.text(50,1625,'两卡看到完整可见 KV，所以这里没有跨卡 LSE 合并。不是多两张卡，也不是切开 hidden 特征做 Norm。',23,MUTED)
    c.save('08_four_token_walkthrough')


def output_projections():
    c=Canvas(1480,'wo_a 与 wo_b：把各 head 的输出变回 hidden')
    c.text(50,110,'只追踪一个 token t3 · TP2 · o 表示 output，a/b 表示先后两段；两卡都执行两段',25,MUTED)
    for r,x in enumerate((100,930)):
        col=BLUE if r==0 else TEAL
        c.text(x+285,180,f'GPU {r}：heads {r*32}–{r*32+31}',30,col,True)
        node(c,x,225,570,100,'Attention 输出 → inverse RoPE','本 token 的 32 heads：[32,512]',RED,col)
        c.arrow(x+285,333,x+285,380,col)
        node(c,x,390,570,105,f'按组整理：groups {r*4}–{r*4+3}','4 组，每组 8 heads：[4,4096]','#FFFFFF',col)
        c.arrow(x+285,503,x+285,550,col)
        node(c,x,560,570,130,'wo_a：每组独立 4096 → 1024','4 个不同的组权重；输出 [4,1024]',GOLD,col)
        c.arrow(x+285,698,x+285,745,col)
        node(c,x,755,570,105,'拼接本卡 4 组结果','Z'+str(r)+'：[4096]；同一个 token 的特征',GREEN,col)
        c.arrow(x+285,868,x+285,915,col)
        node(c,x,925,570,130,'wo_b：本卡权重分片 [4096,5120]','P'+str(r)+'：[5120]；仅本卡 groups 的贡献','#EDE6F8',col)
    c.text(800,1100,'完整 Wob = [8192,5120]，按输入维分片；不是每卡各有完整 Wob',26,INK,True)
    c.arrow(385,1145,1215,1145,BLUE)
    c.text(800,1188,'ReduceScatter 中，t3 的 P0 发给 GPU1，与 P1 求和',27,INK,True)
    c.arrow(1215,1215,1215,1250,TEAL)
    node(c,930,1260,570,100,'GPU1 收到 t3 的完整输出','Y(t3) = P0(t3) + P1(t3)：[5120]','#FFFFFF',TEAL)
    c.text(385,1300,'GPU0 收到的是 t0、t1 的完整输出',25,MUTED,True)
    c.text(800,1425,'wo_a：组内压缩；wo_b：映射到 hidden 并求和。整个过程不混合不同 token。',27,INK,True)
    c.save('08_output_projections')


if __name__=='__main__':
    architecture();gather();balance();context();walkthrough();output_projections()
    print('Wrote 6 attention diagrams as SVG + PNG; all text bounds checked.')
