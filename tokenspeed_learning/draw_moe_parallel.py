#!/usr/bin/env python3
"""Draw repo-native SVG/PNG MoE diagrams from the CPU shape calculation."""

from functools import lru_cache
from html import escape
from pathlib import Path
import json
import math

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
FONT = ASSETS / "fonts/NotoSansCJKsc-Regular.otf"
INK, MUTED = "#14243A", "#52627B"
BLUE, TEAL, PURPLE, ORANGE = "#2563EB", "#087F79", "#7654C4", "#B95A11"


@lru_cache(None)
def font(size):
    return ImageFont.truetype(str(FONT), size)


class Canvas:
    def __init__(self, height, title):
        self.width, self.height = 1600, height
        self.image = Image.new("RGB", (self.width, height), "#F5F7FB")
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="{height}" viewBox="0 0 1600 {height}">',
                    f'<title>{escape(title)}</title>',
                    f'<rect width="1600" height="{height}" fill="#F5F7FB"/>']
        self.text(50, 54, title, 42)

    def text(self, x, y, value, size=26, color=INK, center=False):
        # This CJK font lacks some Unicode subscript glyphs; use explicit
        # ASCII indices in figures. The accompanying document uses LaTeX.
        value = value.translate(str.maketrans({"ᵣ": "_r", "₀": "_0", "₁": "_1"}))
        anchor = "mm" if center else "lm"
        bounds = self.draw.textbbox((x, y), value, font=font(size), anchor=anchor)
        assert 0 <= bounds[0] and bounds[2] <= self.width, (value, bounds)
        assert 0 <= bounds[1] and bounds[3] <= self.height, (value, bounds)
        self.draw.text((x, y), value, font=font(size), fill=color, anchor=anchor)
        align = "middle" if center else "start"
        self.svg.append(f'<text x="{x}" y="{y}" text-anchor="{align}" dominant-baseline="central" font-family="Noto Sans CJK SC,sans-serif" font-size="{size}" fill="{color}">{escape(value)}</text>')

    def box(self, x, y, w, h, fill="#FFFFFF", stroke="#CBD5E1"):
        self.draw.rounded_rectangle((x, y, x+w, y+h), 16, fill=fill, outline=stroke, width=2)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

    def arrow(self, x1, y1, x2, y2, color=MUTED):
        self.draw.line((x1, y1, x2, y2), fill=color, width=3)
        length = math.hypot(x2-x1, y2-y1)
        ux, uy = (x2-x1)/length, (y2-y1)/length
        pts = [(x2,y2),(x2-13*ux+6*uy,y2-13*uy-6*ux),(x2-13*ux-6*uy,y2-13*uy+6*ux)]
        self.draw.polygon(pts, fill=color)
        self.svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="3"/>')
        self.svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in pts)}" fill="{color}"/>')

    def save(self, name):
        self.image.save(ASSETS / f"{name}.png")
        (ASSETS / f"{name}.svg").write_text("\n".join(self.svg + ["</svg>"]) + "\n")


def groups():
    c = Canvas(1000, "MoE 并行：同一组 4 张 GPU，按计算阶段改变分工")
    c.text(50, 112, "示例：16 个独立 token · H=4096 · 8 个路由专家 · Top-2 · BF16", 27, MUTED)
    for dp, x, name in [(0,50,"A"),(1,825,"B")]:
        c.box(x,160,725,290,"#EAF1FF",BLUE)
        c.text(x+362,200,f"Attention 副本 {dp}：TP=2，处理 {name} 的 8 个 token",27,BLUE,True)
        for tp in range(2):
            rank=dp*2+tp
            left=x+20+tp*350
            c.box(left,235,335,180)
            c.text(left+167,270,f"GPU {rank}",32,BLUE,True)
            c.text(left+167,315,f"Q heads {tp*16}–{tp*16+15}",26,INK,True)
            c.text(left+167,355,f"KV heads {tp*4}–{tp*4+3}",26,INK,True)
            c.text(left+167,390,"o_proj 部分和 [8,4096]",23,MUTED,True)
    c.text(800,493,"输入 X：GPU 0/1 持有 A；GPU 2/3 持有 B。每份 X 都是 [8,4096]。",27,INK,True)
    c.arrow(800,520,800,565)
    c.box(50,580,1500,235,"#F3EEFC",PURPLE)
    c.text(800,622,"路由专家：EP=4，expert TP=1；同一专家集合跨越两个 attention 副本",29,PURPLE,True)
    for rank in range(4):
        x=75+rank*370
        c.box(x,665,340,115)
        c.text(x+170,700,f"GPU {rank}：E{rank*2}、E{rank*2+1}",29,PURPLE,True)
        c.text(x+170,746,"每个专家权重完整存放",24,MUTED,True)
    c.text(800,860,"共享专家分支：本例按 (0,1)、(2,3) 做 TP=2；所有 token 都执行。",27,TEAL,True)
    c.text(800,907,"4 = attention DP2 × TP2 = routed EP4 × expert TP1；不是 2 × 2 × 4。",28,INK,True)
    c.text(50,963,"教学布局，未启动四卡服务。共享专家是否存在、采用何种分组，取决于模型和后端。",23,MUTED)
    c.save("moe_4gpu_groups")


def dispatch(data):
    c=Canvas(1330,"两条 EP 路径：广播全部 token，或只发送被选中的 token")
    c.text(50,110,"相同起点：每卡拥有 4 个不重复 token，X=[4,4096]；全局共 16 个 token",27,MUTED)
    for x,w,title,col in [(50,725,"AllGather → 本地专家 → ReduceScatter",BLUE),(825,725,"Top-2 → AllToAll → 专家 → 返回",PURPLE)]:
        c.box(x,150,w,1010,"#FFFFFF",col)
        c.text(x+w/2,191,title,27,col,True)
    stages=[
      ("每卡 X=[4,4096]","每卡 X=[4,4096]"),
      ("EP4 AllGather → X=[16,4096]","Router [4,8] → Top-2 ids/weights [4,2]"),
      ("本地路由筛选 → 本卡专家共 8 行","Dispatch → 每卡收到 8 条专家任务"),
      ("2 个本地专家：各处理 [4,4096]","2 个本地专家：各处理 [4,4096]"),
      ("按原 token 位置写回 Pᵣ=[16,4096]","专家输出 [8,4096] → 返回原 owner"),
      ("EP4 ReduceScatter → Yᵣ=[4,4096]","按 Top-2 权重合并 → Yᵣ=[4,4096]"),
    ]
    for j,(left,right) in enumerate(stages):
        y=240+j*130
        for x,txt,col in [(75,left,BLUE),(850,right,PURPLE)]:
            c.box(x,y,675,80,"#F8FAFC", "#CBD5E1")
            c.text(x+337,y+40,txt,25,INK,True)
            if j<5:c.arrow(x+337,y+87,x+337,y+120,col)
    c.text(412,1052,"部分和按 token 对齐；缺失贡献填零。",24,MUTED,True)
    c.text(1187,1052,"发送完整 H 维向量；按专家目的地分包。",24,MUTED,True)
    traffic=data["ep_send_kib_per_rank"]
    c.text(412,1110,f"每卡发送：{traffic['allgather_plus_reduce_scatter']} KiB / dispatch + combine",26,BLUE,True)
    c.text(1187,1110,f"每卡发送：{traffic['alltoall_dispatch_plus_return']} KiB / dispatch + combine",26,PURPLE,True)
    c.text(50,1210,"通信量仅计 BF16 激活发送：均衡示例、无 padding/元数据；不含 attention 和共享分支。",25,MUTED)
    c.text(50,1255,"AllToAll 不是单个 API 的替换：还需要打包、目标计数、反向回传、恢复 token 顺序和加权合并。",25,INK)
    c.text(50,1300,"图中的每专家 4 行是人为构造的均衡路由，真实推理中行数随输入变化。",23,MUTED)
    c.save("moe_dispatch_shapes")


def sequence():
    c=Canvas(1240,"SP：先合并 attention 部分和，再让每个 token 只有一个 owner")
    c.text(50,110,"以 attention 副本 A（GPU 0/1）为例；副本 B（GPU 2/3）执行相同步骤",27,MUTED)
    c.box(320,150,960,85,"#FFFFFF")
    c.text(800,192,"o_proj：GPU 0、GPU 1 各持 Pᵣ=[8,4096]，数值是部分和",28,INK,True)
    for x,col in [(60,BLUE),(825,TEAL)]:
        c.arrow(800,245,x+357,290,col)
        c.box(x,300,715,230,"#FFFFFF",col)
    c.text(417,342,"AllReduce SUM",31,BLUE,True)
    c.text(417,397,"GPU 0：完整 A [8,4096]",27,INK,True)
    c.text(417,440,"GPU 1：同一份 A [8,4096]",27,INK,True)
    c.text(417,492,"之后逐行 norm/router 可能重复计算",24,MUTED,True)
    c.text(1182,342,"ReduceScatter SUM（沿 token 行）",29,TEAL,True)
    c.text(1182,397,"GPU 0：A₀ [4,4096]，完整数值",27,INK,True)
    c.text(1182,440,"GPU 1：A₁ [4,4096]，完整数值",27,INK,True)
    c.text(1182,492,"特征维 H 不变；token 不重复",24,MUTED,True)
    c.text(800,583,"采用 RS 后：四卡分别持 A₀、A₁、B₀、B₁，每份 [4,4096]",29,TEAL,True)
    c.arrow(800,610,420,650,PURPLE)
    c.arrow(800,610,1180,650,ORANGE)
    c.box(60,665,715,265,"#F3EEFC",PURPLE)
    c.box(825,665,715,265,"#FFF4EA",ORANGE)
    c.text(417,710,"路由专家分支：EP4",30,PURPLE,True)
    c.text(417,764,"采用前图任一正确的 dispatch / combine",24,INK,True)
    c.text(417,817,"每卡返回自身 token 的结果",27,INK,True)
    c.text(417,878,"Y_routed=[4,4096]",30,PURPLE,True)
    c.text(1182,710,"共享专家分支：本例 TP2",30,ORANGE,True)
    c.text(1182,764,"组内 AG：每卡恢复 [8,4096]",26,INK,True)
    c.text(1182,817,"Column → 激活 → Row → 组内 RS",26,INK,True)
    c.text(1182,878,"Y_shared=[4,4096]",30,ORANGE,True)
    c.arrow(417,942,800,985,PURPLE)
    c.arrow(1182,942,800,985,ORANGE)
    c.box(290,995,1020,78,"#FFFFFF")
    c.text(800,1034,"相同 token 顺序逐元素相加 → Y=[4,4096]",29,INK,True)
    c.arrow(800,1082,800,1120)
    c.text(800,1150,"下一次普通 head-TP attention 前，按需在 TP2 组内 AG 回 [8,4096]",27,INK,True)
    c.text(50,1210,"SP 减少部分重复激活/计算；不自动减少专家 GEMM，也不意味着通信从此消失。",24,MUTED)
    c.save("moe_sp_shared")


def progression():
    """Fixed GPU lanes expose groups, replicated rows, and SP token ownership."""
    data = json.loads((ASSETS / "moe_dimensions.json").read_text())
    selected = next(row for row in data["routes"] if row["token"] == 1)
    assert selected["source"] == 0 and selected["experts"] == [3, 4]
    xs = [395, 720, 1045, 1370]
    titles = ["① 基础 AllGather：Attention 副本内复制，跨 DP 收集",
              "② 改用 AllToAll：指定 owner，再按专家目的地发送",
              "③ AllToAll + SP：Attention 后直接保留不重复 token"]
    colors = [BLUE, PURPLE, TEAL]
    fills = {"A0": "#DBEAFE", "A1": "#BFDBFE", "B0": "#FFEDD5", "B1": "#FED7AA"}

    def line(c, x1, y1, x2, y2, color="#CBD5E1", dashed=False):
        if dashed:
            length = math.hypot(x2-x1, y2-y1)
            for pos in range(0, math.ceil(length), 14):
                stop = min(pos+6, length)
                c.draw.line((x1+(x2-x1)*pos/length, y1+(y2-y1)*pos/length,
                             x1+(x2-x1)*stop/length, y1+(y2-y1)*stop/length), fill=color, width=2)
        else:
            c.draw.line((x1,y1,x2,y2), fill=color, width=2)
        dash = ' stroke-dasharray="6 8"' if dashed else ''
        c.svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2"{dash}/>')

    def label(c, x, y, value, color, width=230, size=24):
        c.box(x-width/2,y-19,width,38,"#FFFFFF",color)
        c.text(x,y,value,size,color,True)

    def exchange(c, a, b, y, caption, color, width=245):
        # Two short opposing arrows plus a central label show the participant
        # pair without implying that the label is an extra GPU.
        mid=(a+b)/2
        c.arrow(a,y,b,y,color)
        c.arrow(b,y,a,y,color)
        label(c,mid,y,caption,color,width)

    def token_card(c, x, y, tokens, shape, accent=INK):
        c.box(x-135,y,270,80,"#FFFFFF","#C6D2E1")
        c.text(x,y+23,shape,25,accent,True)
        for j, token in enumerate(tokens):
            left=x-123+j*126
            c.box(left,y+43,120,28,fills[token],fills[token])
            c.text(left+60,y+57,f"{token} · 4行",20,INK,True)

    def note(c, y, lines, color=MUTED):
        for i, txt in enumerate(lines):
            c.text(52,y+i*31,txt,23,color)

    def panel(c, top, version):
        color=colors[version]
        c.box(25,top,1550,1155,"#FFFFFF",color)
        c.text(55,top+35,titles[version],32,color)
        # Fixed rank lanes continue through all phases. The left margin is
        # reserved for phase names and never denotes another participant.
        for x in xs:
            line(c,x,top+145,x,top+1050,dashed=True)
        for dp, left in [(0,245),(1,895)]:
            c.box(left,top+75,625,210,"#EAF1FF",BLUE)
            c.text(left+312,top+101,f"Attention DP 副本 {dp} · TP组 ({2*dp},{2*dp+1})",25,BLUE,True)
            for tp in range(2):
                rank=2*dp+tp
                x=xs[rank]
                c.text(x,top+139,f"GPU {rank}",29,BLUE,True)
                c.box(x-135,top+160,270,65,"#FFFFFF")
                c.text(x,top+180,f"Q heads {16*tp}–{16*tp+15}",22,INK,True)
                c.text(x,top+208,"o_proj 部分和 [8,H]",22,MUTED,True)
            exchange(c,xs[2*dp],xs[2*dp+1],top+257,
                     "TP2 RS" if version==2 else "TP2 AllReduce",
                     TEAL if version==2 else BLUE,width=220)
        note(c,top+185,["Attention", "按 head 分工"],BLUE)
        note(c,top+326,["Norm 输入", "H = 4096"],TEAL if version==2 else BLUE)
        chunks = [["A0"],["A1"],["B0"],["B1"]]
        replicated = [["A0","A1"],["A0","A1"],["B0","B1"],["B0","B1"]]
        for rank,x in enumerate(xs):
            c.arrow(x,top+289,x,top+309,TEAL if version==2 else BLUE)
            token_card(c,x,top+312,chunks[rank] if version==2 else replicated[rank],
                       "[4,H] 完整数值" if version==2 else "[8,H] 完整数值")

        # Draw the EP background before dispatch arrows so endpoints remain
        # visible. Attention replica boundaries are absent from the EP group.
        c.box(245,top+610,1300,145,"#F3EEFC",PURPLE)
        if version==0:
            c.text(895,top+633,"一个 EP4 组 (0,1,2,3)：8 个专家分布在四卡，跨越两个 attention 副本",24,PURPLE,True)
        else:
            c.text(1290,top+633,"EP4 组 (0,1,2,3)",24,PURPLE,True)
        if version==0:
            note(c,top+433,["跨 DP 收集", "每卡得到 A+B"],ORANGE)
            # AG pairs are the corresponding TP positions across replicas.
            pairs = [(0,2,435,"DP组 (0,2) AG"),(1,3,482,"DP组 (1,3) AG")]
            for a,b,offset,caption in pairs:
                line(c,xs[a],top+399,xs[a],top+offset,ORANGE)
                line(c,xs[b],top+399,xs[b],top+offset,ORANGE)
            for a,b,offset,caption in pairs:
                exchange(c,xs[a],xs[b],top+offset,caption,ORANGE,width=275)
            for x in xs:
                c.arrow(x,top+495,x,top+523,ORANGE)
                c.box(x-135,top+528,270,60,"#FFF6EC",ORANGE)
                c.text(x,top+548,"A + B（无重复）",24,INK,True)
                c.text(x,top+573,"[16,H]",22,MUTED,True)
                c.arrow(x,top+590,x,top+608,PURPLE)
            note(c,top+672,["本地专家", "按路由筛选"],PURPLE)
        else:
            note(c,top+442,["dispatch 输入", "唯一 owner"],color)
            for rank,x in enumerate(xs):
                c.arrow(x,top+399,x,top+423,color)
                token_card(c,x,top+426,chunks[rank],
                           "直接使用 SP 行" if version==2 else "本地选 4 行 owner",color)
            c.text(865,top+540,"只画 t1 的 Top-2：E3 在 GPU1，E4 在 GPU2；其余 token 路径省略",23,PURPLE,True)
            # Real selected route from moe_dimensions.json: token 1 goes from
            # owner 0 to E3 on rank 1 and E4 on rank 2.
            label(c,xs[0],top+555,"t1 @ GPU0",PURPLE,width=210,size=23)
            for dest in (1,2):
                c.arrow(xs[0],top+578,xs[dest],top+650,PURPLE)
            note(c,top+672,["AllToAll dispatch", "跨 DP 找专家"],PURPLE)
        for rank,x in enumerate(xs):
            c.box(x-135,top+658,270,78,"#FFFFFF",PURPLE)
            c.text(x,top+681,f"GPU{rank}：E{2*rank} / E{2*rank+1}",25,PURPLE,True)
            c.text(x,top+714,"8 条任务 · [8,H]",23,MUTED,True)

        if version==0:
            note(c,top+796,["合并专家贡献", "不是 attention RS"],PURPLE)
            for x in xs:
                c.arrow(x,top+758,x,top+782,PURPLE)
            c.box(245,top+787,1300,42,"#F3EEFC",PURPLE)
            c.text(895,top+808,"EP4 ReduceScatter：每卡部分贡献 [16,H] → 按 token 求和后，各取 [4,H]",25,PURPLE,True)
            for x in xs:c.arrow(x,top+833,x,top+855,PURPLE)
        else:
            note(c,top+789,["AllToAll 返回", "owner 加权合并"],PURPLE)
            for src in (1,2):
                c.arrow(xs[src],top+758,xs[0],top+854,PURPLE)
            c.text(1125,top+799,"E3、E4 的结果回到 t1 的 owner GPU0",24,PURPLE,True)
            c.text(1125,top+832,"其他 token 同样返回各自 owner，连线省略",22,MUTED,True)
        for rank,x in enumerate(xs):
            token_card(c,x,top+861,chunks[rank],"本卡路由结果 [4,H]",PURPLE)

        if version==2:
            note(c,top+1009,["保持 SP 布局", "暂不复制 token"],TEAL)
            for rank,x in enumerate(xs):
                c.arrow(x,top+947,x,top+988,TEAL)
                token_card(c,x,top+995,chunks[rank],"保留 [4,H]",TEAL)
        else:
            note(c,top+1009,["恢复 TP 输入布局", "组内再次复制"],BLUE)
            for a,b in [(0,1),(2,3)]:
                exchange(c,xs[a],xs[b],top+971,"TP2 AllGather",BLUE,width=225)
            for rank,x in enumerate(xs):
                token_card(c,x,top+995,replicated[rank],"恢复 [8,H]",BLUE)
        if version==2:
            c.text(55,top+1110,"共享分支（并列，未展开）：TP2 AG → MLP → RS，返回相同 [4,H] 后相加；下一 attention 前按需 AG。",23,TEAL)
        else:
            c.text(55,top+1110,"共享分支（并列，未展开）：每个 TP2 组计算共享 MLP → AllReduce，输出 [8,H] 与路由结果相加。",23,MUTED)

    overview = Canvas(3790,"MoE 三个版本：每张 GPU 一列，看见 TP、DP、EP 与 SP")
    overview.text(50,110,"蓝框：一个 attention DP 副本 / TP2组   ·   紫框：跨四卡的 EP4组   ·   色块：实际 token 行",25,MUTED)
    for version in range(3):
        panel(overview,160+version*1190,version)
    overview.text(50,3750,"每个 A0/A1/B0/B1 含 4 行，H 均为 4096。仅示意教学数据流，不是网络拓扑或性能测量。",24,MUTED)
    overview.save("moe_three_versions")
    for version in range(3):
        c=Canvas(1350,f"MoE 版本 {version+1}：逐 GPU 查看分组、token 与通信")
        c.text(50,110,"蓝框是 TP2/attention DP 副本；紫框是 EP4；每个 token 色块含 4 行。",26,MUTED)
        panel(c,155,version)
        c.save(f"moe_version_{version+1}_lanes")


if __name__ == "__main__":
    data=json.loads((ASSETS / "moe_dimensions.json").read_text())
    groups()
    dispatch(data)
    sequence()
    progression()
    print("Created MoE overview, three GPU-lane detail figures, and supporting diagrams; text bounds checked")
