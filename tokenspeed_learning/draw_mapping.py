#!/usr/bin/env python3
"""把已保存的 Mapping 输出画成 SVG 和 PNG，不启动 GPU 或通信进程。"""

import ast
from functools import lru_cache
from html import escape
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
FONT = ASSETS / "fonts/NotoSansCJKsc-Regular.otf"
WIDTH, HEIGHT = 1600, 1600
INK, MUTED = "#16243A", "#53647B"
BLUE, TEAL, ORANGE, PURPLE = "#2563EB", "#087F79", "#C65A13", "#7654C4"
IMAGE = Image.new("RGB", (WIDTH, HEIGHT), "#F5F7FB")
DRAW = ImageDraw.Draw(IMAGE)
SVG = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
       '<title>TokenSpeed 八个 rank 的 TP DP PP EP 分组</title>',
       '<desc>两个 PP 阶段，每阶段 attention TP2 DP2，MoE EP4。底部展示 rank 0 的四种组关系。</desc>',
       f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#F5F7FB"/>']


@lru_cache(maxsize=None)
def font(size):
    return ImageFont.truetype(str(FONT), size)


def text(x, y, value, size=26, color=INK, center=False):
    anchor = "mm" if center else "lm"
    bounds = DRAW.textbbox((x, y), value, font=font(size), anchor=anchor)
    assert 0 <= bounds[0] and bounds[2] <= WIDTH, (value, bounds)
    DRAW.text((x, y), value, font=font(size), fill=color, anchor=anchor)
    svg_anchor = "middle" if center else "start"
    SVG.append(f'<text x="{x}" y="{y}" text-anchor="{svg_anchor}" dominant-baseline="central" '
               f'font-family="Noto Sans CJK SC,Noto Sans SC,sans-serif" font-size="{size}" fill="{color}">{escape(value)}</text>')


def box(x, y, width, height, fill, stroke=None, radius=16, border=2):
    DRAW.rounded_rectangle((x, y, x + width, y + height), radius, fill=fill, outline=stroke, width=border)
    SVG.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
               f'fill="{fill}" stroke="{stroke or "none"}" stroke-width="{border}"/>')


def line(x1, y1, x2, y2, color, width=3, dashed=False, arrow=False):
    length = math.hypot(x2 - x1, y2 - y1)
    if dashed:
        for start in range(0, math.ceil(length), 15):
            end = min(start + 8, length)
            DRAW.line((x1 + (x2 - x1) * start / length, y1 + (y2 - y1) * start / length,
                       x1 + (x2 - x1) * end / length, y1 + (y2 - y1) * end / length), fill=color, width=width)
    else:
        DRAW.line((x1, y1, x2, y2), fill=color, width=width)
    dash = ' stroke-dasharray="8 7"' if dashed else ''
    SVG.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{dash}/>')
    if arrow:
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        points = [(x2, y2), (x2 - 13 * ux + 7 * uy, y2 - 13 * uy - 7 * ux),
                  (x2 - 13 * ux - 7 * uy, y2 - 13 * uy + 7 * ux)]
        DRAW.polygon(points, fill=color)
        SVG.append(f'<polygon points="{" ".join(f"{x},{y}" for x, y in points)}" fill="{color}"/>')


def rank_node(cx, cy, rank, color=BLUE, width=154, height=58):
    box(cx - width / 2, cy - height / 2, width, height, "#FFFFFF", color, radius=12)
    text(cx, cy, f"rank {rank}", size=29, color=color, center=True)


def read_groups():
    rows = {}
    pattern = re.compile(r"rank=(\d+) pp_rank=(\d+) attn_tp=(\([^)]*\)) attn_dp=(\([^)]*\)) pp=(\([^)]*\)) moe_ep=(\([^)]*\))")
    for match in pattern.finditer((ROOT / "logs/05_mapping.log").read_text()):
        rank, stage, *groups = match.groups()
        rows[int(rank)] = dict(zip(("attn_tp", "attn_dp", "pp", "moe_ep"), map(ast.literal_eval, groups)))
        rows[int(rank)]["stage"] = int(stage)
    assert set(rows) == set(range(8))
    for rank, row in rows.items():
        for kind in ("attn_tp", "attn_dp", "pp", "moe_ep"):
            assert rank in row[kind]
    return rows


def main():
    rows = read_groups()
    text(50, 55, "8 个 rank，按层采用不同的并行分工", size=43)
    text(50, 110, "world = 8   ·   PP = 2   ·   每阶段 4 个 rank   ·   分组来自实际 Mapping 输出", size=26, color=MUTED)

    box(30, 150, 1540, 555, "#FFFFFF")
    text(60, 190, "① Attention / Dense：每阶段 TP = 2，DP = 2", size=32)
    text(60, 233, "蓝色横排是 attention 副本；绿色虚线连接不同副本中相同 TP 位置的进程。", size=25, color=MUTED)
    for stage, x in ((0, 60), (1, 850)):
        base = stage * 4
        box(x, 270, 690, 340, "#F8FAFF", "#CBD5E1")
        text(x + 345, 298, f"PP 阶段 {stage}：rank {base}–{base + 3}", size=27, center=True)
        centers = (x + 170, x + 520)
        for dp, cy in ((0, 377), (1, 550)):
            group = rows[base + 2 * dp]["attn_tp"]
            box(x + 20, cy - 47, 650, 88, "#EAF1FF", radius=12)
            text(x + 345, cy - 29, f"副本 {dp} · TP {group}", size=22, color=BLUE, center=True)
            line(centers[0] + 80, cy + 7, centers[1] - 80, cy + 7, BLUE)
            for tp, cx in enumerate(centers):
                rank_node(cx, cy + 7, group[tp], height=48)
        for tp, cx in enumerate(centers):
            line(cx, 409, cx, 532, TEAL, dashed=True)
            box(cx - 105, 443, 210, 43, "#FFFFFF", radius=10)
            text(cx, 464, f"DP {rows[base + tp]['attn_dp']}", size=24, color=TEAL, center=True)
    line(765, 440, 835, 440, ORANGE, width=4, arrow=True)
    text(800, 408, "PP", size=24, color=ORANGE, center=True)
    pp_pairs = "   ".join(f"{a} → {b}" for a, b in (rows[r]["pp"] for r in range(4)))
    text(800, 649, f"阶段间对应位置传递激活： {pp_pairs}", size=28, color=ORANGE, center=True)
    text(800, 684, "虚线表示 DP 分组关系，不表示推理时要做梯度同步。", size=23, color=MUTED, center=True)

    box(30, 730, 1540, 310, "#FFFFFF")
    text(60, 773, "② MoE：同一批 rank，每阶段改为 TP = 1，EP = 4，DP = 1", size=31)
    text(60, 815, "每个阶段的专家集合分布在 4 个 rank 上；没有新增 GPU。", size=25, color=MUTED)
    for stage, x in ((0, 60), (1, 850)):
        group = rows[stage * 4]["moe_ep"]
        box(x, 852, 690, 150, "#F3EEFC", "#D9CDEF")
        text(x + 345, 884, f"阶段 {stage} 的 EP 组 {group}", size=26, color=PURPLE, center=True)
        for position, rank in enumerate(group):
            rank_node(x + 93 + position * 168, 952, rank, color=PURPLE, width=143)

    box(30, 1065, 1540, 450, "#FFFFFF")
    text(60, 1110, "③ 只看 rank 0：同一个进程同时属于四种组", size=32)
    rank_node(170, 1320, 0, width=190, height=82)
    cards = (
        ("attn_tp", BLUE, "#EAF1FF", "与 rank 1 合作计算 attention"),
        ("attn_dp", TEAL, "#E9F7F3", "rank 2 是另一个副本中相同 TP 位置的成员"),
        ("pp", ORANGE, "#FFF3E8", "向下一阶段对应位置的 rank 4 传递激活"),
        ("moe_ep", PURPLE, "#F3EEFC", "与 rank 1、2、3 分担本阶段的不同专家"),
    )
    names = {"attn_tp": "TP", "attn_dp": "DP", "pp": "PP", "moe_ep": "EP"}
    for index, (key, color, fill, explanation) in enumerate(cards):
        cy = 1190 + index * 85
        line(270, 1320, 415, cy, color, width=3)
        box(420, cy - 32, 1100, 64, fill, radius=12)
        text(445, cy, f"{names[key]} {rows[0][key]}", size=26, color=color)
        text(760, cy, explanation, size=25)

    text(50, 1555, "这是逻辑分组示意：CPU 上计算了分组；未启动 8 卡服务，也未验证跨机通信。", size=25, color=MUTED)
    ASSETS.mkdir(exist_ok=True)
    IMAGE.save(ASSETS / "05_mapping_groups.png")
    SVG.append("</svg>")
    (ASSETS / "05_mapping_groups.svg").write_text("\n".join(SVG) + "\n")
    print("Created assets/05_mapping_groups.png and assets/05_mapping_groups.svg from logs/05_mapping.log")


if __name__ == "__main__":
    main()
