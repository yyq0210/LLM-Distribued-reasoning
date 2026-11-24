#!/usr/bin/env python3
"""仅在 CPU 上调用真实 Mapping 类计算 8 个 rank 的分组，不启动 8 卡任务。"""

from tokenspeed.runtime.distributed.mapping import Mapping


def main():
    print("CPU-only layout calculation; no GPU workers or network collectives started")
    print("world=8; PP=2; per stage: attention TP=2 DP=2; dense TP=2 DP=2; MoE TP=1 EP=4 DP=1")
    for rank in range(8):
        mapping = Mapping(
            rank=rank, world_size=8,
            attn_tp_size=2, attn_dp_size=2,
            dense_tp_size=2, dense_dp_size=2,
            moe_tp_size=1, moe_ep_size=4, moe_dp_size=1,
            pp_size=2,
        )
        print(
            f"rank={rank} pp_rank={mapping.pp_rank} "
            f"attn_tp={mapping.attn.tp_group} attn_dp={mapping.attn.dp_group} "
            f"pp={mapping.pp_group} moe_ep={mapping.moe.ep_group}"
        )


if __name__ == "__main__":
    main()
