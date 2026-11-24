#!/usr/bin/env python3
"""第一课：把一个线性层分给两张 GPU，再用 AllReduce 合并结果。

需要 torchrun 启动两个进程，每个进程控制一张卡。
--api pytorch 使用 PyTorch 的 NCCL 接口；--api tokenspeed 使用真实 TokenSpeed 后端。
这只是通信正确性练习，不加载模型，也不是性能基准。
"""

import argparse
from datetime import timedelta
import os

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True, choices=("pytorch", "tokenspeed"))
    args = parser.parse_args()

    # torchrun 设置这些变量：全局进程编号、本机进程编号、总进程数。
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError("本例需要恰好两个进程，对应两张 GPU")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=60), device_id=device)

    try:
        # 完整计算 y = W @ x。这里只用小整数，方便手算核对。
        # W = [[1, 0, 1, 0], [0, 1, 0, 1]]，x = [1, 2, 3, 4]。
        # 按输入维度拆成两份，每张卡计算一个不完整的 y。
        # 真实大模型通常在加载时就把权重分片；这里先构造完整矩阵仅便于教学。
        weight = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]], device=device)
        x = torch.tensor([1., 2., 3., 4.], device=device)
        begin, end = rank * 2, (rank + 1) * 2
        partial = weight[:, begin:end] @ x[begin:end]
        before = partial.clone()

        if args.api == "tokenspeed":
            from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
            from tokenspeed.runtime.distributed.process_group_manager import (
                process_group_manager,
            )

            group = tuple(range(world_size))
            process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
            backend = NcclBackend()
            backend.configure(use_pynccl=False)
            result = backend.all_reduce(partial, group, op=dist.ReduceOp.SUM)
        else:
            # 所有参与者必须按相同顺序调用这个操作，不能只让 rank 0 调用。
            dist.all_reduce(partial, op=dist.ReduceOp.SUM)
            result = partial

        expected = weight @ x
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.cuda.synchronize(device)
        # 按 rank 顺序打印，额外 barrier 仅为了让教学输出易读。
        for printing_rank in range(world_size):
            if rank == printing_rank:
                print(
                    f"rank={rank}, local_rank={local_rank}, GPU={torch.cuda.get_device_name(device)}\n"
                    f"  自己算出的部分结果: {before.cpu().tolist()}\n"
                    f"  AllReduce(SUM) 后:  {result.cpu().tolist()}\n"
                    f"  完整矩阵参考答案:   {expected.cpu().tolist()}",
                    flush=True,
                )
            dist.barrier()
        if rank == 0:
            print(f"PASS: api={args.api}，两张卡都得到完整答案 [4.0, 6.0]", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
