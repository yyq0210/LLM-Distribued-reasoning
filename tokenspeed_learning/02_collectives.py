#!/usr/bin/env python3
"""第二课预备练习：对照观察 AllGather、ReduceScatter、AllToAll。

使用 TokenSpeed 的 NCCL 后端，两进程各控制一张 GPU。
每一步都打印输入/输出，并检查实际数值；这不是带宽测速。
"""

from datetime import timedelta
import os

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager


def report(name, rank, before, after, expected):
    torch.testing.assert_close(after, expected, rtol=0, atol=0)
    for printing_rank in range(2):
        if printing_rank == rank:
            print(f"{name}: rank {rank}: {before.tolist()} -> {after.tolist()}", flush=True)
        dist.barrier()


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise ValueError("本例需要两个进程")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=60), device_id=device)
    try:
        group = (0, 1)
        process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
        backend = NcclBackend()
        backend.configure(use_pynccl=False)

        # Gather 是拼接，不是相加；每个 rank 最后都有全部分片。
        x = torch.tensor([1., 2.], device=device) + 2 * rank
        gathered = backend.all_gather(x, group)
        report("AllGather", rank, x, gathered, torch.tensor([1., 2., 3., 4.], device=device))

        # ReduceScatter 先逐元素求和，再让各 rank 只保留各自的结果分片。
        x = torch.tensor([1., 2., 3., 4.], device=device) * (10 if rank else 1)
        reduced = backend.reduce_scatter(x, group)
        expected = torch.tensor([11., 22.] if rank == 0 else [33., 44.], device=device)
        report("ReduceScatter", rank, x, reduced, expected)

        # AllToAll：输入按目标 rank 分块，输出按来源 rank 排列。
        # rank 0 发 [0] 给 0、[1] 给 1；rank 1 发 [10] 给 0、[11] 给 1。
        x = torch.tensor([0., 1.], device=device) + 10 * rank
        exchanged = torch.empty_like(x)
        backend.all_to_all_single(exchanged, x, group)
        expected = torch.tensor([0., 10.] if rank == 0 else [1., 11.], device=device)
        report("AllToAll", rank, x, exchanged, expected)
        if rank == 0:
            print("PASS: 三种通信操作的实际结果均正确", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
