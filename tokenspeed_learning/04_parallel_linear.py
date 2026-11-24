#!/usr/bin/env python3
"""用真实 TokenSpeed 线性层观察 column → ReLU → row 的 TP 数据流。

恰好两个进程，每进程一张 GPU。小矩阵仅验证数学和通信，不测性能。
为便于教学，每个进程先构造完整参考矩阵，再给真实层装入本地权重分片。
"""

from datetime import timedelta
import os

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.layers.linear import ColumnParallelLinear, RowParallelLinear


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
        set_global_backend(backend)

        # 按类文档使用 Y = X @ A；实际 parameter.weight 存的是 A.T。
        x = torch.tensor([[1., 2., 3., 4.]], device=device)
        a = torch.tensor([
            [1., 0., 0., 0., 1., 0.],
            [0., 1., 0., 0., 0., 1.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0.],
        ], device=device)
        b = torch.tensor([
            [1., 0.], [0., 1.], [1., 0.],
            [0., 1.], [1., 0.], [0., 1.],
        ], device=device)
        layer_args = dict(
            bias=False, params_dtype=torch.float32,
            tp_rank=rank, tp_size=2, tp_group=group,
        )
        # 四个实例用于分别展示开关行为，不是一个模型的四层。
        col = ColumnParallelLinear(4, 6, gather_output=False, **layer_args).to(device)
        col_gather = ColumnParallelLinear(4, 6, gather_output=True, **layer_args).to(device)
        row_partial = RowParallelLinear(
            6, 2, input_is_parallel=True, reduce_results=False, **layer_args,
        ).to(device)
        row = RowParallelLinear(
            6, 2, input_is_parallel=True, reduce_results=True, **layer_args,
        ).to(device)
        begin, end = rank * 3, (rank + 1) * 3
        with torch.no_grad():
            for layer in (col, col_gather):
                layer.weight.copy_(a[:, begin:end].T)
            for layer in (row_partial, row):
                layer.weight.copy_(b[begin:end, :].T)

            z_local, _ = col(x)
            z_full, _ = col_gather(x)
            # 正式示范链直接消费本地分片，不消费 z_full；无需中间 AllGather。
            activated = torch.relu(z_local)
            partial, _ = row_partial(activated)
            y, _ = row(activated)
            expected_z = x @ a
            expected_y = torch.relu(expected_z) @ b
            expected_partial = torch.relu(expected_z[:, begin:end]) @ b[begin:end]
            for actual, expected in (
                (z_local, expected_z[:, begin:end]), (z_full, expected_z),
                (partial, expected_partial), (y, expected_y),
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        for printing_rank in range(2):
            if rank == printing_rank:
                print(
                    f"rank={rank}\n"
                    f"  input X: {x.tolist()}\n"
                    f"  column weight [out_local, in]: {list(col.weight.shape)}\n"
                    f"  column gather_output=False: {z_local.tolist()}\n"
                    f"  column gather_output=True:  {z_full.tolist()}\n"
                    f"  row weight [out, in_local]: {list(row.weight.shape)}\n"
                    f"  row reduce_results=False: {partial.tolist()}\n"
                    f"  row reduce_results=True:  {y.tolist()}\n"
                    f"  dense reference: {expected_y.tolist()}", flush=True,
                )
            dist.barrier()
        if rank == 0:
            print("PASS: actual TokenSpeed ColumnParallelLinear + RowParallelLinear = dense reference", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
