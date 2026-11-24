#!/usr/bin/env python3
"""CPU-only MoE shape / communication accounting example, not a GPU benchmark."""

from collections import Counter
from pathlib import Path
import json


def calculate():
    ranks, hidden, tokens, experts, topk, item_bytes = 4, 4096, 16, 8, 2, 2
    owned = tokens // ranks
    routes = []
    matrix = [[0] * ranks for _ in range(ranks)]
    expert_counts = Counter()
    for source in range(ranks):
        for offset in range(owned):
            token = source * owned + offset
            # Two distinct destination ranks per token; exactly two assignments
            # from each source to each destination over its four owned tokens.
            destinations = ((source + offset) % ranks, (source + offset + 1) % ranks)
            selected = [2 * destinations[0] + offset % 2,
                        2 * destinations[1] + (1 - offset % 2)]
            for expert in selected:
                matrix[source][expert // 2] += 1
                expert_counts[expert] += 1
            routes.append({"token": token, "source": source, "experts": selected})
    assert all(row == [2, 2, 2, 2] for row in matrix)
    assert all(expert_counts[e] == 4 for e in range(experts))
    # A scalar expert-output stand-in checks that the two layouts sum exactly
    # the same routed contributions. It is not a neural expert or GEMM test.
    dense_partials = [[0] * tokens for _ in range(ranks)]
    returned = [0] * tokens
    reference = [0] * tokens
    for row in routes:
        token = row["token"]
        for slot, expert in enumerate(row["experts"]):
            contribution = (slot + 1) * (token + 1) * (expert + 1)
            dense_partials[expert // 2][token] += contribution
            returned[token] += contribution
            reference[token] += contribution
    reduced = [sum(rank[t] for rank in dense_partials) for t in range(tokens)]
    assert reduced == returned == reference
    row_bytes = hidden * item_bytes
    full_bytes = tokens * row_bytes
    ag_send = (ranks - 1) * full_bytes // ranks
    remote_rows = [sum(row) - row[i] for i, row in enumerate(matrix)]
    result = {
        "scope": "CPU arithmetic only; no GPU or network execution",
        "config": dict(world_size=ranks, hidden=hidden, tokens=tokens,
                       experts=experts, topk=topk, item_bytes=item_bytes),
        "rank_owned_rows": owned,
        "dispatch_source_destination_assignments": matrix,
        "expert_rows": [expert_counts[e] for e in range(experts)],
        "remote_rows_sent_per_rank": remote_rows,
        "ep_send_kib_per_rank": {
            "allgather": ag_send // 1024,
            "reduce_scatter": ag_send // 1024,
            "allgather_plus_reduce_scatter": 2 * ag_send // 1024,
            "alltoall_dispatch": remote_rows[0] * row_bytes // 1024,
            "alltoall_return": remote_rows[0] * row_bytes // 1024,
            "alltoall_dispatch_plus_return": 2 * remote_rows[0] * row_bytes // 1024,
        },
        "assumptions": [
            "Send bytes only; do not add receive bytes to the table",
            "Ring-equivalent AllGather/ReduceScatter, balanced toy routing",
            "No padding, metadata, protocol, quantization, or multi-hop accounting",
            "EP routed branch only; excludes attention/shared-expert communication",
        ],
        "routes": routes,
        "scalar_routed_output": reference,
    }
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = calculate()
    root.joinpath("assets/moe_dimensions.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [result["scope"],
             "world=4; attention TP=2 DP=2; routed experts TP=1 EP=4 DP=1",
             "tokens=16; per-DP=8; unique rows/rank=4; H=4096; E=8; topk=2; BF16",
             "dispatch count matrix (source rows, destination columns):"]
    lines.extend(str(row) for row in result["dispatch_source_destination_assignments"])
    lines.append(f"rows per expert: {result['expert_rows']}")
    for name, amount in result["ep_send_kib_per_rank"].items():
        lines.append(f"{name}: {amount} KiB sent per rank")
    lines.append("PASS: broadcast/local-filter/reduce and routed-return have identical scalar sums")
    lines.append("Not a performance measurement; excludes shared branch and other communication")
    output = "\n".join(lines) + "\n"
    root.joinpath("logs/moe_dimensions.log").write_text(output)
    print(output, end="")
