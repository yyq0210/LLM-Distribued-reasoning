#!/usr/bin/env python3
"""CPU 上演示分块 softmax 的 LSE 合并公式，不执行分布式或 TokenSpeed kernel。"""

import math


def local_attention(scores, values):
    maximum = max(scores)
    masses = [math.exp(score - maximum) for score in scores]
    denominator = math.fsum(masses)
    output = math.fsum(mass * value for mass, value in zip(masses, values)) / denominator
    lse = maximum + math.log(denominator)
    return output, lse


def main():
    # e^score 为 [1, 3, 2, 4]；value 为标量，便于手算。
    scores = [math.log(value) for value in (1, 3, 2, 4)]
    values = [10, 20, 30, 40]
    states = [local_attention(scores[:2], values[:2]), local_attention(scores[2:], values[2:])]
    maximum_lse = max(lse for _, lse in states)
    masses = [math.exp(lse - maximum_lse) for _, lse in states]
    weights = [mass / math.fsum(masses) for mass in masses]
    merged = math.fsum(weight * output for weight, (output, _) in zip(weights, states))
    full, _ = local_attention(scores, values)
    # 完整参考式的分子是 1*10+3*20+2*30+4*40=290，分母是 10。
    assert math.isclose(full, 29.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(merged, full, rel_tol=0, abs_tol=1e-12)
    print("CPU formula demonstration; no GPU or distributed communication")
    for index, (output, lse) in enumerate(states):
        print(f"shard {index}: local_output={output:.6f}, LSE={lse:.6f}, merge_weight={weights[index]:.6f}")
    print(f"naive average (wrong): {(states[0][0] + states[1][0]) / 2:.6f}")
    print(f"LSE-weighted merge: {merged:.6f}")
    print(f"full attention:    {full:.6f}")
    print("PASS: LSE-weighted merge equals full attention within 1e-12")


if __name__ == "__main__":
    main()
