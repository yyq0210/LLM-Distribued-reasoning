#!/usr/bin/env python3
"""CPU arithmetic and small numerical examples for chapter 8; no torch/GPU."""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent


def norm(x):
    scale = math.sqrt(sum(v*v for v in x)/len(x) + 1e-6)
    return [v/scale for v in x]


def linear(x, w):
    return [sum(a*b for a, b in zip(x, col)) for col in zip(*w)]


def main():
    cfg = json.loads((ROOT/'references/deepseek_v41_flash/config.json').read_text())['text_config']
    h, heads, d, rq, ro, groups, layers = [cfg[k] for k in (
        'hidden_size', 'num_attention_heads', 'head_dim', 'q_lora_rank',
        'o_lora_rank', 'o_groups', 'num_hidden_layers')]
    t, sp, item = 2048, 4, 2
    counts = {
        'input_rmsnorm': 4*t*h,
        'q_down': 2*t*h*rq,
        'q_latent_rmsnorm': 4*t*rq,
        'swa_kv_projection': 2*t*h*d,
    }
    # AllGather moves complete rows; identical row-local weights commute with it.
    x = [[float(i+1), 2., -1., float(i % 3)] for i in range(8)]
    w = [[1., 2.], [-1., 3.], [0.5, -2.], [2., 1.]]
    f = lambda rows: [linear(norm(row), w) for row in rows]
    reference = f(x)
    shards = [x[i:i+2] for i in range(0, len(x), 2)]
    moved = sum((f(shard) for shard in shards), [])
    assert reference == moved
    # A nonlinear norm does not commute with a SUM reduction.
    a, b = [1., 2., 3., 4.], [4., 0., -1., 2.]
    correct = norm([u+v for u, v in zip(a,b)])
    wrong = [u+v for u, v in zip(norm(a),norm(b))]
    assert max(abs(u-v) for u,v in zip(correct,wrong)) > 0.1
    # Mixed rows preserve request and position identities, then invert the permutation.
    ids = [('A', i) for i in range(6)] + [('B', i) for i in range(2)]
    perm = [0,7,1,6,2,5,3,4]
    assigned = [ids[i] for i in perm]
    restored = [None]*len(ids)
    for j, i in enumerate(perm): restored[i] = assigned[j]
    assert restored == ids
    # Uniform token counts do not balance dense causal query work.
    contiguous = [sum(range(1,5)),sum(range(5,9))]
    zigzag = [1+2+7+8,3+4+5+6]
    assert contiguous == [10,26] and zigzag == [18,18]
    # Distributed softmax with one zero-valued attention sink, counted once.
    scores, vals, sink = [0.,1.,2.,3.], [10.,20.,30.,40.], 0.5
    partials=[]
    for rank in range(2):
        inds=range(rank*2,(rank+1)*2)
        z=sum(math.exp(scores[i]) for i in inds)
        n=sum(math.exp(scores[i])*vals[i] for i in inds)
        if rank==0: z += math.exp(sink)
        partials.append((n/z,math.log(z)))
    m=max(l for _,l in partials)
    result=sum(o*math.exp(l-m) for o,l in partials)/sum(math.exp(l-m) for _,l in partials)
    full=sum(math.exp(s)*v for s,v in zip(scores,vals))/(sum(map(math.exp,scores))+math.exp(sink))
    assert abs(result-full)<1e-12
    wo_a=heads*d*ro
    wo_b=groups*ro*h
    metrics={
        'model':'deepseek-ai/DeepSeek-V4.1-Flash', 'tokens':t, 'sp':sp,
        'flops_per_rank_before':counts,
        'flops_per_rank_after':{k:v//sp for k,v in counts.items()},
        'bf16_allgather_send_mib_per_rank':{
            'hidden': (sp-1)/sp*t*h*item/2**20,
            'q_latent_plus_swa_kv':(sp-1)/sp*t*(rq+d)*item/2**20},
        'output_weights':{'wo_a_per_layer':wo_a,'wo_b_per_layer':wo_b,
            'all_40_layers_fp8_payload_gib':layers*(wo_a+wo_b)/2**30,
            'all_40_layers_bf16_payload_gib':layers*(wo_a+wo_b)*2/2**30},
        'causal_pairs':{'contiguous':contiguous,'zigzag':zigzag},
        'kv_page_owners':{str(r):list(range(r,8,4)) for r in range(4)},
        'source_config_sha256':hashlib.sha256((ROOT/'references/deepseek_v41_flash/config.json').read_bytes()).hexdigest(),
    }
    lines=['CPU arithmetic only; no model weights, GPU workers or network collectives',
        f'DeepSeek-V4.1-Flash: H={h}, heads={heads}, d={d}, rq={rq}, groups={groups}, ro={ro}',
        f'T={t}, SP={sp}; 2 FLOPs per multiply-add; BF16 communication example']
    for name,v in counts.items(): lines.append(f'{name}: {v:,} -> {v//sp:,} FLOPs/rank')
    for name,v in metrics['bf16_allgather_send_mib_per_rank'].items(): lines.append(f'AllGather {name}: {v:.2f} MiB sent/rank')
    lines += [f'wo_a + wo_b: {wo_a:,} + {wo_b:,} = {wo_a+wo_b:,} parameters/layer',
        f'40-layer O weights: FP8 payload {layers*(wo_a+wo_b)/2**30:.4f} GiB; BF16 payload {layers*(wo_a+wo_b)*2/2**30:.4f} GiB',
        f'Causal pairs, contiguous={contiguous}; zigzag={zigzag}',
        'PASS: AllGather commutes with identical row-local norm/projection',
        'PASS: moving norm before SUM gives a different result (counterexample)',
        'PASS: mixed-request row permutation restores request/position identities',
        f'PASS: KV-partial LSE merge with sink counted once equals full attention ({full:.9f})',
        'No latency/speedup measurement; communication excludes scales, compressor, metadata and temporary buffers']
    (ROOT/'assets/08_attention_calculations.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (ROOT/'logs/08_attention_calculations.log').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
