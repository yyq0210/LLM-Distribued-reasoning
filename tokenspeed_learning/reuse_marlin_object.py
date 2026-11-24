#!/usr/bin/env python3
"""仅复用本次失败构建中已成功生成且未改动的 Marlin 对象。

本次只补齐 cuBLAS/cuRAND 头文件；Marlin 源目录不包含这两种库。
其余文件均交给真实 nvcc，缓存只在 manifest 的源文件和对象哈希一致时生效。
此 launcher 不在 env.sh 中启用，不影响之后的普通构建。
"""
import hashlib
import json
import os
from pathlib import Path
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


args = sys.argv[1:]
entry = json.loads(Path(__file__).with_name("marlin_object_manifest.json").read_text())
if "-c" in args and "-o" in args:
    source = args[args.index("-c") + 1]
    output = args[args.index("-o") + 1]
    if (
        source == entry["source"]
        and output == entry["object"]
        and "-gencode=arch=compute_90a,code=sm_90a" in args
        and "-std=c++20" in args
        and sha256(source) == entry["source_sha256"]
        and Path(output).is_file()
        and sha256(output) == entry["object_sha256"]
    ):
        print("Reusing verified, unchanged sm90a Marlin object from this build", flush=True)
        sys.exit(0)
os.execv(args[0], args)
