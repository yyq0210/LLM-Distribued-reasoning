#!/usr/bin/env python3
"""Copy the tutorial source while excluding machine-local installation artifacts."""
import argparse
from pathlib import Path
import re
import shutil

PIN = '8289d0f4a9e122cec9ad7bc7bbb843fd9ade1613'
EXCLUDED = {'.git', '.tools', 'downloads', '__pycache__'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    repo = Path(__file__).resolve().parents[1]
    target = repo / 'tokenspeed_learning'
    if not (source / '08_attention_parallel_optimization.md').is_file():
        parser.error('Source must be the tokenspeed_learning tutorial directory')
    if source == target.resolve() or source in target.resolve().parents:
        parser.error('Source must be separate from the publication directory')
    count = 0
    for p in sorted(source.rglob('*')):
        relative = p.relative_to(source)
        if any(part in EXCLUDED for part in relative.parts) or not p.is_file():
            continue
        if p.is_symlink():
            raise ValueError(f'Refusing to follow a symlink: {relative}')
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == '.md' and relative.parts[0] != 'references':
            # GitHub cannot traverse a submodule as an ordinary directory.
            # Pin code links so they are readable in the web interface too.
            text = re.sub(
                r'(?<=\()\.\./tokenspeed/([^\s)]+)(?=\))',
                lambda m: f'https://github.com/lightseekorg/tokenspeed/blob/{PIN}/{m[1]}',
                p.read_text(),
            )
            dest.write_text(text)
            shutil.copymode(p, dest)
        else:
            shutil.copy2(p, dest)
        count += 1
    print(f'Synced {count} files to {target}')
    print('Excluded CUDA tools, downloaded packages and Python caches; source untouched.')


if __name__ == '__main__':
    main()
