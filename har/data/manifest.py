# ===========================
# scripts/create_manifest.py
# ===========================
import os
import re
import pandas as pd

import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Generate manifest.parquet for sequence data"
    )
    parser.add_argument(
        "--seq-dir", type=str, required=True,
        help="Directory containing .npz sequence files (e.g., data/sequences)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output manifest path (default: <seq_dir>/manifest.parquet)"
    )
    args = parser.parse_args()

    seq_dir = args.seq_dir
    out_path = args.out or os.path.join(seq_dir, 'manifest.parquet')

    # Паттерн имени: actor_<actor>_seq_<seq>.npz
    pat = re.compile(r'actor_(?P<actor>\d+)_seq_(?P<seq>\d+)\.npz', re.IGNORECASE)

    rows = []
    for fname in os.listdir(seq_dir):
        if not fname.lower().endswith('.npz'):
            continue
        m = pat.match(fname)
        if not m:
            print(f"Skipping non-matching file: {fname}")
            continue
        actor = int(m.group('actor'))
        seqid = int(m.group('seq'))
        path = os.path.join(seq_dir, fname)
        rows.append({'filepath': path, 'actor_id': actor, 'seq_id': seqid})

    if not rows:
        print(f"No sequences found in '{seq_dir}'. Check directory and filename pattern.")
        return

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Manifest saved: {out_path}, sequences: {len(df)}")

if __name__ == '__main__':
    main()
