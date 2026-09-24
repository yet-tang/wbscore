#!/usr/bin/env python3
"""
Combine real training data (9179) with synthetic negatives (2700) into
a balanced training set for v4 retraining.

Output: data/training_set_v4.parquet (11879 rows)
"""
import pyarrow.parquet as pq
from pathlib import Path

WORKSPACE = Path(__file__).parent
REAL = WORKSPACE / "data" / "training_set.parquet"
SYNTH = WORKSPACE / "data" / "synthetic_negatives.parquet"
OUT = WORKSPACE / "data" / "training_set_v4.parquet"

real_rows = pq.read_table(REAL).to_pylist()
synth_rows = pq.read_table(SYNTH).to_pylist()
print(f"real: {len(real_rows)}, synth: {len(synth_rows)}")

combined = real_rows + synth_rows
print(f"combined: {len(combined)}")

table = pq.read_table(REAL).combine_chunks()  # get schema
out_schema = table.schema

# Convert dict to fit real schema (synth may have extra fields)
filtered = []
for r in combined:
    out = {k: r.get(k) for k in out_schema.names}
    filtered.append(out)

new_table = pq.read_table(REAL).from_pylist(filtered) if False else None
import pyarrow as pa
new_table = pa.Table.from_pylist(filtered, schema=out_schema)
pq.write_table(new_table, OUT)
print(f"saved {OUT}")