"""Load the FlyWire FAFB connectome (v783) + annotations into memory.

Usage:
    from load import fw
    fw.partners("KCab-c1")          # upstream + downstream of a cell type
    fw.upstream("mushroom body")    # everything feeding a neuropil/cell-type match
"""
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"


def _read(name, **kw):
    return pd.read_csv(DATA / f"{name}.csv.gz", **kw)


class FlyWire:
    def __init__(self):
        self.conn = _read("connections", dtype={"pre_root_id": np.uint64, "post_root_id": np.uint64})
        self.classif = _read("classification", dtype={"root_id": np.uint64})
        self.ctypes = _read("consolidated_cell_types", dtype={"root_id": np.uint64})
        self.neurons = _read("neurons", dtype={"root_id": np.uint64})
        self.stats = _read("cell_stats", dtype={"root_id": np.uint64})
        self.names = _read("names", dtype={"root_id": np.uint64})
        ann = pd.read_csv(
            DATA / "neuron_annotations.tsv", sep="\t", dtype={"root_id": "str"}
        )
        ann["root_id"] = ann["root_id"].astype(np.uint64)
        self.ann = ann

        # one master annotation frame: root_id -> labels
        self.meta = (
            self.classif.merge(self.ctypes, on="root_id", how="outer")
            .merge(self.names, on="root_id", how="outer")
            .merge(self.neurons, on="root_id", how="outer")
            .merge(self.stats, on="root_id", how="outer")
            .merge(self.ann, on="root_id", how="left", suffixes=("", "_ann"))
        )
        self.meta["label"] = self.meta["primary_type"].fillna(self.meta["name"])

        # edges with labels attached on both ends
        # (conn already carries per-edge nt_type -> don't re-merge the neuron-level one)
        self.edges = self.conn.merge(
            self.meta[["root_id", "label", "class", "super_class"]],
            left_on="pre_root_id", right_on="root_id", how="left",
        ).rename(columns={"label": "pre_label", "class": "pre_class", "super_class": "pre_super"})
        self.edges = self.edges.merge(
            self.meta[["root_id", "label", "class", "super_class"]],
            left_on="post_root_id", right_on="root_id", how="left",
        ).rename(columns={"label": "post_label", "class": "post_class", "super_class": "post_super"})

    # ---------- search ----------
    def find(self, pattern, col="label"):
        """Regex match on a meta column; returns matching rows."""
        rx = re.compile(pattern, re.I)
        m = self.meta[col].astype("string").fillna("").str.contains(rx)
        return self.meta[m]

    def ids(self, pattern, col="label"):
        return self.find(pattern, col)["root_id"].to_numpy()

    # ---------- connectivity ----------
    def partners(self, pattern, min_syn=3, col="label"):
        """Upstream + downstream partners of everything matching `pattern`."""
        ids = set(self.ids(pattern, col))
        e = self.edges[self.edges["syn_count"] >= min_syn]
        up = e[e["post_root_id"].isin(ids)].sort_values("syn_count", ascending=False)
        down = e[e["pre_root_id"].isin(ids)].sort_values("syn_count", ascending=False)
        return up[["pre_label", "pre_class", "post_label", "post_class", "neuropil", "syn_count", "nt_type"]], \
               down[["pre_label", "pre_class", "post_label", "post_class", "neuropil", "syn_count", "nt_type"]]

    def between(self, src_pattern, dst_pattern, min_syn=1):
        """Edges between two populations, aggregated by (pre_label, post_label)."""
        s = set(self.ids(src_pattern)); d = set(self.ids(dst_pattern))
        e = self.edges[self.edges["syn_count"] >= min_syn]
        e = e[e["pre_root_id"].isin(s) & e["post_root_id"].isin(d)]
        return (e.groupby(["pre_label", "post_label", "nt_type"], as_index=False)
                 .agg(synapses=("syn_count", "sum"), edges=("syn_count", "size"),
                      top_neuropil=("neuropil", lambda x: x.mode().iat[0]))
                 .sort_values("synapses", ascending=False))


fw = FlyWire()

if __name__ == "__main__":
    print(f"neurons annotated: {len(fw.meta):,}")
    print(f"connectivity rows: {len(fw.conn):,}")
    print(f"total synapses:    {fw.conn['syn_count'].sum():,}")
    print()
    print(fw.find("^KC").head(10)[["root_id", "label", "class", "nt_type"]].to_string(index=False))
