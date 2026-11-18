"""Extract real, browser-sized subcircuits from FlyWire v783 for flymind.html.

Everything is indexed by root_id (individual neurons); type labels are display-only.
Output: flybrain/circuit.json
"""
import json
import re
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from load import fw  # noqa: E402

GLOM = ["DA1", "VA1v", "VA1d", "VA6", "DL1", "DL3", "DM1", "DM2", "DM3", "DM6", "VM2", "VM5d"]
GLOM_RX = "^ORN_(" + "|".join(GLOM) + ")$"
PN_RX = "^(" + "|".join(GLOM) + ")_(lPN|adPN|mPN)$"

# ---------- antennal lobe ----------
pn_rows = fw.find(PN_RX, col="label")
pn_list = []          # [{id, glom, type}]
pn_index = {}
for _, r in pn_rows.iterrows():
    m = re.match(PN_RX, r["label"])
    if m:
        rid = int(r["root_id"])
        if rid not in pn_index:
            pn_index[rid] = len(pn_list)
            pn_list.append({"id": rid, "glom": m.group(1), "type": r["label"]})
pn_ids = set(pn_index)

orns = fw.find(GLOM_RX, col="label")
orn_by_glom = {g: {int(i) for i in orns[orns["label"] == f"ORN_{g}"]["root_id"]} for g in GLOM}

glom_out = []
for g in GLOM:
    pn_g = {p["id"] for p in pn_list if p["glom"] == g}
    sub = fw.edges[fw.edges["pre_root_id"].isin(orn_by_glom[g]) & fw.edges["post_root_id"].isin(pn_g)]
    glom_out.append({
        "name": g,
        "orn_count": len(orn_by_glom[g]),
        "pn_count": len(pn_g),
        "syn_total": int(sub["syn_count"].sum()),
        "max_edge": int(sub["syn_count"].max()) if len(sub) else 0,
    })

# ---------- PN -> KC (individual KCs, real edges) ----------
kc = fw.find("^KC", col="label")
kc = kc[kc["class"] == "Kenyon_Cell"]
kc_type_of = {int(r["root_id"]): r["label"] for _, r in kc.iterrows()}
kc_ids_all = set(kc_type_of)

pnkc = fw.edges[fw.edges["pre_root_id"].isin(pn_ids) & fw.edges["post_root_id"].isin(kc_ids_all)]
inputs_per_kc = pnkc.groupby("post_root_id").agg(n=("pre_root_id", "nunique"), syn=("syn_count", "sum"))
good = inputs_per_kc[(inputs_per_kc["n"] >= 3) & (inputs_per_kc["syn"] >= 30)]
good = good.sort_values("syn", ascending=False)
KCAP = 300
chosen = set(good.head(KCAP).index.astype(int))

kc_list = sorted(chosen)                       # deterministic order
kc_index = {rid: i for i, rid in enumerate(kc_list)}
kc_types = [kc_type_of[rid] for rid in kc_list]

pn2kc = []
for pre, post, syn in pnkc[pnkc["post_root_id"].isin(chosen)][["pre_root_id", "post_root_id", "syn_count"]].itertuples(index=False):
    pn2kc.append([pn_index[int(pre)], kc_index[int(post)], int(syn)])

# per-KC threshold: ~30% of that KC's total real input drive -> sparse response
kc_in = np.zeros(len(kc_list))
for _, ki, s in pn2kc:
    kc_in[ki] += s
theta = (kc_in * 0.30 + 6.0).round(2).tolist()

# ---------- KC -> MBON ----------
mbon = fw.find("^MBON", col="label")
mbon_list = []
mbon_index = {}
for _, r in mbon.iterrows():
    rid = int(r["root_id"])
    if rid not in mbon_index:
        mbon_index[rid] = len(mbon_list)
        mbon_list.append({"id": rid, "type": r["label"]})
mbon_ids = set(mbon_index)
k2m = fw.edges[fw.edges["pre_root_id"].isin(chosen) & fw.edges["post_root_id"].isin(mbon_ids)]
kc2mbon = []
for pre, post, syn in k2m[["pre_root_id", "post_root_id", "syn_count"]].itertuples(index=False):
    kc2mbon.append([kc_index[int(pre)], mbon_index[int(post)], int(syn)])
mbon_types = [m["type"] for m in mbon_list]

# ---------- DANs ----------
dans = fw.meta[fw.meta["cell_type"].astype("string").str.contains("PPL1|PAM", case=False, na=False)]
dan_ids = {int(i) for i in dans["root_id"]}
dan_type_of = {int(r["root_id"]): r["cell_type"] for _, r in dans.iterrows()}
d2m = fw.edges[fw.edges["pre_root_id"].isin(dan_ids) & fw.edges["post_root_id"].isin(mbon_ids)]
dan_agg = (
    d2m.groupby("pre_root_id")
    .agg(syn=("syn_count", "sum"), targets=("post_root_id", "nunique"))
    .sort_values("syn", ascending=False)
    .head(20)
)
dan_out = [
    {"type": dan_type_of[int(rid)], "syn": int(row["syn"]), "mbon_targets": int(row["targets"])}
    for rid, row in dan_agg.iterrows()
]

# ---------- giant fiber ----------
gf_edges = fw.edges[fw.edges["pre_root_id"].isin(fw.ids("Giant Fiber", col="hemibrain_type"))]
gf_out = (
    gf_edges.groupby("post_label", as_index=False).agg(syn=("syn_count", "sum"))
    .sort_values("syn", ascending=False).head(12)
)
gf_outputs = [{"label": r["post_label"], "syn": int(r["syn"])} for _, r in gf_out.iterrows()]

# ---------- real soma positions (nm -> um) ----------
from load import DATA
import pandas as pd  # noqa: E402
coords = pd.read_csv(DATA / "coordinates.csv.gz", dtype={"root_id": np.uint64})
pos_map = {}
for rid, s in zip(coords["root_id"], coords["position"]):
    if rid not in pos_map:
        v = np.array([float(t) for t in str(s).strip("[]").split()], dtype=float) / 1000.0
        pos_map[int(rid)] = [round(float(v[0]), 1), round(float(v[1]), 1), round(float(v[2]), 1)]

def pos_or_none(rid):
    return pos_map.get(int(rid))

# ---------- ALL neurons at real positions (the honest full-brain point cloud) ----------
all_pn_set = {p["id"] for p in pn_list}
all_kc_set = set(kc_list)
all_mbon_set = {m["id"] for m in mbon_list}
all_dan_set = {int(r) for r in dan_agg.index}
gf_real = [int(i) for i in fw.ids("Giant Fiber", col="hemibrain_type")]

all_pts = []  # [ [x,y,z] in um, pop, pop_idx ]  pop: 0 backdrop, 1 pn, 2 kc, 3 mbon, 4 dan, 5 gf
for rid, p in pos_map.items():
    if rid in all_pn_set:
        pop, pop_idx = 1, pn_index[rid]
    elif rid in all_kc_set:
        pop, pop_idx = 2, kc_index[rid]
    elif rid in all_mbon_set:
        pop, pop_idx = 3, mbon_index[rid]
    elif rid in all_dan_set:
        di = 0
        for k, rr in dan_agg.reset_index().iterrows():
            if int(rr["pre_root_id"]) == rid:
                di = k
                break
        pop, pop_idx = 4, min(di, 19)  # danAct has 20 slots
    elif rid in gf_real:
        pop, pop_idx = 5, 0
    else:
        pop, pop_idx = 0, 0
    all_pts.append([p, pop, pop_idx])

# ---------- packed binary positions for fast load (Int16 um, base64) ----------
import base64
import struct

cx0 = sum(p[0][0] for p in all_pts) / len(all_pts)
cy0 = sum(p[0][1] for p in all_pts) / len(all_pts)
cz0 = sum(p[0][2] for p in all_pts) / len(all_pts)
pos_bytes = bytearray()
pop_bytes = bytearray()
idx_bytes = bytearray()
for p, pop, pop_idx in all_pts:
    pos_bytes += struct.pack("<hhh", int(round(p[0] - cx0)), int(round(p[1] - cy0)), int(round(p[2] - cz0)))
    pop_bytes.append(pop)
    idx_bytes += struct.pack("<H", pop_idx)
packed = {
    "posB64": base64.b64encode(bytes(pos_bytes)).decode(),
    "popB64": base64.b64encode(bytes(pop_bytes)).decode(),
    "idxB64": base64.b64encode(bytes(idx_bytes)).decode(),
    "center": [round(cx0, 1), round(cy0, 1), round(cz0, 1)],
    "count": len(all_pts),
}
print("packed positions:", len(pos_bytes), "bytes raw,", len(packed["posB64"]), "b64")
print("all-neuron point cloud:", len(all_pts))

dans_pts = []
dan_syn = d2m.groupby("pre_root_id").agg(syn=("syn_count", "sum")).sort_values("syn", ascending=False)
for rid, row in dan_syn.head(24).iterrows():
    dans_pts.append({
        "id": int(rid),
        "type": dan_type_of.get(int(rid), "DAN"),
        "syn": int(row["syn"]),
        "p": pos_or_none(rid),
    })
gf_ids = [int(i) for i in fw.ids("Giant Fiber", col="hemibrain_type")]
gf_pts = [{"id": rid, "p": pos_or_none(rid)} for rid in gf_ids]

# ---------- central complex ----------
cx_counts = {}
for t in ["EPG", "PEN_a", "PEN_b", "D7", "PFR"]:
    n = fw.meta[fw.meta["cell_type"].astype("string") == t]
    if len(n):
        cx_counts[t] = int(len(n))
epg_ids = set(fw.ids("^EPG$", col="cell_type"))
pen_ids = set(fw.ids("^PEN_", col="cell_type"))
epg_pen_syn = int(fw.edges[fw.edges["pre_root_id"].isin(epg_ids) & fw.edges["post_root_id"].isin(pen_ids)]["syn_count"].sum())

out = {
    "source": "FlyWire FAFB v783 + flywire_annotations v3.1.0",
    "totals": {
        "neurons": int(len(fw.meta)),
        "edges": int(len(fw.conn)),
        "synapses": int(fw.conn["syn_count"].sum()),
    },
    "glomeruli": glom_out,
    "odors": {
        "fruit": {"DM1": 0.95, "DM2": 0.7, "VA1v": 0.6, "DL1": 0.5, "DL3": 0.35, "VM2": 0.3},
        "co2": {"VM2": 1.0, "DL3": 0.25},
        "cva": {"DA1": 1.0, "VA6": 0.2},
    },
    "pns": pn_list,
    "kcs": [{"id": rid, "type": kc_types[i]} for i, rid in enumerate(kc_list)],
    "mbons": mbon_list,
    "pn2kc": pn2kc,
    "kc2mbon": kc2mbon,
    "kc_theta": theta,
    "dans": dan_out,
    "gf_outputs": gf_outputs,
    "pos": {
        "pns": [pos_or_none(p["id"]) for p in pn_list],
        "kcs": [pos_or_none(rid) for rid in kc_list],
        "mbons": [pos_or_none(m["id"]) for m in mbon_list],
        "dans": dans_pts,
        "gf": gf_pts,
    },
    "packed": packed,
    "cx": {"counts": cx_counts, "epg_to_pen_syn": epg_pen_syn},
    "convergence": {
        "mean_pn_per_kc": float(inputs_per_kc.loc[list(chosen), "n"].mean()),
        "mean_kc_per_mbon": float(k2m.groupby("post_root_id").size().mean()),
        "kc_fanout_syn_mean": float(k2m.groupby("pre_root_id")["syn_count"].sum().mean()),
    },
}
with open("circuit.json", "w") as f:
    json.dump(out, f)

print("glom:", [(g["name"], g["orn_count"], g["syn_total"]) for g in glom_out][:6])
print("PNs:", len(pn_list), " KCs:", len(kc_list), " MBONs:", len(mbon_list))
print("PN->KC edges:", len(pn2kc), " KC->MBON edges:", len(kc2mbon), " DANs:", len(dan_out))
print("mean PNs/KC:", out["convergence"]["mean_pn_per_kc"], " mean KCs/MBON:", out["convergence"]["mean_kc_per_mbon"])
print("cx:", cx_counts, " EPG->PEN syn:", epg_pen_syn)
print("GF:", gf_outputs[:5])
