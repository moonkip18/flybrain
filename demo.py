"""Demo: the real olfactory -> mushroom body circuit, straight from FlyWire v783."""
from load import fw

print("=== 1. olfactory projection neurons: who feeds them (ORN -> PN) ===")
up, down = fw.partners("_lPN$|_adPN$", min_syn=10)
print(up.head(8).to_string(index=False))
print()

print("=== 2. PNs -> Kenyon cells (the learning input, aggregated) ===")
p2k = fw.between("_lPN$|_adPN$", "^KC", min_syn=3)
print(p2k.head(10).to_string(index=False))
print(f"   ... {len(p2k)} PN-type -> KC-type connections total")
print()

print("=== 3. Kenyon cells -> MBONs (the learned readout) ===")
k2m = fw.between("^KC", "MBON", min_syn=3)
print(k2m.head(10).to_string(index=False))
print()

print("=== 4. the giant fiber escape circuit ===")
gf = fw.find("giant fiber|GF_", col="label")
cols = [c for c in ["root_id", "label", "class", "nt_type"] if c in gf.columns]
print(gf[cols].head(8).to_string(index=False))
