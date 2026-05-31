#!/usr/bin/env python3
"""Parse Vivado OOC utilization (+ optional WNS) reports into one CSV.

Usage:  python3 parse_util.py [synth_sweep_dir] [out.csv]
Default: ./synth_sweep   resource_sweep.csv

Pulls Slice LUTs, DSPs, Block RAM Tile from each utilization_MAC_BITS_<n>.rpt.
If wns_MAC_BITS_<n>.txt exists, also reports WNS and an Fmax estimate.
Exit code 1 (and a GATE message) unless there are 5 complete rows.
"""
import sys, os, re, csv

DIR    = sys.argv[1] if len(sys.argv) > 1 else "./synth_sweep"
OUT    = sys.argv[2] if len(sys.argv) > 2 else "resource_sweep.csv"
PERIOD = 10.0  # ns -- MUST match CLK_PERIOD/clk_10ns.xdc used in the Tcl run
EXPECTED = 9   # number of MAC_BITS sweep points; match the MAC_LIST in the Tcl

# Utilization row label (trailing '*' stripped) -> CSV column name.
TARGETS = {
    "Slice LUTs":     "LUT",
    "DSPs":           "DSP",
    "Block RAM Tile": "BRAM",
}

def parse_util(path):
    """Return {'LUT':.., 'DSP':.., 'BRAM':..} from a report_utilization file."""
    found = {col: "" for col in TARGETS.values()}
    with open(path) as fh:
        for line in fh:
            if "|" not in line:
                continue
            cells = [c.strip() for c in line.split("|")]
            # Table rows look like: ['', 'Slice LUTs', '1234', '0', '53200', '2.32', '']
            if len(cells) < 4:
                continue
            name = cells[1].rstrip("*").strip()
            if name in TARGETS and cells[2].isdigit():
                found[TARGETS[name]] = cells[2]
    return found

def parse_wns(path):
    if not os.path.exists(path):
        return ""
    with open(path) as fh:
        toks = fh.read().split()
    return toks[0] if toks else ""

def fmax_from_wns(wns):
    try:
        achieved = PERIOD - float(wns)        # achieved min period (ns)
        return round(1000.0 / achieved, 1) if achieved > 0 else ""
    except ValueError:
        return ""

rows = []
pat = re.compile(r"utilization_MAC_BITS_(\d+)\.rpt$")
for fn in sorted(os.listdir(DIR)):
    m = pat.search(fn)
    if not m:
        continue
    mb  = int(m.group(1))
    u   = parse_util(os.path.join(DIR, fn))
    wns = parse_wns(os.path.join(DIR, "wns_MAC_BITS_%d.txt" % mb))
    rows.append({
        "MAC_BITS": mb, "LUT": u["LUT"], "DSP": u["DSP"], "BRAM": u["BRAM"],
        "WNS_ns": wns, "Fmax_MHz": fmax_from_wns(wns),
    })

rows.sort(key=lambda r: r["MAC_BITS"])

fields = ["MAC_BITS", "LUT", "DSP", "BRAM", "WNS_ns", "Fmax_MHz"]
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print("Wrote %s with %d rows:" % (OUT, len(rows)))
for r in rows:
    print("  ", r)

incomplete = [r["MAC_BITS"] for r in rows if "" in (r["LUT"], r["DSP"], r["BRAM"])]
if len(rows) != EXPECTED or incomplete:
    print("GATE NOT MET: %d rows (need %d); incomplete points: %s"
          % (len(rows), EXPECTED, incomplete))
    sys.exit(1)
print("GATE MET: %d complete utilization rows." % len(rows))