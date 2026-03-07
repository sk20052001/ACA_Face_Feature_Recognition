import csv
import os
from datetime import datetime

SUMMARY_PATHS = [
    ("serial",          "results/run_summary.csv"),
    ("omp_e2e_1",       "results_omp_1/run_summary.csv"),
    ("omp_e2e_2",       "results_omp_2/run_summary.csv"),
    ("omp_e2e_4",       "results_omp_4/run_summary.csv"),
    ("omp_e2e_8",       "results_omp_8/run_summary.csv"),
    ("preload_1",       "results_preload_1/run_summary.csv"),
    ("preload_2",       "results_preload_2/run_summary.csv"),
    ("preload_4",       "results_preload_4/run_summary.csv"),
    ("preload_8",       "results_preload_8/run_summary.csv"),
]

def read_summary(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return rows[0]

def fnum(x):
    if x is None: return None
    x = str(x).strip()
    if x == "": return None
    return float(x)

def main():
    lines = []
    lines.append("Serial vs Parallel Analysis (500 images) — Batch-time Based\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")

    data = {}
    for name, path in SUMMARY_PATHS:
        if not os.path.exists(path):
            data[name] = None
            continue
        data[name] = read_summary(path)

    if data["serial"] is None:
        raise RuntimeError("Missing results/run_summary.csv for serial")

    serial_batch_total = fnum(data["serial"].get("batch_total_ms"))
    serial_images = data["serial"].get("images")

    lines.append("=== End-to-End Speedup (Batch Total Time) ===\n")
    lines.append(f"Serial images: {serial_images}\n")
    lines.append(f"{'Config':<14} {'Batch(ms)':>12} {'Speedup':>10}\n")
    lines.append("-"*40 + "\n")
    lines.append(f"{'serial':<14} {serial_batch_total:>12.2f} {1.00:>10.2f}\n")

    for key in ["omp_e2e_1","omp_e2e_2","omp_e2e_4","omp_e2e_8"]:
        row = data.get(key)
        if row is None:
            lines.append(f"{key:<14} MISSING\n")
            continue
        bt = fnum(row.get("batch_total_ms"))
        sp = serial_batch_total / bt if bt else None
        lines.append(f"{key:<14} {bt:>12.2f} {sp:>10.2f}\n")

    lines.append("\n")

    # Compute-only speedup (preload)
    lines.append("=== Compute-only Speedup (Batch Compute Time) ===\n")
    lines.append(f"{'Config':<14} {'Compute(ms)':>12} {'Speedup':>10}\n")
    lines.append("-"*40 + "\n")

    # For compute-only baseline, use serial compute time estimated as:
    # serial_batch_compute = sum per-image compute or (optional) you can add it later to serial run_summary.
    # For now, we can approximate serial compute batch time using serial_compute_times.csv if you want.
    # BUT since we didn't log it into summary yet, we will not compute speedup unless present.
    # We'll compute speedup relative to preload_1 as a consistent compute-only baseline.

    base = data.get("preload_1")
    if base is None:
        lines.append("preload_1 MISSING (cannot compute compute-only speedup)\n")
    else:
        base_ct = fnum(base.get("batch_compute_ms"))
        lines.append(f"{'preload_1':<14} {base_ct:>12.2f} {1.00:>10.2f}\n")
        for key in ["preload_2","preload_4","preload_8"]:
            row = data.get(key)
            if row is None:
                lines.append(f"{key:<14} MISSING\n")
                continue
            ct = fnum(row.get("batch_compute_ms"))
            sp = base_ct / ct if ct else None
            lines.append(f"{key:<14} {ct:>12.2f} {sp:>10.2f}\n")

    out_path = "results_serVpar_500.txt"
    with open(out_path, "w") as f:
        f.writelines(lines)

    print("".join(lines))
    print(f"\nSaved summary to {out_path}")

if __name__ == "__main__":
    main()