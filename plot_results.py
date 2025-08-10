#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, Tuple, List

# Use a non-interactive backend to work on headless boards
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def find_default_csv_path(user_path: str | None) -> str:
    if user_path:
        return user_path
    candidates = [
        "/home/orangepi/rk3588/matmul-bench/result.csv",
        "/home/orangepi/rk3588/matmul-bench/result_pytorch.csv",
        "/home/orangepi/rk3588/result.csv",
        "result.csv",
        "result_pytorch.csv",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("Could not find result.csv or result_pytorch.csv. Pass --csv <path>.")


def detect_csv_format(csv_path: str) -> str:
    """Detect if CSV is from C++ benchmark or PyTorch benchmark"""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        
        if "ac_native" in fieldnames and "b_native" in fieldnames:
            return "cpp"  # C++ benchmark format
        elif "device" in fieldnames:
            return "pytorch"  # PyTorch benchmark format
        else:
            raise ValueError("Unknown CSV format. Expected either C++ (ac_native, b_native) or PyTorch (device) format.")


def load_and_filter_cpp(
    csv_path: str,
    m: int,
    mm_type: str,
    ac_native: int,
    b_native: int,
    metric: str = "gops",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and filter data for C++ benchmark format"""
    # Aggregate metric by (k, n). If multiple counts exist, average them
    sums: Dict[Tuple[int, int], float] = defaultdict(float)
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    k_vals: set[int] = set()
    n_vals: set[int] = set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"m", "k", "n", "type", "ac_native", "b_native", metric}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            try:
                if int(row["m"]) != m:
                    continue
                if row["type"].strip() != mm_type:
                    continue
                if int(row["ac_native"]) != ac_native:
                    continue
                if int(row["b_native"]) != b_native:
                    continue

                k = int(row["k"])  # RKNN uses multiples of 16/32/64
                n = int(row["n"])  # Same for n
                val = float(row[metric])

                key = (k, n)
                sums[key] += val
                counts[key] += 1
                k_vals.add(k)
                n_vals.add(n)
            except Exception:
                # Skip malformed rows gracefully
                continue

    if not k_vals or not n_vals:
        raise ValueError(
            "No data after filtering. Check your filters or run the benchmark to generate result.csv."
        )

    k_sorted = sorted(k_vals)
    n_sorted = sorted(n_vals)
    X, Y = np.meshgrid(k_sorted, n_sorted)
    Z = np.full_like(X, np.nan, dtype=float)

    for (k, n), s in sums.items():
        c = counts[(k, n)]
        mean_val = s / max(c, 1)
        i = n_sorted.index(n)
        j = k_sorted.index(k)
        Z[i, j] = mean_val

    return X, Y, Z


def load_and_filter_pytorch(
    csv_path: str,
    m: int,
    mm_type: str,
    device: str,
    metric: str = "gops",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and filter data for PyTorch benchmark format"""
    # Aggregate metric by (k, n). If multiple counts exist, average them
    sums: Dict[Tuple[int, int], float] = defaultdict(float)
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    k_vals: set[int] = set()
    n_vals: set[int] = set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"m", "k", "n", "type", "device", metric}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            try:
                if int(row["m"]) != m:
                    continue
                if row["type"].strip() != mm_type:
                    continue
                if row["device"].strip() != device:
                    continue

                k = int(row["k"])
                n = int(row["n"])
                val = float(row[metric])

                key = (k, n)
                sums[key] += val
                counts[key] += 1
                k_vals.add(k)
                n_vals.add(n)
            except Exception:
                # Skip malformed rows gracefully
                continue

    if not k_vals or not n_vals:
        raise ValueError(
            "No data after filtering. Check your filters or run the benchmark to generate result.csv."
        )

    k_sorted = sorted(k_vals)
    n_sorted = sorted(n_vals)
    X, Y = np.meshgrid(k_sorted, n_sorted)
    Z = np.full_like(X, np.nan, dtype=float)

    for (k, n), s in sums.items():
        c = counts[(k, n)]
        mean_val = s / max(c, 1)
        i = n_sorted.index(n)
        j = k_sorted.index(k)
        Z[i, j] = mean_val

    return X, Y, Z


def load_and_filter(
    csv_path: str,
    m: int,
    mm_type: str,
    ac_native: int = None,
    b_native: int = None,
    device: str = None,
    metric: str = "gops",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Main load and filter function that handles both formats"""
    csv_format = detect_csv_format(csv_path)
    
    if csv_format == "cpp":
        if ac_native is None or b_native is None:
            raise ValueError("For C++ format, both ac_native and b_native must be specified")
        return load_and_filter_cpp(csv_path, m, mm_type, ac_native, b_native, metric)
    elif csv_format == "pytorch":
        if device is None:
            raise ValueError("For PyTorch format, device must be specified")
        return load_and_filter_pytorch(csv_path, m, mm_type, device, metric)
    else:
        raise ValueError(f"Unknown CSV format: {csv_format}")


def make_surface(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    title: str,
    out_path: str,
    zlabel: str,
) -> None:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Mask NaNs to avoid breaks in the surface
    Z_masked = np.ma.array(Z, mask=np.isnan(Z))
    surf = ax.plot_surface(
        X,
        Y,
        Z_masked,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
    )
    fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.08, label=zlabel)

    ax.set_xlabel("k")
    ax.set_ylabel("n")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(elev=30, azim=-60)

    # Improve tick readability for large values
    def _fmt(v):
        if v >= 1000:
            return f"{int(v/1000)}k"
        return str(int(v))

    # Choose more ticks along X (k) and Y (n)
    from typing import List as _List  # reuse imported List, but avoid shadowing

    def _choose_ticks(values: np.ndarray, max_labels: int = 6) -> _List[float]:
        vals = list(np.unique(values))
        if len(vals) <= max_labels:
            return vals
        lin = np.linspace(0, len(vals) - 1, num=max_labels)
        idxs = [int(round(i)) for i in lin]
        seen = set()
        picked = []
        for idx in idxs:
            if idx not in seen:
                seen.add(idx)
                picked.append(vals[idx])
        # Ensure endpoints are included
        if vals[0] not in picked:
            picked.insert(0, vals[0])
        if vals[-1] not in picked:
            picked.append(vals[-1])
        # Keep order stable according to original ordering
        order = {v: i for i, v in enumerate(vals)}
        picked = sorted(set(picked), key=lambda v: order[v])
        return picked

    k_ticks = _choose_ticks(X[0, :], max_labels=6)
    n_ticks = _choose_ticks(Y[:, 0], max_labels=6)
    ax.set_xticks(k_ticks)
    ax.set_xticklabels([_fmt(v) for v in k_ticks])
    ax.set_yticks(n_ticks)
    ax.set_yticklabels([_fmt(v) for v in n_ticks])

    # Highlight and annotate the peak value on the surface (max Z ignoring NaNs)
    try:
        # Use masked array filled with -inf so NaNs are ignored robustly
        Z_filled = np.ma.filled(Z_masked, -np.inf)
        if np.isfinite(np.max(Z_filled)):
            flat_index = int(np.argmax(Z_filled))
            i_max, j_max = np.unravel_index(flat_index, Z_filled.shape)
            x_peak = float(X[i_max, j_max])
            y_peak = float(Y[i_max, j_max])
            z_peak = float(Z_filled[i_max, j_max])
            # Draw a clearly visible marker and label; disable depthshade so it's not hidden
            ax.scatter(
                [x_peak],
                [y_peak],
                [z_peak + 1e-6],  # slight lift to avoid z-fighting
                color="red",
                s=64,
                depthshade=False,
                edgecolors="black",
                linewidths=0.5,
                zorder=10,
            )
            ax.text(
                x_peak,
                y_peak,
                z_peak,
                f"{zlabel}={z_peak:.2f}\n(k={x_peak:.0f}, n={y_peak:.0f})",
                color="red",
                fontsize=9,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.7),
                zorder=11,
            )
    except Exception:
        # Do not break plotting if annotation fails
        pass

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot RKNN matmul 3D surface from result.csv or result_pytorch.csv")
    parser.add_argument("--csv", type=str, default=None, help="Path to result.csv or result_pytorch.csv")
    parser.add_argument("--m", type=int, default=128, help="M dimension to filter")
    parser.add_argument(
        "--type",
        type=str,
        default="FLOAT16_MM_FLOAT16_TO_FLOAT32",
        help="Matmul type string from CSV (e.g., FLOAT16_MM_FLOAT16_TO_FLOAT32 for PyTorch)",
    )
    parser.add_argument("--ac", type=int, default=None, choices=[0, 1], help="AC layout native flag (C++ format only)")
    parser.add_argument("--bn", type=int, default=None, choices=[0, 1], help="B layout native flag (C++ format only)")
    parser.add_argument("--device", type=str, default=None, help="Device to filter (PyTorch format only, e.g., 'cpu', 'cuda')")
    parser.add_argument(
        "--metric",
        type=str,
        default="gops",
        choices=["gops", "time_ns"],
        help="Metric to plot on Z axis",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/home/orangepi/rk3588/matmul-bench/surface.png",
        help="Output image path",
    )

    args = parser.parse_args()
    csv_path = find_default_csv_path(args.csv)
    
    # Auto-detect format and set defaults
    csv_format = detect_csv_format(csv_path)
    print(f"Detected CSV format: {csv_format}")
    
    if csv_format == "cpp":
        if args.ac is None:
            args.ac = 1
            print(f"Using default ac_native={args.ac} for C++ format")
        if args.bn is None:
            args.bn = 1
            print(f"Using default b_native={args.bn} for C++ format")
        
        title = f"m == {args.m} && type == \"{args.type}\" && ac_native=={args.ac} && b_native=={args.bn}"
        
        X, Y, Z = load_and_filter(
            csv_path=csv_path,
            m=args.m,
            mm_type=args.type,
            ac_native=args.ac,
            b_native=args.bn,
            metric=args.metric,
        )
        
    elif csv_format == "pytorch":
        if args.device is None:
            args.device = "cpu"
            print(f"Using default device='{args.device}' for PyTorch format")
        
        title = f"m == {args.m} && type == \"{args.type}\" && device=={args.device}"
        
        X, Y, Z = load_and_filter(
            csv_path=csv_path,
            m=args.m,
            mm_type=args.type,
            device=args.device,
            metric=args.metric,
        )

    zlabel = "gops" if args.metric == "gops" else "time_ns"
    make_surface(X, Y, Z, title=title, out_path=args.out, zlabel=zlabel)

    print(f"Saved 3D surface to: {args.out}")


if __name__ == "__main__":
    main()

