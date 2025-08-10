#!/usr/bin/env python3
"""
PyTorch Matrix Multiplication Benchmark Script

This script benchmarks matrix multiplication performance on CPU using PyTorch,
generating data in the same format as the C++ benchmark for compatibility
with the plotting script.
"""

import csv
import time
import torch
import numpy as np
from typing import List, Tuple
import argparse
import os


def benchmark_matmul(
    m: int, k: int, n: int, 
    dtype: torch.dtype = torch.float16,
    num_warmup: int = 10, 
    num_runs: int = 100
) -> Tuple[float, float]:
    """
    Benchmark matrix multiplication A @ B where A is (m, k) and B is (k, n)
    
    Returns:
        Tuple of (time_ns, gops) where gops is Giga Operations Per Second
    """
    # Create input tensors
    A = torch.randn(m, k, dtype=dtype, device='cpu')
    B = torch.randn(k, n, dtype=dtype, device='cpu')
    
    # Warmup runs
    for _ in range(num_warmup):
        _ = torch.mm(A, B)
    
    # Synchronize CPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter_ns()
        _ = torch.mm(A, B)
        end = time.perf_counter_ns()
        times.append(end - start)
    
    # Calculate statistics
    time_ns = np.mean(times)
    
    # Calculate GOPS: 2*m*k*n operations (multiply-add) per second
    ops = 2 * m * k * n
    gops = (ops / time_ns) * 1e9 / 1e9  # Convert to GOPS
    
    return time_ns, gops


def get_matmul_type(dtype: torch.dtype) -> str:
    """Convert PyTorch dtype to matmul type string"""
    if dtype == torch.float16:
        return "FLOAT16_MM_FLOAT16_TO_FLOAT32"
    elif dtype == torch.float32:
        return "FLOAT32_MM_FLOAT32_TO_FLOAT32"
    elif dtype == torch.int8:
        return "INT8_MM_INT8_TO_INT32"
    else:
        return f"{dtype}_MM_{dtype}_TO_{dtype}"


def run_benchmark_suite(
    m_values: List[int],
    k_values: List[int], 
    n_values: List[int],
    dtype: torch.dtype = torch.float16,
    output_file: str = "result_pytorch.csv"
) -> None:
    """
    Run the full benchmark suite and save results to CSV
    """
    print(f"Starting PyTorch matrix multiplication benchmark on CPU")
    print(f"Matrix dimensions: m={m_values}, k={k_values}, n={n_values}")
    print(f"Data type: {dtype}")
    print(f"Output file: {output_file}")
    
    # Prepare CSV file
    fieldnames = ['m', 'k', 'n', 'type', 'device', 'time_ns', 'gops', 'dtype']
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        total_combinations = len(m_values) * len(k_values) * len(n_values)
        current = 0
        
        for m in m_values:
            for k in k_values:
                for n in n_values:
                    current += 1
                    print(f"Progress: {current}/{total_combinations} - Testing m={m}, k={k}, n={n}")
                    
                    try:
                        time_ns, gops = benchmark_matmul(m, k, n, dtype)
                        
                        row = {
                            'm': m,
                            'k': k, 
                            'n': n,
                            'type': get_matmul_type(dtype),
                            'device': 'cpu',
                            'time_ns': time_ns,
                            'gops': gops,
                            'dtype': str(dtype)
                        }
                        
                        writer.writerow(row)
                        csvfile.flush()  # Ensure data is written immediately
                        
                        print(f"  Result: {gops:.2f} GOPS, {time_ns:.0f} ns")
                        
                    except Exception as e:
                        print(f"  Error: {e}")
                        continue
    
    print(f"Benchmark completed. Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Matrix Multiplication Benchmark")
    parser.add_argument("--m", type=int, nargs="+", default=[128], 
                       help="M dimension values to test")
    parser.add_argument("--k", type=int, nargs="+", 
                       default=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
                       help="K dimension values to test")
    parser.add_argument("--n", type=int, nargs="+",
                       default=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
                       help="N dimension values to test")
    parser.add_argument("--dtype", type=str, default="float16", 
                       choices=["float16", "float32", "int8"],
                       help="Data type for matrices")
    parser.add_argument("--output", type=str, default="result_pytorch.csv",
                       help="Output CSV file path")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=100,
                       help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "int8": torch.int8
    }
    dtype = dtype_map[args.dtype]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # Run benchmark
    run_benchmark_suite(
        m_values=args.m,
        k_values=args.k,
        n_values=args.n,
        dtype=dtype,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
