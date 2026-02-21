# PyEvolve
High-performance cellular automata engine with CPU (Numba JIT) and GPU (CuPy convolution) backends, featuring real-time visualization, dynamic rule parsing, and benchmarking support.
# Overview
PyEvolveX is a high-performance cellular automata simulation engine built in Python. It supports both CPU and GPU backends, enabling real-time simulation of large-scale grids with dynamic rule configuration and interactive visualization.
The engine implements generalized Life-like cellular automata rules (e.g., B3/S23) and optimizes computation using:
- Numba JIT parallelization (CPU)
- CuPy + 2D convolution (GPU)
- Vectorized logical masking
- Toroidal boundary conditions
- Interactive Jupyter-based UI

Designed to explore performance engineering, parallel computing, and GPU acceleration in scientific simulations.
# Features
- Dual backend architecture (CPU / GPU)
- Numba JIT-compiled parallel kernel
- GPU-accelerated convolution-based neighbor aggregation
- Runtime rule parsing (B/S notation)
- Real-time animation via Matplotlib
- Interactive UI with ipywidgets
- Built-in benchmarking (ms/frame reporting)
- Supports large grids up to 1000x1000
# Architecture
## CPU Backend
- Uses @njit(parallel=True, fastmath=True)
- Parallelized neighbor computation via prange
- Explicit neighbor summation with toroidal wrapping
## GPU Backend
- Converts grid to CuPy array
- Computes neighbor counts using a 2D convolution kernel
- Applies vectorized logical masks for birth and survival rules
- Transfers result back to NumPy

This design abstracts computation behind a unified engine interface.
