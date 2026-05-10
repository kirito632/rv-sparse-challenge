# RV-Sparse Coding Challenge

Implemented in ISO C with zero heap allocation inside the target function.

## Overview

This implementation completes the `sparse_multiply()` function with:

1. **CSR (Compressed Sparse Row)** extraction from a dense row-major matrix.
2. **Sparse matrix-vector multiplication**:

   ```
   y = A * x
   ```

## Key Design Decisions

### Zero Dynamic Memory Allocation

The function performs **no dynamic memory allocation**.

All required buffers (`values`, `col_indices`, `row_ptrs`, `y`) are provided by the caller.

### One-Pass CSR Construction

CSR data is built while scanning the matrix only once.

This reduces memory traffic and improves cache efficiency. The key insight is that `row_ptrs[i]` is simply the non-zero count *before* row `i` begins — the prefix sum is computed on the fly.

### Sparse Multiply

After CSR extraction, matrix-vector multiplication is performed directly using CSR traversal. Row start/end bounds are hoisted into local variables so the compiler can auto-vectorize the inner reduction loop without worrying about pointer aliasing on `row_ptrs[]`.

## Complexity

### Time Complexity

| Phase               | Complexity       |
|---------------------|------------------|
| CSR extraction      | O(rows × cols)   |
| SpMV (y = A × x)    | O(nnz)           |
| **Total**           | O(rows × cols + nnz) |

### Space Complexity

No dynamic allocation is performed. Additional space usage depends entirely on caller-provided buffers.

## Numerical Correctness

Results are validated against dense matrix multiplication using a mixed absolute/relative floating-point tolerance (`1e-7 + 1e-7 × |y_ref[i]|`).

## Build and Run

```bash
gcc -lm -o run challenge.c
./run
```

Expected output (100 randomized test iterations):

```
All tests passed! (100/100 iterations passed)
```

## Function Prototype

```c
void sparse_multiply(int rows,
                     int cols,
                     const double* A,
                     const double* x,
                     int* out_nnz,
                     double* values,
                     int* col_indices,
                     int* row_ptrs,
                     double* y);
```

| Parameter     | Description                                           |
|---------------|-------------------------------------------------------|
| `rows`        | Number of rows in matrix A                            |
| `cols`        | Number of columns in matrix A                         |
| `A`           | Dense row-major input matrix                          |
| `x`           | Dense input vector (length `cols`)                    |
| `out_nnz`     | Output: number of non-zero elements found             |
| `values`      | Output buffer for non-zero values (`capacity >= nnz`) |
| `col_indices` | Output buffer for column indices (`capacity >= nnz`)  |
| `row_ptrs`    | Output buffer for row pointers (length `rows + 1`)    |
| `y`           | Output buffer: result vector (length `rows`)          |
