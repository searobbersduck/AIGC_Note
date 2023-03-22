# []

## [CUTLASS 3.0 GEMM procedural names](https://github.com/NVIDIA/cutlass/blob/master/media/docs/profiler.md#cutlass-30-gemm-procedural-names)

CUTLASS 3.0 introduces a new naming convention for GEMMs used by the profiler targeting the NVIDIA
Hopper architecture and beyond so as to indicate new features of the kernel within the name
(e.g., the cluster shape).

To best illustrate this naming convention, we will walk through the meaning of each of the components
in a GEMM kernel used by the profiler:
```
cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f32_128x128x64_2x1x1_0_ntn_align8
```

The components within this name are as follows:

* `cutlass3x`: indicates that the kernel was generated through the CUTLASS 3.0 API
* `sm90`: indicates that the kernel targets NVIDIA GPUs with compute capability 90
* `tensorop`: indicates that the kernel makes use of NVIDIA Tensor Cores
(as opposed to `simt`, which indicates the use of "CUDA cores")
* `s`: indicates that the Tensor Core instruction being used accumulates in single precision
(as opposed to `h`, which indicates half precision)
* `64x128x16gemm`: indicates that the shape of the Tensor Core instruction being used (MxNxK) is 64x128x16
* `f16_f16_f32_f16`: indicates that the data types for operands A, B, and C are each `f16`
(half precision) and that accumulation is performed using `f32` (single precision)
* `128x128x64`: indicates that the thread block shape used in the GEMM (MxNxK) is 128x128x64
* `2x1x1`: indicates that the cluster shape being used is 2x1x1
* `0`: indicates that the kernel uses the CollectiveBuilder's automatic stage calculation to determine the
number of pipeline stages in the kernel. Note that `0` does not mean that no stages are used. A nonzero value indicates that automatic stage calculation is not performed and indicates the number of pipeline stages to be used.
This 0 is only added to the kernel's procedural name, the profiler will still report the actual stage count
when printing the kernel argument details (`--stages=N`) and kernel discovery will still support filtering through the `--stages` argument.
* `ntn`: indicates that the layouts for operands A, B, and C are column major ("n"; non-transposed),
row major ("t"; transposed), and column major, respectively.
* `align8`: indicates that the maximum alignment between operands A and B is 8.

Note that in some special cases where the input A/B types do not match that of the MMA
instruction's, the MMA facing input type is added to the instruction string as well.

```
cutlass3x_sm90_tensorop_s64x128x8tf32gemm_f32_f32_f32_f32_128x128x32_2x1x1_0_tnn_align4
```

* `s64x128x8tf32gemm`: indicates that the MMA consumes inputs in `tf32` format, and therefore
the kernel performs rounding of the `f32` values in global memory while loading them into shared memory.

