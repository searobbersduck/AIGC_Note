```
cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
-- The CXX compiler identification is GNU 11.3.0
-- The CUDA compiler identification is NVIDIA 11.5.119
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Looking for C++ include pthread.h
-- Looking for C++ include pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
-- Found CUDA: /usr (found suitable version "11.5", minimum required is "10.2")
CUDA_VERSION 11.5 is greater or equal than 11.0, enable -DENABLE_BF16 flag
-- Could NOT find CUDNN (missing: CUDNN_LIBRARY_PATH CUDNN_INCLUDE_PATH)
-- Add DBUILD_CUTLASS_MOE, requires CUTLASS. Increases compilation time
-- Add DBUILD_CUTLASS_MIXED_GEMM, requires CUTLASS. Increases compilation time
-- Running submodule update to fetch cutlass
-- Add DBUILD_MULTI_GPU, requires MPI and NCCL
-- Could NOT find MPI_CXX (missing: MPI_CXX_LIB_NAMES MPI_CXX_HEADER_DIR MPI_CXX_WORKS)
CMake Error at /usr/share/cmake-3.22/Modules/FindPackageHandleStandardArgs.cmake:230 (message):
  Could NOT find MPI (missing: MPI_CXX_FOUND)
Call Stack (most recent call first):
  /usr/share/cmake-3.22/Modules/FindPackageHandleStandardArgs.cmake:594 (_FPHSA_FAILURE_MESSAGE)
  /usr/share/cmake-3.22/Modules/FindMPI.cmake:1830 (find_package_handle_standard_args)
  CMakeLists.txt:83 (find_package)


-- Configuring incomplete, errors occurred!
See also "/home/nvidia/weidong/code/demo/transformer/FasterTransformer/build/CMakeFiles/CMakeOutput.log".
See also "/home/nvidia/weidong/code/demo/transformer/FasterTransformer/build/CMakeFiles/CMakeError.log".

```