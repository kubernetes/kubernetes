Go Bindings for NVML
--------------------

[NVML or NVIDIA Management
Library](https://developer.nvidia.com/nvidia-management-library-nvml) is a
C-based API that can be used for monitoring NVIDIA GPU devices. It's closed
source but can be downloaded as part of the [GPU Deployment
Kit](https://developer.nvidia.com/gpu-deployment-kit).

The [NVML API
Reference](http://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html)
describe various methods that are available as part of NVML.

The `nvml.h` file is included in this repository so that we don't depend on
the presence of NVML in the build environment.

The `bindings.go` file is the cgo bridge which calls the NVML functions. The
cgo preamble in `bindings.go` uses `dlopen` to dynamically load NVML and makes
its functions available.
