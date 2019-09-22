## cuda_vector_add

This is a small CUDA application that performs a simple vector addition. Useful for testing CUDA support in Kubernetes.

## How to release:

```
# Build
$ cd $K8S_ROOT/test/images
$ make all WHAT=cuda-vector-add

# Push
$ cd $K8S_ROOT/test/images
$ make all-push WHAT=cuda-vector-add
```

## Version history:

1.0: build cuda-vector-add from CUDA 8.0.
2.0: build cuda-vector-add from CUDA 10.0
