## Toolbox image with perf installed

The image is installed with perf tools and includes scripts to start, kill perf, and
generate flame graph.

## How to release:
```
# Build
$ cd $K8S_ROOT/test/images
$ make all WHAT=profiling-toolbox

# Push
$ cd $K8S_ROOT/test/images
$ make all-push WHAT=profiling-toolbox
```
