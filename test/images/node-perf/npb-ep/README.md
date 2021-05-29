## NAS Parallel Benchmark Suite - Embarrassingly Parallel (EP) Benchmark

The container image described here runs the EP benchmark from the
[NAS parallel benchmark suite.](https://www.nas.nasa.gov/publications/npb.html)
This image is used as a workload in node performance testing.

## How to release:

```
# Build
$ cd $K8S_ROOT/test/images
$ make all WHAT=node-perf/npb-ep

# Push
$ cd $K8S_ROOT/test/images
$ make all-push WHAT=node-perf/npb-ep
```
