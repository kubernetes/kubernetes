## Tensorflow Official Wide Deep Model

The container image described here predicts the income using the census income dataset in Tensorflow. For more
information, see 
[https://github.com/tensorflow/models/tree/master/official/wide_deep](https://github.com/tensorflow/models/tree/master/official/wide_deep).
This image is used as a workload in in node performance testing. 

## How to release:
```
# Build
$ cd $K8S_ROOT/test/images
$ make all WHAT=node-perf/tf-wide-deep

# Push
$ cd $K8S_ROOT/test/images
$ make all-push WHAT=node-perf/tf-wide-deep
```
