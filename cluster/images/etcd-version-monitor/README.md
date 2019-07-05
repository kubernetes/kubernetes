# etcd-version-monitor

This is a tool for exporting etcd metrics and supplementing them with etcd
server binary version and cluster version. These metrics are in
prometheus format and can be scraped by a prometheus server.
The metrics are exposed at the http://localhost:9101/metrics endpoint.

For etcd 3.1+, the
[go-grpc-prometheus](https://github.com/grpc-ecosystem/go-grpc-prometheus)
metrics format, which backward incompatibly replaces the 3.0 legacy grpc metric
format, is exposed in both the 3.1 format and in the 3.0. This preserves
backward compatibility.

For etcd 3.1+, the `--metrics=extensive` must be set on etcd for grpc request
latency metrics (`etcd_grpc_unary_requests_duration_seconds`) to be exposed.

**RUNNING THE TOOL**

To run this tool as a docker container:
- make build
- docker run --net=host -i -t k8s.gcr.io/etcd-version-monitor:test /etcd-version-monitor --logtostderr

To run this as a pod on the kubernetes cluster:
- Place the 'etcd-version-monitor.yaml' in the manifests directory of
  kubelet on the master machine.

*Note*: This tool has to run on the same machine as etcd, as communication
with etcd is over localhost.

**VERIFYING THE TOOL**

- Goto [http://localhost:9101/metrics](http://localhost:9101/metrics) in order to view the exported metrics.
- The metrics prefixed with "etcd_" are the ones of interest to us.
