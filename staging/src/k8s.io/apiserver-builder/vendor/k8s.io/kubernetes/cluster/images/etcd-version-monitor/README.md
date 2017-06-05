# etcd-version-monitor

This is a tool for exporting metrics related to etcd version, like etcd
server's binary version, cluster version, and counts of different kinds of
gRPC calls (which is a characteristic of v3), etc. These metrics are in
prometheus format and can be scraped by a prometheus server.
The metrics are exposed at the http://localhost:9101/metrics endpoint.

**RUNNING THE TOOL**

To run this tool as a docker container:
- make build
- docker run --net=host -i -t gcr.io/google_containers/etcd-version-monitor:test /etcd-version-monitor --logtostderr

To run this as a pod on the kubernetes cluster:
- Place the 'etcd-version-monitor.yaml' in the manifests directory of
  kubelet on the master machine.

*Note*: This tool has to run on the same machine as etcd, as communication
with etcd is over localhost.

**VERIFYING THE TOOL**

- Goto [http://localhost:9101/metrics](http://localhost:9101/metrics) in order to view the exported metrics.
- The metrics prefixed with "etcd_" are the ones of interest to us.
