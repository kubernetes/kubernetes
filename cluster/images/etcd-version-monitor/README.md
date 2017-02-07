# etcd-version-monitor

This is a tool for exporting metrics exposed by etcd endpoints that are not
in prometheus metrics format. The exported metrics would be in prometheus
metrics format and would be exposed by this tool at localhost:9101/metrics
http endpoint.

**RUNNING THE TOOL**

To run this tool as a binary:
- make
- ./bin/etcd-version-monitor

To run this tool as a docker container:
- make build-image
- docker run --net=host -i -t gcr.io/google_containers/etcd_version_monitor:test /bin/etcd-version-monitor
<br /> Optionally, change the 'IMAGE' variable inside the Makefile and run 'make push_image'
if you want to upload to and use image from your own docker registry.

To run this as a pod on the kubernetes cluster:
- Place the 'etcd-version-monitor.yaml' in the manifests directory of kubelet on
  the master machine.

*Note*: In all the 3 modes above, you have to ensure that this tool runs on
the same machine as etcd, as communication with etcd is over localhost.

**VERIFYING THE TOOL**

- Goto [http://localhost:9101/metrics](http://localhost:9101/metrics) in order to view the exported metrics.
<br />Note: [http://localhost:9101/](http://localhost:9101/) has a link to the above endpoint.
