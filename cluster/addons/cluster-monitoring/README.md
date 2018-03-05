# Kubernetes Monitoring

[Heapster](https://github.com/kubernetes/heapster) enables monitoring and performance analysis in Kubernetes Clusters.
Heapster collects signals from kubelets and the api server, processes them, and exports them via REST APIs or to a configurable timeseries storage backend.

More details can be found in [Monitoring user guide](http://kubernetes.io/docs/user-guide/monitoring/).

## Troubleshooting

Heapster supports up to 30 pods per cluster node. In clusters where there are more running pods, Heapster may be throttled or fail with OOM error. Starting with Kubernetes 1.9.2, Heapster resource requirements may be overwritten manually. [Learn more about Addon Resizer configuration](https://github.com/kubernetes/autoscaler/tree/master/addon-resizer#addon-resizer-configuration)

### Important notices

Decreasing resource requirements for cluster addons may cause system instability. The effects may include (but are not limited to):
  - Metrics not being exported
  - Horizontal Pod Autoscaler not working
  - `kubectl top` not working

Overwritten configuration persists through cluster updates, therefore may cause all effects above after a cluster update.

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/cluster-monitoring/README.md?pixel)]()
