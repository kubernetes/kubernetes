## Heapster Debugging:

This is a collection of common issues faced by users and ways to debug them.

Depending on the deployment setup, the issue could be either with Heapster, cAdvisor, Kubernetes, or the monitoring backend.

### Heapster Core

#### Common Problems

* Some distros (including Debian) ship with memory accounting disabled by default. To enable memory and swap accounting on the nodes, follow [these instructions](https://docs.docker.com/installation/ubuntulinux/#memory-and-swap-accounting).

#### Debuging

There are 2 endpoints that can give you an insight into what is going on in Heapster:

* `/metrics` contains lots of metrics in Prometheus format that can indicate the root cause of Heapster problems. Example:
```
master:~$ curl 10.244.1.3:8082/metrics
# HELP heapster_exporter_duration_microseconds Time spent exporting data to sink in microseconds.
# TYPE heapster_exporter_duration_microseconds summary
heapster_exporter_duration_microseconds{exporter="InfluxDB Sink",quantile="0.5"} 3.497
heapster_exporter_duration_microseconds{exporter="InfluxDB Sink",quantile="0.9"} 5.296
heapster_exporter_duration_microseconds{exporter="InfluxDB Sink",quantile="0.99"} 5.296
heapster_exporter_duration_microseconds_sum{exporter="InfluxDB Sink"} 16698.508000000013
heapster_exporter_duration_microseconds_count{exporter="InfluxDB Sink"} 3089
heapster_exporter_duration_microseconds{exporter="Metric Sink",quantile="0.5"} 4.546
heapster_exporter_duration_microseconds{exporter="Metric Sink",quantile="0.9"} 7.632
heapster_exporter_duration_microseconds{exporter="Metric Sink",quantile="0.99"} 7.632
heapster_exporter_duration_microseconds_sum{exporter="Metric Sink"} 25597.190999999973
heapster_exporter_duration_microseconds_count{exporter="Metric Sink"} 3089
[...]
```
This endpoint is enabled for both metrics(Heapster) and events(Eventer).


* `/api/v1/model/debug/allkeys` has a list of all metrics sets that are processed inside Heapster. This can be usefull to check what is 
passed to your configured sinks Example:

```
master:~$ curl 10.244.1.3:8082/api/v1/model/debug/allkeys
[
  "namespace:kube-system/pod:kube-dns-v10-qey9d",
  "namespace:default/pod:resource-consumer-qcnzr",
  "namespace:default",
  "cluster",
  "node:kubernetes-minion-fpdd/container:kubelet",
  "namespace:kube-system/pod:kube-proxy-kubernetes-minion-fpdd/container:kube-proxy",
  "node:kubernetes-minion-j82g/container:system",
  "namespace:kube-system/pod:kube-proxy-kubernetes-minion-j82g/container:kube-proxy",
  "node:kubernetes-minion-j82g/container:docker-daemon",
  "namespace:kube-system/pod:monitoring-influxdb-grafana-v3-q3ozn/container:grafana",
  "namespace:kube-system/pod:kubernetes-dashboard-v1.0.0beta1-085ag",
  "node:kubernetes-minion-j82g/container:kubelet",
  "namespace:kube-system/pod:kube-dns-v10-qey9d/container:healthz",
  "node:kubernetes-minion-fhue",
  [...]
 ``` 
This is enabled for metrics only.

#### Extra Logging

Moreover additional logging can be enabled by setting an extra flag `--vmodule=*=4`. 
You can also enable a sink that writes all metrics or events to stdout with `--sink=log` added to command line parameters.
Both changes require restarting Heapster though.

### InfluxDB & Grafana

Ensure Influxdb is up and reachable. Heapster attempts to create a database by default, which will fail eventually after a fixed number of retries.
If the Grafana queries are stuck or slow, it is due to InfluxDB being unresponsive. Consider providing InfluxDB more compute resources (CPU and Memory).
The default database on Influxdb is 'k8s'. 
A `list series` query on 'k8s' database should list all the series being pushed by heapster. If you do not see any series, take a look at heapster logs.
