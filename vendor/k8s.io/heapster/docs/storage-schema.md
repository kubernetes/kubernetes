## Metrics

Heapster exports the following metrics to its backends.

| Metric Name | Description |
|------------|-------------|
| cpu/limit | CPU hard limit in millicores. |
| cpu/node_capacity | Cpu capacity of a node. |
| cpu/node_reservation | Share of cpu that is reserved on the node. |
| cpu/node_utilization | CPU utilization as a share of node capacity. |
| cpu/request | CPU request (the guaranteed amount of resources) in millicores. |
| cpu/usage | Cumulative CPU usage on all cores. |
| cpu/usage_rate | CPU usage on all cores in millicores. |
| filesystem/usage | Total number of bytes consumed on a filesystem. |
| filesystem/limit | The total size of filesystem in bytes. |
| filesystem/available | The number of available bytes remaining in a the filesystem |
| memory/limit | Memory hard limit in bytes. |
| memory/major_page_faults | Number of major page faults. |
| memory/major_page_faults_rate | Number of major page faults per second. |
| memory/node_capacity | Memory capacity of a node. |
| memory/node_reservation | Share of memory that is reserved on the node. |
| memory/node_utilization | Memory utilization as a share of memory capacity. |
| memory/page_faults | Number of page faults. |
| memory/page_faults_rate | Number of page faults per second. |
| memory/request | Memory request (the guaranteed amount of resources) in bytes. |
| memory/usage | Total memory usage. |
| memory/working_set | Total working set usage. Working set is the memory being used and not easily dropped by the kernel. |
| network/rx | Cumulative number of bytes received over the network. |
| network/rx_errors | Cumulative number of errors while receiving over the network. |
| network/rx_errors_rate | Number of errors while receiving over the network per second. |
| network/rx_rate | Number of bytes received over the network per second. |
| network/tx | Cumulative number of bytes sent over the network |
| network/tx_errors | Cumulative number of errors while sending over the network |
| network/tx_errors_rate | Number of errors while sending over the network |
| network/tx_rate | Number of bytes sent over the network per second. |
| uptime  | Number of milliseconds since the container was started. |

All custom (aka application) metrics are prefixed with 'custom/'.

## Labels

Heapster tags each metric with the following labels.

| Label Name     | Description                                                                   |
|----------------|-------------------------------------------------------------------------------|
| pod_id         | Unique ID of a Pod                                                            |
| pod_name       | User-provided name of a Pod                                                   |
| pod_namespace  | The namespace of a Pod                                                        |
| container_base_image | Base image for the container |  
| container_name | User-provided name of the container or full cgroup name for system containers |
| host_id        | Cloud-provider specified or user specified Identifier of a node               | 
| hostname       | Hostname where the container ran                                              | 
| labels         | Comma-separated list of user-provided labels. Format is 'key:value'           |
| namespace_id   | UID of the namespace of a Pod                                                 |
| resource_id    | An unique identifier used to differentiate multiple metrics of the same type. e.x. Fs partitions under filesystem/usage | 

## Aggregates

The metrics are collected initally collected for nodes and containers and latter aggregated for pods, namespaces and clusters. 
Disk and network metrics are not available at container level (only at pod and node level). 

## Storage Schema

### InfluxDB

##### Default

Each metric translates to a separate 'series' in InfluxDB. Labels are stored as tags.
The metric name is not modified.

##### Using fields

If you want to use InfluxDB fields, you have to add `withfields=true` as parameter in InfluxDB sink URL.
(More information here: https://docs.influxdata.com/influxdb/v0.9/concepts/key_concepts/)

In that case, each metric translates to a separate in 'series' in InfluxDB. This means that some metrics are grouped in the same 'measurement'.
For example, we have the measurement 'cpu' with fields 'node_reservation', 'node_utilization', 'request', 'usage', 'usage_rate'.
Also, all labels are stored as tags.
Here the measurement list: cpu, filesystem, memory, network, uptime

Also, standard Grafana dashboard are not working with this new schema, you have to use [new dashboards](/grafana/dashboards/influxdb_withfields)

### Google Cloud Monitoring

Metrics mentioned above are stored along with corresponding labels as [custom metrics](https://cloud.google.com/monitoring/custom-metrics/) in Google Cloud Monitoring.

* Metrics are collected every 2 minutes by default and pushed with a 1 minute precision.
* Each metric has a custom metric prefix - `custom.cloudmonitoring.googleapis.com`
* Each metric is pushed with an additonal namespace prefix - `kubernetes.io`.
* GCM does not support visualizing cumulative metrics yet. To work around that, heapster exports an equivalent gauge metric for all cumulative metrics mentioned above.

  The gauge metrics use their parent cumulative metric name as the prefix, followed by a "_rate" suffix. 
   E.x.: "cpu/usage", which is cumulative, will have a corresponding gauge metric "cpu/usage_rate"
   NOTE: The gauge metrics will be deprecated as soon as GCM supports visualizing cumulative metrics.

TODO: Add a snapshot of all the metrics stored in GCM.

### Hawkular

Each metric is stored as separate timeseries (metric) in Hawkular-Metrics with tags being inherited from common ancestor type. The metric name is created with the following format: `containerName/podId/metricName` (`/` is separator). Each definition stores the labels as tags with following addons:

* All the Label descriptions are stored as label_description
* The ancestor metric name (such as cpu/usage) is stored under the tag `descriptor_name`
* To ease search, a tag with `group_id` stores the key `containerName/metricName` so each podId can be linked under a single timeseries if necessary.
* Units are stored under `units` tag
* If labelToTenant parameter is given, any metric with the label will use this label's value as the target tenant. If the metric doesn't have the label defined, default tenant is used.

At the start, all the definitions are fetched from the Hawkular-Metrics tenant and filtered to cache only the Heapster metrics. It is recommended to use a separate tenant for Heapster information if you have lots of metrics from other systems, but not required.

The Hawkular-Metrics instance can be a standalone installation of Hawkular-Metrics or the full installation of Hawkular. 
