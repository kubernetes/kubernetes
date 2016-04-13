## Metrics

Heapster exports the following metrics to its backends.

| Metric Name              | Description                                                                                        | Type       | Units        | Supported Since |
|--------------------------|----------------------------------------------------------------------------------------------------|------------|--------------|-----------------|
| uptime                   | Number of millisecond since the container was started                                              | Cumulative | Milliseconds | v0.9            |
| cpu/usage                | Cumulative CPU usage on all cores                                                                  | Cumulative | Nanoseconds  | v0.9            |
| memory/usage             | Total memory usage                                                                                 | Gauge      | Bytes        | v0.9            |
| memory/working_set       | Total working set usage. Working set is the memory being used and not easily dropped by the Kernel | Gauge      | Bytes        | v0.9            |
| memory/page_faults       | Total number of page faults                                                                        | Cumulative | Count        | v0.9            |
| memory/major_page_faults | Number of major page faults                                                                        | Cumulative | Count        | HEAD            |
| network/rx               | Cumulative number of bytes received over the network                                               | Cumulative | Bytes        | v0.9            |
| network/rx_errors        | Cumulative number of errors while receiving over the network                                       | Cumulative | Count        | v0.9            |
| network/tx               | Cumulative number of bytes sent over the network                                                   | Cumulative | Bytes        | v0.9            |
| network/tx_errors        | Cumulative number of errors while sending over the network                                         | Cumulative | Count        | v0.9            |
| filesystem/usage         | Total number of bytes used on a filesystem identified by label 'resource_id'                       | Gauge      | Bytes        | v0.11.0         |

*Note: Gauge refers to instantaneous metrics*

## Labels

Heapster tags each metric with the following labels.

| Label Name     | Description                                                                   | Supported Since | Kubernetes specific |
|----------------|-------------------------------------------------------------------------------|-----------------|---------------------|
| pod_id         | Unique ID of a Pod                                                            | v0.9            | Yes                 |
| pod_name       | User-provided name of a Pod                                                   | v0.13           | Yes                 |
| pod_namespace  | The namespace of a Pod                                                        | v0.10           | Yes                 |
| container_name | User-provided name of the container or full cgroup name for system containers | v0.9            | No                  |
| labels         | Comma-separated list of user-provided labels. Format is 'key:value'           | v0.9            | Yes                 |
| hostname       | Hostname where the container ran                                              | v0.9            | No                  |
| namespace_id   | UID of the namespace of a Pod                                                 | v0.14.1         | Yes                 |
| host_id        | Cloud-provider specified or user specified Identifier of a node               | v0.14.1         | Yes                 |
| resource_id    | An unique identifier used to differentiate multiple metrics of the same type. e.x. Fs partitions under filesystem/usage | v0.11.0 | No |


## Storage Schema

### InfluxDB

Each metric translates to a separate 'series' in InfluxDB. Labels are stored as additional columns.
The series name is constructed by combining the metric name with its type and units: "metric Name"_"units"_"type".

###### Query
`list series`

###### Output

```
cpu/usage_ns_cumulative
filesystem/usage
memory/page_faults_gauge
memory/usage_bytes_gauge
memory/working_set_bytes_gauge
network/rx_bytes_cumulative
network/rx_errors_cumulative
network/tx_bytes_cumulative
network/tx_errors_cumulative
uptime_ms_cumulative
```

*Note: Unit 'Count' is ignored*

Heapster adds timestamp and sequence number to every metric.

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
