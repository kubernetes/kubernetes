# Release Notes

## 0.19.1 (15-12-2015)
- Better connection handling for InfluxDB.
- Refresh node and pod lists every 1 hour.

## 0.19.0 (9-12-2015)
- Switched to InfluxDB 0.9. The data layout was changed.
- Added Kafka and Riemann sinks.
- Removed poll_duration flag.
- Authentication and security improvements.
- Fixed issue with unavailable sink.

## 0.18.1 (9-18-2015)
- Avoid using UIDs for Kubernetes events.
- Export container labels.
- Hawkular sink upgrades

## 0.18.0
- Cluster Model APIs enabled by default
- Garbage collection in the cluster model
- New rolling window based bucketing storage engine for model that significantly reduces memory footprint
- Many fixes around race conditions with the cache
- GCP authorization policy updated to reduce flakiness

## 0.17.0
- Fixes service account handling.
- New APIs that provide insight into a kubernetes cluster - usage distribution across nodes, namespaces, pods, etc.

## 0.16.0 (7-7-2015)
- Add new sink for GCM autoscaling.
- Force heapster to use kubernetes v1.0 API.
- Bug fixes.

## 0.15.0 (6-26-2015)
- Use service accounts while running in kubernetes clusters
- Expose kubernetes system containers with standard names
- Minor bug fixes.

## 0.14.3 (6-19-2015)
- Expose HostID for nodes. 
- Added an API '/metric-export-schema' that exposes the schema for '/metric-export' API.

## 0.14.2 (6-16-2015)
- Expose HostID for all containers.

## 0.14.1 (6-15-2015)
- Expose External ID of nodes as a label and namespace UID as a label.
- Fix bug in handling metric specific labels.

## 0.14.0 (6-9-2015)
- Heapster exposes a REST endpoint that serves the metrics being pushed to sinks.
- Support for service accounts.

## 0.13.0 (5-29-2015)
- Switch use of Kubernetes API to v1beta3.
- Add option to connect to Kubelets through HTTPS.

## 0.12.1 (5-13-2015)
- Fixes kube master url handling via --source flag.

## 0.12.0 (5-12-2015)
- Fixes issues related to supporting multiple sinks.
- Resource usage of kubernetes system daemons are exported.
- Support for kubernetes specific auth (secrets).
- Scalability improvements.

## 0.11.0 (4-28-2015)
- Export filesystem usage metrics for the node and containers.
- Support for Kubernetes events
- New Sink - Google Cloud Logging. Supports events only.
- New Sink - Google Cloud Monitoring. Supports metrics only.
- New metric labels - 'pod_name' and 'resource_id'.
- Extensible source and sinks configuration. It is now possible to export metrics and events to multiple sinks simultaneously.

## 0.10.0 (3-30-2015)
- Downsampling - Resolution of metrics is set to 5s by default.
- Support for using Kube client auth.
- Improved Influxdb sink - sequence numbers are generated for every metric.
- Reliability improvements
- Bug fixes.

## 0.9 (3-13-2015)
- [Standardized metrics](sinks/api/supported_metrics.go)
- New [common API](sinks/api/types.go) in place for all external storage drivers.
- Simplified heapster deployment scripts.
- Bug fixes and misc enhancements.

## 0.8 (2-22-2015)
- Avoid expecting HostIP of Pod to match node's HostIP. 

## 0.7 (2-18-2015)
- Support for Google Cloud Monitoring Backend
- Watch kubernetes api-server instead of polling for pods and nodes info.
- Fetch stats in parallel.
- Refactor code and improve testing.
- Miscellaneous bug fixes.
- Native support for CoreOS.

## 0.6 (1-21-2015)
- New /validate REST endpoint to probe heapster.
- Heapster supports kube namespaces.
- Heapster uses InfluxDB service DNS name while running in kube mode.

## 0.5 (12-11-2014)
- Compatiblity with updated InfluxDB service names.

## 0.4 (12-02-2014)
- Compatibility with cAdvisor v0.6.2

## 0.3 (11-26-2014)
- Handle updated Pod API in Kubernetes.

## 0.2 (10-06-2014)
- Use kubernetes master readonly service which does not require auth

## 0.1 (10-05-2014)
- First version of heapster.
- Native support for kubernetes and CoreOS.
- For Kubernetes gets pods and rootcgroup information.
- For CoreOS gets containers and rootcgroup information.
- Supports InfluxDB and bigquery.
- Exports pods and container stats in table 'stats' in InfluxDB
- rootCgroup is exported in table 'machine' in InfluxDB
- Special dashboard for kubernetes.
