Heapster Overview
===================

## Sources
Heapster can pull data from multiple sources. See [source-configuration.md](source-configuration.md) for information on how to configure heapster sources. Heapster currently supports the following sources:
* Kubernetes
  * Pod Metrics
    * Metrics for each Kubernetes pod.
  * Node Metrics
    * Metrics for each Kubernetes node/minion machine.
  * Events
    * Kubernetes does not hold on to events indefinitely. Heapster can be used to fetch events from Kubernetes and store them to a more persistent store.
    * Heapster issues a list/watch against the specified Kubernetes API server to fetch all existing and future events from that server.
    * Error conditions can cause heapster to re-list and watch events, which could result in duplicate events being pulled from Kubernetes. Sinks are responsible for de-duplication.
* Cadvisor
  * Standalone/External
    * External cadvisor source "discovers" hosts from the specified file.
  * CoreOS
    * CoreOS cadvisor source discovers nodes from the specified fleet endpoints.

## Sinks
Heapster can push data to multiple sinks. Heapster currently supports the following sinks:
* InfluxDB
  * Sinks metrics and events.
  * See [influxdb.md](influxdb.md) for information on accessing InfluxDB UI.
* Google Cloud Monitoring (GCM)
  * Only sinks metrics.
  * Only supported when heapster is running on Google Compute Engine (GCE).
  * See [google.md](google.md) for information on accessing data collected by GCM.
* Google Cloud Logging (GCL)
  * Only sinks events.
  * Only supported when heapster is running on Google Compute Engine (GCE).
  * GCE instance must have “https://www.googleapis.com/auth/logging.write” auth scope
  * GCE instance must have Logging API enabled for the project in Google Developer Console
  * GCL Logs are accessible via:
    * `https://console.developers.google.com/project/<project_ID>/logs?service=custom.googleapis.com`
      * Where `project_ID` is the project ID of the Google Cloud Platform project ID.
    * Select `kubernetes.io/events` from the `All logs` drop down menu.
* Hawkular
  * This sink supports monitoring metrics only.Hawkular is a an opensource monitoring solution, 
    it provides a set of REST services that makes it easy to store metrics, display metrics.
  * See [sink-configuration.md](https://github.com/kubernetes/heapster/blob/master/docs/sink-configuration.md#hawkular-metrics) for information on how to config the Hawkular sink.
* Kafka
  * This sink supports monitoring metrics and events.Kafka is a high-throughput distributed messaging system.
  * See [sink-configuration.md](https://github.com/kubernetes/heapster/blob/master/docs/sink-configuration.md#kafka) for information on how to config the Kafka sink.
* Riemann
  * This sink supports metrics and events.Riemann is a network monitoring system, 
    it can aggregate events from your servers and applications with a powerful stream processing language.
  * See [sink-configuration.md](https://github.com/kubernetes/heapster/blob/master/docs/sink-configuration.md#riemann) for information on how to config the Riemann sink.
