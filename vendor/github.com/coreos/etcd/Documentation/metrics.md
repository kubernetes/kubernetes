# Metrics

**NOTE: The metrics feature is considered experimental. We may add/change/remove metrics without warning in future releases.**

etcd uses [Prometheus][prometheus] for metrics reporting in the server. The metrics can be used for real-time monitoring and debugging.
etcd only stores these data in memory. If a member restarts, metrics will reset.

The simplest way to see the available metrics is to cURL the metrics endpoint `/metrics` of etcd. The format is described [here](http://prometheus.io/docs/instrumenting/exposition_formats/).

Follow the [Prometheus getting started doc][prometheus-getting-started] to spin up a Prometheus server to collect etcd metrics.

The naming of metrics follows the suggested [best practice of Prometheus][prometheus-naming]. A metric name has an `etcd` prefix as its namespace and a subsystem prefix (for example `wal` and `etcdserver`).

etcd now exposes the following metrics:

## etcdserver

| Name                                    | Description                                      | Type      |
|-----------------------------------------|--------------------------------------------------|-----------|
| file_descriptors_used_total             | The total number of file descriptors used        | Gauge     |
| proposal_durations_seconds              | The latency distributions of committing proposal | Histogram |
| pending_proposal_total                  | The total number of pending proposals            | Gauge     |
| proposal_failed_total                   | The total number of failed proposals             | Counter   |

High file descriptors (`file_descriptors_used_total`) usage (near the file descriptors limitation of the process) indicates a potential out of file descriptors issue. That might cause etcd fails to create new WAL files and panics.

[Proposal][glossary-proposal] durations (`proposal_durations_seconds`) provides a histogram about the proposal commit latency. Latency can be introduced into this process by network and disk IO.

Pending proposal (`pending_proposal_total`) gives you an idea about how many proposal are in the queue and waiting for commit. An increasing pending number indicates a high client load or an unstable cluster.

Failed proposals (`proposal_failed_total`) are normally related to two issues: temporary failures related to a leader election or longer duration downtime caused by a loss of quorum in the cluster.

## wal

| Name                               | Description                                      | Type      |
|------------------------------------|--------------------------------------------------|-----------|
| fsync_durations_seconds            | The latency distributions of fsync called by wal | Histogram |
| last_index_saved                   | The index of the last entry saved by wal         | Gauge     |

Abnormally high fsync duration (`fsync_durations_seconds`) indicates disk issues and might cause the cluster to be unstable.


## http requests

These metrics describe the serving of requests (non-watch events) served by etcd members in non-proxy mode: total 
incoming requests, request failures and processing latency (inc. raft rounds for storage). They are useful for tracking
 user-generated traffic hitting the etcd cluster . 

All these metrics are prefixed with `etcd_http_`

| Name                           | Description                                                                         | Type                   |
|--------------------------------|-----------------------------------------------------------------------------------------|--------------------|
| received_total                 | Total number of events after parsing and auth.                                      | Counter(method)        |
| failed_total                   | Total number of failed events.                                                      | Counter(method,error)  |
| successful_duration_second     |  Bucketed handling times of the requests, including raft rounds for writes.          | Histogram(method)      |


Example Prometheus queries that may be useful from these metrics (across all etcd members):
 
 * `sum(rate(etcd_http_failed_total{job="etcd"}[1m]) by (method) / sum(rate(etcd_http_events_received_total{job="etcd"})[1m]) by (method)` 
    
    Shows the fraction of events that failed by HTTP method across all members, across a time window of `1m`.
 
 * `sum(rate(etcd_http_received_total{job="etcd",method="GET})[1m]) by (method)`
   `sum(rate(etcd_http_received_total{job="etcd",method~="GET})[1m]) by (method)`
    
    Shows the rate of successful readonly/write queries across all servers, across a time window of `1m`.
    
 * `histogram_quantile(0.9, sum(increase(etcd_http_successful_processing_seconds{job="etcd",method="GET"}[5m]) ) by (le))`
   `histogram_quantile(0.9, sum(increase(etcd_http_successful_processing_seconds{job="etcd",method!="GET"}[5m]) ) by (le))`
    
    Show the 0.90-tile latency (in seconds) of read/write (respectively) event handling across all members, with a window of `5m`.      

## snapshot

| Name                                       | Description                                                | Type      |
|--------------------------------------------|------------------------------------------------------------|-----------|
| snapshot_save_total_durations_seconds      | The total latency distributions of save called by snapshot | Histogram |

Abnormally high snapshot duration (`snapshot_save_total_durations_seconds`) indicates disk issues and might cause the cluster to be unstable.


## rafthttp

| Name                              | Description                                | Type         | Labels                         |
|-----------------------------------|--------------------------------------------|--------------|--------------------------------|
| message_sent_latency_seconds      | The latency distributions of messages sent | HistogramVec | sendingType, msgType, remoteID |
| message_sent_failed_total         | The total number of failed messages sent   | Summary      | sendingType, msgType, remoteID |


Abnormally high message duration (`message_sent_latency_seconds`) indicates network issues and might cause the cluster to be unstable.

An increase in message failures (`message_sent_failed_total`) indicates more severe network issues and might cause the cluster to be unstable.

Label `sendingType` is the connection type to send messages. `message`, `msgapp` and `msgappv2` use HTTP streaming, while `pipeline` does HTTP request for each message.

Label `msgType` is the type of raft message. `MsgApp` is log replication message; `MsgSnap` is snapshot install message; `MsgProp` is proposal forward message; the others are used to maintain raft internal status. If you have a large snapshot, you would expect a long msgSnap sending latency. For other types of messages, you would expect low latency, which is comparable to your ping latency if you have enough network bandwidth.

Label `remoteID` is the member ID of the message destination.


## proxy

etcd members operating in proxy mode do not do store operations. They forward all requests
 to cluster instances.

Tracking the rate of requests coming from a proxy allows one to pin down which machine is performing most reads/writes.

All these metrics are prefixed with `etcd_proxy_`

| Name                      | Description                                                                         | Type                   |
|---------------------------|-----------------------------------------------------------------------------------------|--------------------|
| requests_total            | Total number of requests by this proxy instance.    .                               | Counter(method)        |
| handled_total             | Total number of fully handled requests, with responses from etcd members.           | Counter(method)        |
| dropped_total             | Total number of dropped requests due to forwarding errors to etcd members.          | Counter(method,error)  |
| handling_duration_seconds | Bucketed handling times by HTTP method, including round trip to member instances.   | Histogram(method)      |  

Example Prometheus queries that may be useful from these metrics (across all etcd servers):

 *  `sum(rate(etcd_proxy_handled_total{job="etcd"}[1m])) by (method)`
    
    Rate of requests (by HTTP method) handled by all proxies, across a window of `1m`. 
 * `histogram_quantile(0.9, sum(increase(etcd_proxy_events_handling_time_seconds_bucket{job="etcd",method="GET"}[5m])) by (le))`
   `histogram_quantile(0.9, sum(increase(etcd_proxy_events_handling_time_seconds_bucket{job="etcd",method!="GET"}[5m])) by (le))`
    
    Show the 0.90-tile latency (in seconds) of handling of user requests across all proxy machines, with a window of `5m`.  
 * `sum(rate(etcd_proxy_dropped_total{job="etcd"}[1m])) by (proxying_error)`
    
    Number of failed request on the proxy. This should be 0, spikes here indicate connectivity issues to etcd cluster.

[glossary-proposal]: glossary.md#proposal
[prometheus]: http://prometheus.io/
[prometheus-getting-started](http://prometheus.io/docs/introduction/getting_started/)
[prometheus-naming]: http://prometheus.io/docs/practices/naming/
