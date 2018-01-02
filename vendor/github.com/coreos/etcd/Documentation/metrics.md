# Metrics

etcd uses [Prometheus][prometheus] for metrics reporting. The metrics can be used for real-time monitoring and debugging. etcd does not persist its metrics; if a member restarts, the metrics will be reset.

The simplest way to see the available metrics is to cURL the metrics endpoint `/metrics`. The format is described [here](http://prometheus.io/docs/instrumenting/exposition_formats/).

Follow the [Prometheus getting started doc][prometheus-getting-started] to spin up a Prometheus server to collect etcd metrics.

The naming of metrics follows the suggested [Prometheus best practices][prometheus-naming]. A metric name has an `etcd` or `etcd_debugging` prefix as its namespace and a subsystem prefix (for example `wal` and `etcdserver`).

## etcd namespace metrics

The metrics under the `etcd` prefix are for monitoring and alerting. They are stable high level metrics. If there is any change of these metrics, it will be included in release notes.

Metrics that are etcd2 related are documented [v2 metrics guide][v2-http-metrics].

### Server

These metrics describe the status of the etcd server. In order to detect outages or problems for troubleshooting, the server metrics of every production etcd cluster should be closely monitored.

All these metrics are prefixed with `etcd_server_`

| Name                      | Description                                              | Type    |
|---------------------------|----------------------------------------------------------|---------|
| has_leader                | Whether or not a leader exists. 1 is existence, 0 is not.| Gauge   |
| leader_changes_seen_total | The number of leader changes seen.                       | Counter |
| proposals_committed_total | The total number of consensus proposals committed.       | Gauge   |
| proposals_applied_total   | The total number of consensus proposals applied.         | Gauge   |
| proposals_pending         | The current number of pending proposals.                 | Gauge   |
| proposals_failed_total    | The total number of failed proposals seen.               | Counter |

`has_leader` indicates whether the member has a leader. If a member does not have a leader, it is
totally unavailable. If all the members in the cluster do not have any leader, the entire cluster
is totally unavailable.

`leader_changes_seen_total` counts the number of leader changes the member has seen since its start. Rapid leadership changes impact the performance of etcd significantly. It also signals that the leader is unstable, perhaps due to network connectivity issues or excessive load hitting the etcd cluster.

`proposals_committed_total` records the total number of consensus proposals committed. This gauge should increase over time if the cluster is healthy. Several healthy members of an etcd cluster may have different total committed proposals at once. This discrepancy may be due to recovering from peers after starting, lagging behind the leader, or being the leader and therefore having the most commits. It is important to monitor this metric across all the members in the cluster; a consistently large lag between a single member and its leader indicates that member is slow or unhealthy.

`proposals_applied_total` records the total number of consensus proposals applied. The etcd server applies every committed proposal asynchronously. The difference between `proposals_committed_total` and `proposals_applied_total` should usually be small (within a few thousands even under high load). If the difference between them continues to rise, it indicates that the etcd server is overloaded. This might happen when applying expensive queries like heavy range queries or large txn operations.

`proposals_pending` indicates how many proposals are queued to commit. Rising pending proposals suggests there is a high client load or the member cannot commit proposals.

`proposals_failed_total` are normally related to two issues: temporary failures related to a leader election or longer downtime caused by a loss of quorum in the cluster.

### Disk

These metrics describe the status of the disk operations.

All these metrics are prefixed with `etcd_disk_`.

| Name                               | Description                                           | Type      |
|------------------------------------|-------------------------------------------------------|-----------|
| wal_fsync_duration_seconds         | The latency distributions of fsync called by wal      | Histogram |
| backend_commit_duration_seconds    | The latency distributions of commit called by backend.| Histogram |

A `wal_fsync` is called when etcd persists its log entries to disk before applying them.

A `backend_commit` is called when etcd commits an incremental snapshot of its most recent changes to disk.

High disk operation latencies (`wal_fsync_duration_seconds` or `backend_commit_duration_seconds`) often indicate disk issues. It may cause high request latency or make the cluster unstable.

### Network

These metrics describe the status of the network.

All these metrics are prefixed with `etcd_network_`

| Name                      | Description                                                        | Type          |
|---------------------------|--------------------------------------------------------------------|---------------|
| peer_sent_bytes_total           | The total number of bytes sent to the peer with ID `To`.         | Counter(To)   |
| peer_received_bytes_total       | The total number of bytes received from the peer with ID `From`. | Counter(From) |
| peer_sent_failures_total        | The total number of send failures from the peer with ID `To`.         | Counter(To)   |
| peer_received_failures_total    | The total number of receive failures from the peer with ID `From`. | Counter(From) |
| peer_round_trip_time_seconds    | Round-Trip-Time histogram between peers.                         | Histogram(To) |
| client_grpc_sent_bytes_total    | The total number of bytes sent to grpc clients.                  | Counter   |
| client_grpc_received_bytes_total| The total number of bytes received to grpc clients.              | Counter   |

`peer_sent_bytes_total` counts the total number of bytes sent to a specific peer. Usually the leader member sends more data than other members since it is responsible for transmitting replicated data.

`peer_received_bytes_total` counts the total number of bytes received from a specific peer. Usually follower members receive data only from the leader member.

### gRPC requests

These metrics are exposed via [go-grpc-prometheus][go-grpc-prometheus].

## etcd_debugging namespace metrics

The metrics under the `etcd_debugging` prefix are for debugging. They are very implementation dependent and volatile. They might be changed or removed without any warning in new etcd releases. Some of the metrics might be moved to the `etcd` prefix when they become more stable.


### Snapshot

| Name                                       | Description                                                | Type      |
|--------------------------------------------|------------------------------------------------------------|-----------|
| snapshot_save_total_duration_seconds      | The total latency distributions of save called by snapshot | Histogram |

Abnormally high snapshot duration (`snapshot_save_total_duration_seconds`) indicates disk issues and might cause the cluster to be unstable.

## Prometheus supplied metrics

The Prometheus client library provides a number of metrics under the `go` and `process` namespaces. There are a few that are particlarly interesting.

| Name                              | Description                                | Type         |
|-----------------------------------|--------------------------------------------|--------------|
| process_open_fds                  | Number of open file descriptors.           | Gauge        |
| process_max_fds                   | Maximum number of open file descriptors.   | Gauge        |

Heavy file descriptor (`process_open_fds`) usage (i.e., near the process's file descriptor limit, `process_max_fds`) indicates a potential file descriptor exhaustion issue. If the file descriptors are exhausted, etcd may panic because it cannot create new WAL files.

[glossary-proposal]: learning/glossary.md#proposal
[prometheus]: http://prometheus.io/
[prometheus-getting-started]: http://prometheus.io/docs/introduction/getting_started/
[prometheus-naming]: http://prometheus.io/docs/practices/naming/
[v2-http-metrics]: v2/metrics.md#http-requests
[go-grpc-prometheus]: https://github.com/grpc-ecosystem/go-grpc-prometheus