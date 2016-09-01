<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Performance Monitoring

## Reason for this document

This document serves as a place to gather information about past performance regressions, their reason and impact and discuss ideas to avoid similar regressions in the future.
Main reason behind doing this is to understand what kind of monitoring needs to be in place to keep Kubernetes fast.

## Known past and present performance issues

### Higher logging level causing scheduler stair stepping

Issue https://github.com/kubernetes/kubernetes/issues/14216 was opened because @spiffxp observed a regression in scheduler performance in 1.1 branch in comparison to `old` 1.0
cut. In the end it turned out the be caused by `--v=4` (instead of default `--v=2`) flag in the scheduler together with the flag `--logtostderr` which disables batching of
log lines and a number of logging without explicit V level. This caused weird behavior of the whole component.

Because we now know that logging may have big performance impact we should consider instrumenting logging mechanism and compute statistics such as number of logged messages,
total and average size of them. Each binary should be responsible for exposing its metrics. An unaccounted but way too big number of days, if not weeks, of engineering time was
lost because of this issue.

### Adding per-pod probe-time, which increased the number of PodStatus updates, causing major slowdown

In September 2015 we tried to add per-pod probe times to the PodStatus. It caused (https://github.com/kubernetes/kubernetes/issues/14273) a massive increase in both number and
total volume of object (PodStatus) changes. It drastically increased the load on API server which wasn’t able to handle new number of requests quickly enough, violating our
response time SLO. We had to revert this change.

### Late Ready->Running PodPhase transition caused test failures as it seemed like slowdown

In late September we encountered a strange problem (https://github.com/kubernetes/kubernetes/issues/14554): we observed an increased observed latencies in small clusters (few
Nodes). It turned out that it’s caused by an added latency between PodRunning and PodReady phases. This was not a real regression, but our tests thought it were, which shows
how careful we need to be.

### Huge number of handshakes slows down API server

It was a long standing issue for performance and is/was an important bottleneck for scalability (https://github.com/kubernetes/kubernetes/issues/13671). The bug directly
causing this problem was incorrect (from the golangs standpoint) handling of TCP connections. Secondary issue was that elliptic curve encryption (only one available in go 1.4)
is unbelievably slow.

## Proposed metrics/statistics to gather/compute to avoid problems

### Cluster-level metrics

Basic ideas:
- number of Pods/ReplicationControllers/Services in the cluster
- number of running replicas of master components (if they are replicated)
- current elected master of ectd cluster (if running distributed version)
- nuber of master component restarts
- number of lost Nodes

### Logging monitoring

Log spam is a serious problem and we need to keep it under control. Simplest way to check for regressions, suggested by @bredanburns, is to compute the rate in which log files
grow in e2e tests.

Basic ideas:
- log generation rate (B/s)

### REST call monitoring

We do measure REST call duration in the Density test, but we need an API server monitoring as well, to avoid false failures caused e.g. by the network traffic. We already have
some metrics in place (https://github.com/kubernetes/kubernetes/blob/master/pkg/apiserver/metrics/metrics.go), but we need to revisit the list and add some more.

Basic ideas:
- number of calls per verb, client, resource type
- latency distribution per verb, client, resource type
- number of calls that was rejected per client, resource type and reason (invalid version number, already at maximum number of requests in flight)
- number of relists in various watchers

### Rate limit monitoring

Reverse of REST call monitoring done in the API server. We need to know when a given component increases a pressure it puts on the API server. As a proxy for number of
requests sent we can track how saturated are rate limiters. This has additional advantage of giving us data needed to fine-tune rate limiter constants.

Because we have rate limitting on both ends (client and API server) we should monitor number of inflight requests in API server and how it relates to `max-requests-inflight`.

Basic ideas:
- percentage of used non-burst limit,
- amount of time in last hour with depleted burst tokens,
- number of inflight requests in API server.

### Network connection monitoring

During development we observed incorrect use/reuse of HTTP connections multiple times already. We should at least monitor number of created connections.

### ETCD monitoring

@xiang-90 and @hongchaodeng - you probably have way more experience on what'd be good to look at from the ETCD perspective.

Basic ideas:
- ETCD memory footprint
- number of objects per kind
- read/write latencies per kind
- number of requests from the API server
- read/write counts per key (it may be too heavy though)

### Resource consumption

On top of all things mentioned above we need to monitor changes in resource usage in both: cluster components (API server, Kubelet, Scheduler, etc.) and system add-ons
(Heapster, L7 load balancer, etc.). Monitoring memory usage is tricky, because if no limits are set, system won't apply memory pressure to processes, which makes their memory
footprint constantly grow. We argue that monitoring usage in tests still makes sense, as tests should be repeatable, and if memory usage will grow drastically between two runs
it most likely can be attributed to some kind of regression (assuming that nothing else has changed in the environment).

Basic ideas:
- CPU usage
- memory usage

### Other saturation metrics

We should monitor other aspects of the system, which may indicate saturation of some component.

Basic ideas:
- queue length for queues in the system,
- wait time for WaitGroups.



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/performance-related-monitoring.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
