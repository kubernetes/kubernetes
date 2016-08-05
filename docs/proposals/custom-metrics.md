<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/proposals/custom-metrics.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Custom Metrics in Kubernetes


## Preface

Our aim is to create a mechanism in Kubernetes that will allow pods to expose custom system metrics, collect them, and make them accessible.
Custom metrics are needed by:
* horizontal pod autoscaler, for autoscaling the number of pods based on them;
* scheduler, for using them in a sophisticated scheduling algorithm.

High level goals for our solution for version 1.2 are:
* easy to use (it should be easy to export custom metric from user's application),
* works for most application (it should be easy to configure monitoring for third-party applications),
* performance & scalability (the largest supported cluster should be able to handle ~5 custom metrics per pod with reporting latency ~30 seconds).

For version 1.2, we are not going to address the following issues (non-goals):
* security of access to custom metrics,
* general monitoring of application health by a user
(out of the heapster scope, see [#665](https://github.com/kubernetes/heapster/issues/665)).

## Design

For the Kubernetes version 1.2, we plan to implement aggregation of pod custom metrics in Prometheus format by pull.

Each pod, to expose custom metrics, will expose a set of Prometheus endpoints.
(For version 1.2, we assume that custom metrics are not private information and they are accessible by everyone.
In future, we may restrict it by making the endpoints accessible only by kubelet/cAdvisor).
CAdvisor will collect metrics from such endpoints of all pods on each node by pulling, and expose them to Heapster.
Heapster will:
* collect custom metrics from all CAdvisors in the cluster, together with pulling system metrics
(for version 1.2: we assume pooling period of ~30 seconds),
* store them in a metrics backend (influxDB, Prometheus, Hawkular, GCM, …),
* expose the latest snapshot of custom metrics for queries (by HPA/scheduler/…) using [model API](https://github.com/kubernetes/heapster/blob/master/docs/model.md).

User can easily expose Prometheus metrics for her own application by using Prometheus [client](http://prometheus.io/docs/instrumenting/clientlibs/) library.
To monitor third-party applications, Prometheus [exporters](http://prometheus.io/docs/instrumenting/exporters/) run as side-cars containers may be used.

For version 1.2, to prevent a huge number of metrics negatively affect the system performance,
the number of metrics that can be exposed by each pod will be limited to the configurable value (default: 5).
In future, we will need a way to cap the number of exposed metrics per pod,
one of possible solutions is using LimtRanger admission control plugin.

In future versions (later than 1.2), we want to extend our solution by:
* accepting pod metrics exposed in different formats than Prometheus
(collecting of the different formats will need to be supported by cAdvisor),
* support push metrics by exposing push API on heapster (e.g. in StatsD format) or on a local node collector
(if heapster performance is insufficient),
* support metrics not associated with an individual pod.


## API

For Kubernetes 1.2, defining pod Prometheus endpoints will be done using annotations.
Later, when we are sure that our API is correct and stable, we will make it a part of `PodSpec`.

We will add a new optional pod annotation with the following key: `metrics.alpha.kubernetes.io/custom-endpoints`.
It will contain a string-value in JSON format.
The value will be a list of tuples defining ports, paths and API
(currently, we will support only Prometheus API, this will be also the default value if format is empty)
of metrics endpoints exposed by the pod, and names of metrics which should be taken from the endpoint (obligatory, no more than the configurable limit).

The annotation will be interpreted by kubelet during pod creation.
It will not be possible to add/delete/edit it during the life time of a pod: such operations will be rejected.

For example, the following configuration:

```
"metrics.alpha.kubernetes.io/custom-endpoints" = [
	{
		"api": "prometheus",
		"path": "/status",
		"port": "8080",
		"names": ["qps", "activeConnections"]
	},
	{
		"path": "/metrics",
		"port": "9090"
		"names": ["myMetric"]
	}
]
```

will expose metrics with names `qps` and `activeConnections` from `localhost:8080/status` and metric `myMetric` from `localhost:9090/metrics`.
Please note that both endpoints are in Prometheus format.


## Implementation notes

1. Kubelet will parse value of `metrics.alpha.kubernetes.io/custom-endpoints` annotation for pods.
In case of error, pod will not be started (will be marked as failed) and kubelet will generate `FailedToCreateContainer` event with appropriate message
(we will not introduce any new event type, as types of events are considered a part of kubelet API and we do not want to change it).

1. Kubelet will use application metrics in CAdvisor for implementation:
	* It will create a configuration file for CAdvisor based on the annotation,
	* It will mount this file as a part of a docker image to run,
	* It will set a docker label for the image to point CAdvisor to this file.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/custom-metrics.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
