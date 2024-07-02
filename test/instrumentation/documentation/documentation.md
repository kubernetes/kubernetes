---
title: Kubernetes Metrics Reference
content_type: reference
auto_generated: true
description: >-
  Details of the metric data that Kubernetes components export.
---

## Metrics (v1.30)

<!-- (auto-generated 2024 Jul 02) -->
<!-- (auto-generated v1.30) -->
This page details the metrics that different Kubernetes components export. You can query the metrics endpoint for these 
components using an HTTP scrape, and fetch the current metrics data in Prometheus format.

### List of Stable Kubernetes Metrics

Stable metrics observe strict API contracts and no labels can be added or removed from stable metrics during their lifetime.

<div class="metrics"><div class="metric" data-stability="stable">
	<div class="metric_name">apiserver_admission_controller_admission_duration_seconds</div>
	<div class="metric_help">Admission controller latency histogram in seconds, identified by name and broken out for each operation and API resource and type (validate or admit).</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span><span class="metric_label">operation</span><span class="metric_label">rejected</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">apiserver_admission_step_admission_duration_seconds</div>
	<div class="metric_help">Admission sub-step latency histogram in seconds, broken out for each operation and API resource and step type (validate or admit).</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation</span><span class="metric_label">rejected</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">apiserver_admission_webhook_admission_duration_seconds</div>
	<div class="metric_help">Admission webhook latency histogram in seconds, identified by name and broken out for each operation and API resource and type (validate or admit).</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span><span class="metric_label">operation</span><span class="metric_label">rejected</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">apiserver_current_inflight_requests</div>
	<div class="metric_help">Maximal number of currently used inflight request limit of this apiserver per request kind in last second.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">request_kind</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">apiserver_longrunning_requests</div>
	<div class="metric_help">Gauge of all active long-running apiserver requests broken out by verb, group, version, resource, scope and component. Not all requests are tracked this way.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">component</span><span class="metric_label">group</span><span class="metric_label">resource</span><span class="metric_label">scope</span><span class="metric_label">subresource</span><span class="metric_label">verb</span><span class="metric_label">version</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">apiserver_request_duration_seconds</div>
	<div class="metric_help">Response latency distribution in seconds for each verb, dry run value, group, version, resource, subresource, scope and component.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">component</span><span class="metric_label">dry_run</span><span class="metric_label">group</span><span class="metric_label">resource</span><span class="metric_label">scope</span><span class="metric_label">subresource</span><span class="metric_label">verb</span><span class="metric_label">version</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">apiserver_request_total</div>
	<div class="metric_help">Counter of apiserver requests broken out for each verb, dry run value, group, version, resource, scope, component, and HTTP response code.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span><span class="metric_label">component</span><span class="metric_label">dry_run</span><span class="metric_label">group</span><span class="metric_label">resource</span><span class="metric_label">scope</span><span class="metric_label">subresource</span><span class="metric_label">verb</span><span class="metric_label">version</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">apiserver_requested_deprecated_apis</div>
	<div class="metric_help">Gauge of deprecated APIs that have been requested, broken out by API group, version, resource, subresource, and removed_release.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">group</span><span class="metric_label">removed_release</span><span class="metric_label">resource</span><span class="metric_label">subresource</span><span class="metric_label">version</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">apiserver_response_sizes</div>
	<div class="metric_help">Response size distribution in bytes for each group, version, verb, resource, subresource, scope and component.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">component</span><span class="metric_label">group</span><span class="metric_label">resource</span><span class="metric_label">scope</span><span class="metric_label">subresource</span><span class="metric_label">verb</span><span class="metric_label">version</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">apiserver_storage_objects</div>
	<div class="metric_help">Number of stored objects at the time of last check split by kind. In case of a fetching error, the value will be -1.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">apiserver_storage_size_bytes</div>
	<div class="metric_help">Size of the storage database file physically allocated in bytes.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">storage_cluster_id</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">container_cpu_usage_seconds_total</div>
	<div class="metric_help">Cumulative cpu time consumed by the container in core-seconds</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">container</span><span class="metric_label">pod</span><span class="metric_label">namespace</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">container_memory_working_set_bytes</div>
	<div class="metric_help">Current working set of the container in bytes</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">container</span><span class="metric_label">pod</span><span class="metric_label">namespace</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">container_start_time_seconds</div>
	<div class="metric_help">Start time of the container since unix epoch in seconds</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">container</span><span class="metric_label">pod</span><span class="metric_label">namespace</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">cronjob_controller_job_creation_skew_duration_seconds</div>
	<div class="metric_help">Time between when a cronjob is scheduled to be run, and when the corresponding job is created</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">job_controller_job_pods_finished_total</div>
	<div class="metric_help">The number of finished Pods that are fully tracked</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">completion_mode</span><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">job_controller_job_sync_duration_seconds</div>
	<div class="metric_help">The time it took to sync a job</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">action</span><span class="metric_label">completion_mode</span><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">job_controller_job_syncs_total</div>
	<div class="metric_help">The number of job syncs</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">action</span><span class="metric_label">completion_mode</span><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">job_controller_jobs_finished_total</div>
	<div class="metric_help">The number of finished jobs</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">completion_mode</span><span class="metric_label">reason</span><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">kube_pod_resource_limit</div>
	<div class="metric_help">Resources limit for workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">namespace</span><span class="metric_label">pod</span><span class="metric_label">node</span><span class="metric_label">scheduler</span><span class="metric_label">priority</span><span class="metric_label">resource</span><span class="metric_label">unit</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">kube_pod_resource_request</div>
	<div class="metric_help">Resources requested by workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">namespace</span><span class="metric_label">pod</span><span class="metric_label">node</span><span class="metric_label">scheduler</span><span class="metric_label">priority</span><span class="metric_label">resource</span><span class="metric_label">unit</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">kubernetes_healthcheck</div>
	<div class="metric_help">This metric records the result of a single healthcheck.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">kubernetes_healthchecks_total</div>
	<div class="metric_help">This metric records the results of all healthcheck.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span><span class="metric_label">status</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">node_collector_evictions_total</div>
	<div class="metric_help">Number of Node evictions that happened since current instance of NodeController started.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">zone</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">node_cpu_usage_seconds_total</div>
	<div class="metric_help">Cumulative cpu time consumed by the node in core-seconds</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	</ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">node_memory_working_set_bytes</div>
	<div class="metric_help">Current working set of the node in bytes</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	</ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">pod_cpu_usage_seconds_total</div>
	<div class="metric_help">Cumulative cpu time consumed by the pod in core-seconds</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">pod</span><span class="metric_label">namespace</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">pod_memory_working_set_bytes</div>
	<div class="metric_help">Current working set of the pod in bytes</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">pod</span><span class="metric_label">namespace</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">resource_scrape_error</div>
	<div class="metric_help">1 if there was an error while getting container metrics, 0 otherwise</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	</ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">scheduler_framework_extension_point_duration_seconds</div>
	<div class="metric_help">Latency for running all plugins of a specific extension point.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">extension_point</span><span class="metric_label">profile</span><span class="metric_label">status</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">scheduler_pending_pods</div>
	<div class="metric_help">Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulablePods that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">queue</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">scheduler_pod_scheduling_attempts</div>
	<div class="metric_help">Number of attempts to successfully schedule a pod.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">scheduler_pod_scheduling_duration_seconds</div>
	<div class="metric_help">E2e latency for a pod being scheduled which may include multiple scheduling attempts.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">attempts</span></li><li class="metric_deprecated_version"><label class="metric_detail">Deprecated Versions:</label><span>1.29.0</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">scheduler_preemption_attempts_total</div>
	<div class="metric_help">Total preemption attempts in the cluster till now</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">scheduler_preemption_victims</div>
	<div class="metric_help">Number of selected preemption victims</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">scheduler_queue_incoming_pods_total</div>
	<div class="metric_help">Number of pods added to scheduling queues by event and queue type.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">event</span><span class="metric_label">queue</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">scheduler_schedule_attempts_total</div>
	<div class="metric_help">Number of attempts to schedule pods, by the result. 'unschedulable' means a pod could not be scheduled, while 'error' means an internal scheduler problem.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">profile</span><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="stable">
	<div class="metric_name">scheduler_scheduling_attempt_duration_seconds</div>
	<div class="metric_help">Scheduling attempt latency in seconds (scheduling algorithm + binding)</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">STABLE</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">profile</span><span class="metric_label">result</span></li></ul>
	</div>
</div>

### List of Beta Kubernetes Metrics

Beta metrics observe a looser API contract than its stable counterparts. No labels can be removed from beta metrics during their lifetime, however, labels can be added while the metric is in the beta stage. This offers the assurance that beta metrics will honor existing dashboards and alerts, while allowing for amendments in the future. 

<div class="metrics"><div class="metric" data-stability="beta">
	<div class="metric_name">apiserver_flowcontrol_current_executing_requests</div>
	<div class="metric_help">Number of requests in initial (for a WATCH) or any (for a non-WATCH) execution stage in the API Priority and Fairness subsystem</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">BETA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">flow_schema</span><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="beta">
	<div class="metric_name">apiserver_flowcontrol_current_executing_seats</div>
	<div class="metric_help">Concurrency (number of seats) occupied by the currently executing (initial stage for a WATCH, any stage otherwise) requests in the API Priority and Fairness subsystem</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">BETA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">flow_schema</span><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="beta">
	<div class="metric_name">apiserver_flowcontrol_current_inqueue_requests</div>
	<div class="metric_help">Number of requests currently pending in queues of the API Priority and Fairness subsystem</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">BETA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">flow_schema</span><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="beta">
	<div class="metric_name">apiserver_flowcontrol_dispatched_requests_total</div>
	<div class="metric_help">Number of requests executed by API Priority and Fairness subsystem</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">BETA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">flow_schema</span><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="beta">
	<div class="metric_name">apiserver_flowcontrol_nominal_limit_seats</div>
	<div class="metric_help">Nominal number of execution seats configured for each priority level</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">BETA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="beta">
	<div class="metric_name">apiserver_flowcontrol_rejected_requests_total</div>
	<div class="metric_help">Number of requests rejected by API Priority and Fairness subsystem</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">BETA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">flow_schema</span><span class="metric_label">priority_level</span><span class="metric_label">reason</span></li></ul>
	</div><div class="metric" data-stability="beta">
	<div class="metric_name">apiserver_flowcontrol_request_wait_duration_seconds</div>
	<div class="metric_help">Length of time a request spent waiting in its queue</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">BETA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">execute</span><span class="metric_label">flow_schema</span><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="beta">
	<div class="metric_name">disabled_metrics_total</div>
	<div class="metric_help">The count of disabled metrics.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">BETA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="beta">
	<div class="metric_name">hidden_metrics_total</div>
	<div class="metric_help">The count of hidden metrics.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">BETA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="beta">
	<div class="metric_name">kubernetes_feature_enabled</div>
	<div class="metric_help">This metric records the data about the stage and enablement of a k8s feature.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">BETA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span><span class="metric_label">stage</span></li></ul>
	</div><div class="metric" data-stability="beta">
	<div class="metric_name">registered_metrics_total</div>
	<div class="metric_help">The count of registered metrics broken by stability level and deprecation version.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">BETA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">deprecated_version</span><span class="metric_label">stability_level</span></li></ul>
	</div><div class="metric" data-stability="beta">
	<div class="metric_name">scheduler_pod_scheduling_sli_duration_seconds</div>
	<div class="metric_help">E2e latency for a pod being scheduled, from the time the pod enters the scheduling queue an d might involve multiple scheduling attempts.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">BETA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">attempts</span></li></ul>
	</div>
</div>

### List of Alpha Kubernetes Metrics

Alpha metrics do not have any API guarantees. These metrics must be used at your own risk, subsequent versions of Kubernetes may remove these metrics altogether, or mutate the API in such a way that breaks existing dashboards and alerts. 

<div class="metrics"><div class="metric" data-stability="alpha">
	<div class="metric_name">aggregator_discovery_aggregation_count_total</div>
	<div class="metric_help">Counter of number of times discovery was aggregated</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">aggregator_openapi_v2_regeneration_count</div>
	<div class="metric_help">Counter of OpenAPI v2 spec regeneration count broken down by causing APIService name and reason.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">apiservice</span><span class="metric_label">reason</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">aggregator_openapi_v2_regeneration_duration</div>
	<div class="metric_help">Gauge of OpenAPI v2 spec regeneration duration in seconds.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">reason</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">aggregator_unavailable_apiservice</div>
	<div class="metric_help">Gauge of APIServices which are marked as unavailable broken down by APIService name.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">aggregator_unavailable_apiservice_total</div>
	<div class="metric_help">Counter of APIServices which are marked as unavailable broken down by APIService name and reason.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span><span class="metric_label">reason</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiextensions_apiserver_validation_ratcheting_seconds</div>
	<div class="metric_help">Time for comparison of old to new for the purposes of CRDValidationRatcheting during an UPDATE in seconds.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiextensions_openapi_v2_regeneration_count</div>
	<div class="metric_help">Counter of OpenAPI v2 spec regeneration count broken down by causing CRD name and reason.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">crd</span><span class="metric_label">reason</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiextensions_openapi_v3_regeneration_count</div>
	<div class="metric_help">Counter of OpenAPI v3 spec regeneration count broken down by group, version, causing CRD and reason.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">crd</span><span class="metric_label">group</span><span class="metric_label">reason</span><span class="metric_label">version</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_admission_match_condition_evaluation_errors_total</div>
	<div class="metric_help">Admission match condition evaluation errors count, identified by name of resource containing the match condition and broken out for each kind containing matchConditions (webhook or policy), operation and admission type (validate or admit).</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">kind</span><span class="metric_label">name</span><span class="metric_label">operation</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_admission_match_condition_evaluation_seconds</div>
	<div class="metric_help">Admission match condition evaluation time in seconds, identified by name and broken out for each kind containing matchConditions (webhook or policy), operation and type (validate or admit).</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">kind</span><span class="metric_label">name</span><span class="metric_label">operation</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_admission_match_condition_exclusions_total</div>
	<div class="metric_help">Admission match condition evaluation exclusions count, identified by name of resource containing the match condition and broken out for each kind containing matchConditions (webhook or policy), operation and admission type (validate or admit).</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">kind</span><span class="metric_label">name</span><span class="metric_label">operation</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_admission_step_admission_duration_seconds_summary</div>
	<div class="metric_help">Admission sub-step latency summary in seconds, broken out for each operation and API resource and step type (validate or admit).</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="summary"><label class="metric_detail">Type:</label> <span class="metric_type">Summary</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation</span><span class="metric_label">rejected</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_admission_webhook_fail_open_count</div>
	<div class="metric_help">Admission webhook fail open count, identified by name and broken out for each admission type (validating or mutating).</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_admission_webhook_rejection_count</div>
	<div class="metric_help">Admission webhook rejection count, identified by name and broken out for each admission type (validating or admit) and operation. Additional labels specify an error type (calling_webhook_error or apiserver_internal_error if an error occurred; no_error otherwise) and optionally a non-zero rejection code if the webhook rejects the request with an HTTP status code (honored by the apiserver when the code is greater or equal to 400). Codes greater than 600 are truncated to 600, to keep the metrics cardinality bounded.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">error_type</span><span class="metric_label">name</span><span class="metric_label">operation</span><span class="metric_label">rejection_code</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_admission_webhook_request_total</div>
	<div class="metric_help">Admission webhook request total, identified by name and broken out for each admission type (validating or mutating) and operation. Additional labels specify whether the request was rejected or not and an HTTP status code. Codes greater than 600 are truncated to 600, to keep the metrics cardinality bounded.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span><span class="metric_label">name</span><span class="metric_label">operation</span><span class="metric_label">rejected</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_audit_error_total</div>
	<div class="metric_help">Counter of audit events that failed to be audited properly. Plugin identifies the plugin affected by the error.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">plugin</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_audit_event_total</div>
	<div class="metric_help">Counter of audit events generated and sent to the audit backend.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_audit_level_total</div>
	<div class="metric_help">Counter of policy levels for audit events (1 per request).</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_audit_requests_rejected_total</div>
	<div class="metric_help">Counter of apiserver requests rejected due to an error in audit logging backend.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_authentication_config_controller_automatic_reload_last_timestamp_seconds</div>
	<div class="metric_help">Timestamp of the last automatic reload of authentication configuration split by status and apiserver identity.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">apiserver_id_hash</span><span class="metric_label">status</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_authentication_config_controller_automatic_reloads_total</div>
	<div class="metric_help">Total number of automatic reloads of authentication configuration split by status and apiserver identity.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">apiserver_id_hash</span><span class="metric_label">status</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_authentication_jwt_authenticator_latency_seconds</div>
	<div class="metric_help">Latency of jwt authentication operations in seconds. This is the time spent authenticating a token for cache miss only (i.e. when the token is not found in the cache).</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">jwt_issuer_hash</span><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_authorization_config_controller_automatic_reload_last_timestamp_seconds</div>
	<div class="metric_help">Timestamp of the last automatic reload of authorization configuration split by status and apiserver identity.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">apiserver_id_hash</span><span class="metric_label">status</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_authorization_config_controller_automatic_reloads_total</div>
	<div class="metric_help">Total number of automatic reloads of authorization configuration split by status and apiserver identity.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">apiserver_id_hash</span><span class="metric_label">status</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_authorization_decisions_total</div>
	<div class="metric_help">Total number of terminal decisions made by an authorizer split by authorizer type, name, and decision.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">decision</span><span class="metric_label">name</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_authorization_match_condition_evaluation_errors_total</div>
	<div class="metric_help">Total number of errors when an authorization webhook encounters a match condition error split by authorizer type and name.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_authorization_match_condition_evaluation_seconds</div>
	<div class="metric_help">Authorization match condition evaluation time in seconds, split by authorizer type and name.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_authorization_match_condition_exclusions_total</div>
	<div class="metric_help">Total number of exclusions when an authorization webhook is skipped because match conditions exclude it.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_authorization_webhook_duration_seconds</div>
	<div class="metric_help">Request latency in seconds.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_authorization_webhook_evaluations_fail_open_total</div>
	<div class="metric_help">NoOpinion results due to webhook timeout or error.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_authorization_webhook_evaluations_total</div>
	<div class="metric_help">Round-trips to authorization webhooks.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_cache_list_fetched_objects_total</div>
	<div class="metric_help">Number of objects read from watch cache in the course of serving a LIST request</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">index</span><span class="metric_label">resource_prefix</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_cache_list_returned_objects_total</div>
	<div class="metric_help">Number of objects returned for a LIST request from watch cache</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource_prefix</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_cache_list_total</div>
	<div class="metric_help">Number of LIST requests served from watch cache</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">index</span><span class="metric_label">resource_prefix</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_cel_compilation_duration_seconds</div>
	<div class="metric_help">CEL compilation time in seconds.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_cel_evaluation_duration_seconds</div>
	<div class="metric_help">CEL evaluation time in seconds.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_certificates_registry_csr_honored_duration_total</div>
	<div class="metric_help">Total number of issued CSRs with a requested duration that was honored, sliced by signer (only kubernetes.io signer names are specifically identified)</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">signerName</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_certificates_registry_csr_requested_duration_total</div>
	<div class="metric_help">Total number of issued CSRs with a requested duration, sliced by signer (only kubernetes.io signer names are specifically identified)</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">signerName</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_client_certificate_expiration_seconds</div>
	<div class="metric_help">Distribution of the remaining lifetime on the certificate used to authenticate a request.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_clusterip_repair_ip_errors_total</div>
	<div class="metric_help">Number of errors detected on clusterips by the repair loop broken down by type of error: leak, repair, full, outOfRange, duplicate, unknown, invalid</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_clusterip_repair_reconcile_errors_total</div>
	<div class="metric_help">Number of reconciliation failures on the clusterip repair reconcile loop</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_conversion_webhook_duration_seconds</div>
	<div class="metric_help">Conversion webhook request latency</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">failure_type</span><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_conversion_webhook_request_total</div>
	<div class="metric_help">Counter for conversion webhook requests with success/failure and failure error type</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">failure_type</span><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_crd_conversion_webhook_duration_seconds</div>
	<div class="metric_help">CRD webhook conversion duration in seconds</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">crd_name</span><span class="metric_label">from_version</span><span class="metric_label">succeeded</span><span class="metric_label">to_version</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_current_inqueue_requests</div>
	<div class="metric_help">Maximal number of queued requests in this apiserver per request kind in last second.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">request_kind</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_delegated_authn_request_duration_seconds</div>
	<div class="metric_help">Request latency in seconds. Broken down by status code.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_delegated_authn_request_total</div>
	<div class="metric_help">Number of HTTP requests partitioned by status code.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_delegated_authz_request_duration_seconds</div>
	<div class="metric_help">Request latency in seconds. Broken down by status code.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_delegated_authz_request_total</div>
	<div class="metric_help">Number of HTTP requests partitioned by status code.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_egress_dialer_dial_duration_seconds</div>
	<div class="metric_help">Dial latency histogram in seconds, labeled by the protocol (http-connect or grpc), transport (tcp or uds)</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">protocol</span><span class="metric_label">transport</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_egress_dialer_dial_failure_count</div>
	<div class="metric_help">Dial failure count, labeled by the protocol (http-connect or grpc), transport (tcp or uds), and stage (connect or proxy). The stage indicates at which stage the dial failed</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">protocol</span><span class="metric_label">stage</span><span class="metric_label">transport</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_egress_dialer_dial_start_total</div>
	<div class="metric_help">Dial starts, labeled by the protocol (http-connect or grpc) and transport (tcp or uds).</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">protocol</span><span class="metric_label">transport</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_encryption_config_controller_automatic_reload_failures_total</div>
	<div class="metric_help">Total number of failed automatic reloads of encryption configuration split by apiserver identity.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">apiserver_id_hash</span></li><li class="metric_deprecated_version"><label class="metric_detail">Deprecated Versions:</label><span>1.30.0</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_encryption_config_controller_automatic_reload_last_timestamp_seconds</div>
	<div class="metric_help">Timestamp of the last successful or failed automatic reload of encryption configuration split by apiserver identity.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">apiserver_id_hash</span><span class="metric_label">status</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_encryption_config_controller_automatic_reload_success_total</div>
	<div class="metric_help">Total number of successful automatic reloads of encryption configuration split by apiserver identity.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">apiserver_id_hash</span></li><li class="metric_deprecated_version"><label class="metric_detail">Deprecated Versions:</label><span>1.30.0</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_encryption_config_controller_automatic_reloads_total</div>
	<div class="metric_help">Total number of reload successes and failures of encryption configuration split by apiserver identity.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">apiserver_id_hash</span><span class="metric_label">status</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_envelope_encryption_dek_cache_fill_percent</div>
	<div class="metric_help">Percent of the cache slots currently occupied by cached DEKs.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_envelope_encryption_dek_cache_inter_arrival_time_seconds</div>
	<div class="metric_help">Time (in seconds) of inter arrival of transformation requests.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">transformation_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_envelope_encryption_dek_source_cache_size</div>
	<div class="metric_help">Number of records in data encryption key (DEK) source cache. On a restart, this value is an approximation of the number of decrypt RPC calls the server will make to the KMS plugin.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">provider_name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_envelope_encryption_invalid_key_id_from_status_total</div>
	<div class="metric_help">Number of times an invalid keyID is returned by the Status RPC call split by error.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">error</span><span class="metric_label">provider_name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_envelope_encryption_key_id_hash_last_timestamp_seconds</div>
	<div class="metric_help">The last time in seconds when a keyID was used.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">apiserver_id_hash</span><span class="metric_label">key_id_hash</span><span class="metric_label">provider_name</span><span class="metric_label">transformation_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_envelope_encryption_key_id_hash_status_last_timestamp_seconds</div>
	<div class="metric_help">The last time in seconds when a keyID was returned by the Status RPC call.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">apiserver_id_hash</span><span class="metric_label">key_id_hash</span><span class="metric_label">provider_name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_envelope_encryption_key_id_hash_total</div>
	<div class="metric_help">Number of times a keyID is used split by transformation type, provider, and apiserver identity.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">apiserver_id_hash</span><span class="metric_label">key_id_hash</span><span class="metric_label">provider_name</span><span class="metric_label">transformation_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_envelope_encryption_kms_operations_latency_seconds</div>
	<div class="metric_help">KMS operation duration with gRPC error code status total.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">grpc_status_code</span><span class="metric_label">method_name</span><span class="metric_label">provider_name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_current_inqueue_seats</div>
	<div class="metric_help">Number of seats currently pending in queues of the API Priority and Fairness subsystem</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">flow_schema</span><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_current_limit_seats</div>
	<div class="metric_help">current derived number of execution seats available to each priority level</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_current_r</div>
	<div class="metric_help">R(time of last change)</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_demand_seats</div>
	<div class="metric_help">Observations, at the end of every nanosecond, of (the number of seats each priority level could use) / (nominal number of seats for that level)</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="timingratiohistogram"><label class="metric_detail">Type:</label> <span class="metric_type">TimingRatioHistogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_demand_seats_average</div>
	<div class="metric_help">Time-weighted average, over last adjustment period, of demand_seats</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_demand_seats_high_watermark</div>
	<div class="metric_help">High watermark, over last adjustment period, of demand_seats</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_demand_seats_smoothed</div>
	<div class="metric_help">Smoothed seat demands</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_demand_seats_stdev</div>
	<div class="metric_help">Time-weighted standard deviation, over last adjustment period, of demand_seats</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_dispatch_r</div>
	<div class="metric_help">R(time of last dispatch)</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_epoch_advance_total</div>
	<div class="metric_help">Number of times the queueset's progress meter jumped backward</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">priority_level</span><span class="metric_label">success</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_latest_s</div>
	<div class="metric_help">S(most recently dispatched request)</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_lower_limit_seats</div>
	<div class="metric_help">Configured lower bound on number of execution seats available to each priority level</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_next_discounted_s_bounds</div>
	<div class="metric_help">min and max, over queues, of S(oldest waiting request in queue) - estimated work in progress</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">bound</span><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_next_s_bounds</div>
	<div class="metric_help">min and max, over queues, of S(oldest waiting request in queue)</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">bound</span><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_priority_level_request_utilization</div>
	<div class="metric_help">Observations, at the end of every nanosecond, of number of requests (as a fraction of the relevant limit) waiting or in any stage of execution (but only initial stage for WATCHes)</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="timingratiohistogram"><label class="metric_detail">Type:</label> <span class="metric_type">TimingRatioHistogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">phase</span><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_priority_level_seat_utilization</div>
	<div class="metric_help">Observations, at the end of every nanosecond, of utilization of seats for any stage of execution (but only initial stage for WATCHes)</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="timingratiohistogram"><label class="metric_detail">Type:</label> <span class="metric_type">TimingRatioHistogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">priority_level</span></li><li class="metric_labels_constant"><label class="metric_detail">Const Labels:</label><span class="metric_label">phase:executing</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_read_vs_write_current_requests</div>
	<div class="metric_help">Observations, at the end of every nanosecond, of the number of requests (as a fraction of the relevant limit) waiting or in regular stage of execution</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="timingratiohistogram"><label class="metric_detail">Type:</label> <span class="metric_type">TimingRatioHistogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">phase</span><span class="metric_label">request_kind</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_request_concurrency_in_use</div>
	<div class="metric_help">Concurrency (number of seats) occupied by the currently executing (initial stage for a WATCH, any stage otherwise) requests in the API Priority and Fairness subsystem</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">flow_schema</span><span class="metric_label">priority_level</span></li><li class="metric_deprecated_version"><label class="metric_detail">Deprecated Versions:</label><span>1.31.0</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_request_concurrency_limit</div>
	<div class="metric_help">Nominal number of execution seats configured for each priority level</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">priority_level</span></li><li class="metric_deprecated_version"><label class="metric_detail">Deprecated Versions:</label><span>1.30.0</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_request_dispatch_no_accommodation_total</div>
	<div class="metric_help">Number of times a dispatch attempt resulted in a non accommodation due to lack of available seats</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">flow_schema</span><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_request_execution_seconds</div>
	<div class="metric_help">Duration of initial stage (for a WATCH) or any (for a non-WATCH) stage of request execution in the API Priority and Fairness subsystem</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">flow_schema</span><span class="metric_label">priority_level</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_request_queue_length_after_enqueue</div>
	<div class="metric_help">Length of queue in the API Priority and Fairness subsystem, as seen by each request after it is enqueued</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">flow_schema</span><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_seat_fair_frac</div>
	<div class="metric_help">Fair fraction of server's concurrency to allocate to each priority level that can use it</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_target_seats</div>
	<div class="metric_help">Seat allocation targets</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_upper_limit_seats</div>
	<div class="metric_help">Configured upper bound on number of execution seats available to each priority level</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_watch_count_samples</div>
	<div class="metric_help">count of watchers for mutating requests in API Priority and Fairness</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">flow_schema</span><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_flowcontrol_work_estimated_seats</div>
	<div class="metric_help">Number of estimated seats (maximum of initial and final seats) associated with requests in API Priority and Fairness</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">flow_schema</span><span class="metric_label">priority_level</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_init_events_total</div>
	<div class="metric_help">Counter of init events processed in watch cache broken by resource type.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_kube_aggregator_x509_insecure_sha1_total</div>
	<div class="metric_help">Counts the number of requests to servers with insecure SHA1 signatures in their serving certificate OR the number of connection failures due to the insecure SHA1 signatures (either/or, based on the runtime environment)</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_kube_aggregator_x509_missing_san_total</div>
	<div class="metric_help">Counts the number of requests to servers missing SAN extension in their serving certificate OR the number of connection failures due to the lack of x509 certificate SAN extension missing (either/or, based on the runtime environment)</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_nodeport_repair_port_errors_total</div>
	<div class="metric_help">Number of errors detected on ports by the repair loop broken down by type of error: leak, repair, full, outOfRange, duplicate, unknown</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_nodeport_repair_reconcile_errors_total</div>
	<div class="metric_help">Number of reconciliation failures on the nodeport repair reconcile loop</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_request_aborts_total</div>
	<div class="metric_help">Number of requests which apiserver aborted possibly due to a timeout, for each group, version, verb, resource, subresource and scope</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">group</span><span class="metric_label">resource</span><span class="metric_label">scope</span><span class="metric_label">subresource</span><span class="metric_label">verb</span><span class="metric_label">version</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_request_body_size_bytes</div>
	<div class="metric_help">Apiserver request body size in bytes broken out by resource and verb.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span><span class="metric_label">verb</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_request_filter_duration_seconds</div>
	<div class="metric_help">Request filter latency distribution in seconds, for each filter type</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">filter</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_request_post_timeout_total</div>
	<div class="metric_help">Tracks the activity of the request handlers after the associated requests have been timed out by the apiserver</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">source</span><span class="metric_label">status</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_request_sli_duration_seconds</div>
	<div class="metric_help">Response latency distribution (not counting webhook duration and priority & fairness queue wait times) in seconds for each verb, group, version, resource, subresource, scope and component.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">component</span><span class="metric_label">group</span><span class="metric_label">resource</span><span class="metric_label">scope</span><span class="metric_label">subresource</span><span class="metric_label">verb</span><span class="metric_label">version</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_request_slo_duration_seconds</div>
	<div class="metric_help">Response latency distribution (not counting webhook duration and priority & fairness queue wait times) in seconds for each verb, group, version, resource, subresource, scope and component.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">component</span><span class="metric_label">group</span><span class="metric_label">resource</span><span class="metric_label">scope</span><span class="metric_label">subresource</span><span class="metric_label">verb</span><span class="metric_label">version</span></li><li class="metric_deprecated_version"><label class="metric_detail">Deprecated Versions:</label><span>1.27.0</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_request_terminations_total</div>
	<div class="metric_help">Number of requests which apiserver terminated in self-defense.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span><span class="metric_label">component</span><span class="metric_label">group</span><span class="metric_label">resource</span><span class="metric_label">scope</span><span class="metric_label">subresource</span><span class="metric_label">verb</span><span class="metric_label">version</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_request_timestamp_comparison_time</div>
	<div class="metric_help">Time taken for comparison of old vs new objects in UPDATE or PATCH requests</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code_path</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_rerouted_request_total</div>
	<div class="metric_help">Total number of requests that were proxied to a peer kube apiserver because the local apiserver was not capable of serving it</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_selfrequest_total</div>
	<div class="metric_help">Counter of apiserver self-requests broken out for each verb, API resource and subresource.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span><span class="metric_label">subresource</span><span class="metric_label">verb</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_storage_data_key_generation_duration_seconds</div>
	<div class="metric_help">Latencies in seconds of data encryption key(DEK) generation operations.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_storage_data_key_generation_failures_total</div>
	<div class="metric_help">Total number of failed data encryption key(DEK) generation operations.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_storage_db_total_size_in_bytes</div>
	<div class="metric_help">Total size of the storage database file physically allocated in bytes.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">endpoint</span></li><li class="metric_deprecated_version"><label class="metric_detail">Deprecated Versions:</label><span>1.28.0</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_storage_decode_errors_total</div>
	<div class="metric_help">Number of stored object decode errors split by object type</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_storage_envelope_transformation_cache_misses_total</div>
	<div class="metric_help">Total number of cache misses while accessing key decryption key(KEK).</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_storage_events_received_total</div>
	<div class="metric_help">Number of etcd events received split by kind.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_storage_list_evaluated_objects_total</div>
	<div class="metric_help">Number of objects tested in the course of serving a LIST request from storage</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_storage_list_fetched_objects_total</div>
	<div class="metric_help">Number of objects read from storage in the course of serving a LIST request</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_storage_list_returned_objects_total</div>
	<div class="metric_help">Number of objects returned for a LIST request from storage</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_storage_list_total</div>
	<div class="metric_help">Number of LIST requests served from storage</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_storage_transformation_duration_seconds</div>
	<div class="metric_help">Latencies in seconds of value transformation operations.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">transformation_type</span><span class="metric_label">transformer_prefix</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_storage_transformation_operations_total</div>
	<div class="metric_help">Total number of transformations. Successful transformation will have a status 'OK' and a varied status string when the transformation fails. This status and transformation_type fields may be used for alerting on encryption/decryption failure using transformation_type from_storage for decryption and to_storage for encryption</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">status</span><span class="metric_label">transformation_type</span><span class="metric_label">transformer_prefix</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_stream_translator_requests_total</div>
	<div class="metric_help">Total number of requests that were handled by the StreamTranslatorProxy, which processes streaming RemoteCommand/V5</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_terminated_watchers_total</div>
	<div class="metric_help">Counter of watchers closed due to unresponsiveness broken by resource type.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_tls_handshake_errors_total</div>
	<div class="metric_help">Number of requests dropped with 'TLS handshake error from' error</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_validating_admission_policy_check_duration_seconds</div>
	<div class="metric_help">Validation admission latency for individual validation expressions in seconds, labeled by policy and further including binding, state and enforcement action taken.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">enforcement_action</span><span class="metric_label">policy</span><span class="metric_label">policy_binding</span><span class="metric_label">state</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_validating_admission_policy_check_total</div>
	<div class="metric_help">Validation admission policy check total, labeled by policy and further identified by binding, enforcement action taken, and state.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">enforcement_action</span><span class="metric_label">policy</span><span class="metric_label">policy_binding</span><span class="metric_label">state</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_validating_admission_policy_definition_total</div>
	<div class="metric_help">Validation admission policy count total, labeled by state and enforcement action.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">enforcement_action</span><span class="metric_label">state</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_watch_cache_events_dispatched_total</div>
	<div class="metric_help">Counter of events dispatched in watch cache broken by resource type.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_watch_cache_events_received_total</div>
	<div class="metric_help">Counter of events received in watch cache broken by resource type.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_watch_cache_initializations_total</div>
	<div class="metric_help">Counter of watch cache initializations broken by resource type.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_watch_cache_read_wait_seconds</div>
	<div class="metric_help">Histogram of time spent waiting for a watch cache to become fresh.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_watch_events_sizes</div>
	<div class="metric_help">Watch event size distribution in bytes</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">group</span><span class="metric_label">kind</span><span class="metric_label">version</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_watch_events_total</div>
	<div class="metric_help">Number of events sent in watch clients</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">group</span><span class="metric_label">kind</span><span class="metric_label">version</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_watch_list_duration_seconds</div>
	<div class="metric_help">Response latency distribution in seconds for watch list requests broken by group, version, resource and scope.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">group</span><span class="metric_label">resource</span><span class="metric_label">scope</span><span class="metric_label">version</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_webhooks_x509_insecure_sha1_total</div>
	<div class="metric_help">Counts the number of requests to servers with insecure SHA1 signatures in their serving certificate OR the number of connection failures due to the insecure SHA1 signatures (either/or, based on the runtime environment)</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">apiserver_webhooks_x509_missing_san_total</div>
	<div class="metric_help">Counts the number of requests to servers missing SAN extension in their serving certificate OR the number of connection failures due to the lack of x509 certificate SAN extension missing (either/or, based on the runtime environment)</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">attach_detach_controller_attachdetach_controller_forced_detaches</div>
	<div class="metric_help">Number of times the A/D Controller performed a forced detach</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">reason</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">attachdetach_controller_total_volumes</div>
	<div class="metric_help">Number of volumes in A/D Controller</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">plugin_name</span><span class="metric_label">state</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">authenticated_user_requests</div>
	<div class="metric_help">Counter of authenticated requests broken out by username.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">username</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">authentication_attempts</div>
	<div class="metric_help">Counter of authenticated attempts.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">authentication_duration_seconds</div>
	<div class="metric_help">Authentication duration in seconds broken out by result.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">authentication_token_cache_active_fetch_count</div>
	<div class="metric_help"></div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">status</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">authentication_token_cache_fetch_total</div>
	<div class="metric_help"></div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">status</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">authentication_token_cache_request_duration_seconds</div>
	<div class="metric_help"></div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">status</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">authentication_token_cache_request_total</div>
	<div class="metric_help"></div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">status</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">authorization_attempts_total</div>
	<div class="metric_help">Counter of authorization attempts broken down by result. It can be either 'allowed', 'denied', 'no-opinion' or 'error'.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">authorization_duration_seconds</div>
	<div class="metric_help">Authorization duration in seconds broken out by result.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">cloud_provider_webhook_request_duration_seconds</div>
	<div class="metric_help">Request latency in seconds. Broken down by status code.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span><span class="metric_label">webhook</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">cloud_provider_webhook_request_total</div>
	<div class="metric_help">Number of HTTP requests partitioned by status code.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span><span class="metric_label">webhook</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">cloudprovider_gce_api_request_duration_seconds</div>
	<div class="metric_help">Latency of a GCE API call</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">region</span><span class="metric_label">request</span><span class="metric_label">version</span><span class="metric_label">zone</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">cloudprovider_gce_api_request_errors</div>
	<div class="metric_help">Number of errors for an API call</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">region</span><span class="metric_label">request</span><span class="metric_label">version</span><span class="metric_label">zone</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">container_swap_usage_bytes</div>
	<div class="metric_help">Current amount of the container swap usage in bytes. Reported only on non-windows systems</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">container</span><span class="metric_label">pod</span><span class="metric_label">namespace</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">csi_operations_seconds</div>
	<div class="metric_help">Container Storage Interface operation duration with gRPC error code status total</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">driver_name</span><span class="metric_label">grpc_status_code</span><span class="metric_label">method_name</span><span class="metric_label">migrated</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_controller_changes</div>
	<div class="metric_help">Number of EndpointSlice changes</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_controller_desired_endpoint_slices</div>
	<div class="metric_help">Number of EndpointSlices that would exist with perfect endpoint allocation</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_controller_endpoints_added_per_sync</div>
	<div class="metric_help">Number of endpoints added on each Service sync</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_controller_endpoints_desired</div>
	<div class="metric_help">Number of endpoints desired</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_controller_endpoints_removed_per_sync</div>
	<div class="metric_help">Number of endpoints removed on each Service sync</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_controller_endpointslices_changed_per_sync</div>
	<div class="metric_help">Number of EndpointSlices changed on each Service sync</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">topology</span><span class="metric_label">traffic_distribution</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_controller_num_endpoint_slices</div>
	<div class="metric_help">Number of EndpointSlices</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_controller_services_count_by_traffic_distribution</div>
	<div class="metric_help">Number of Services using some specific trafficDistribution</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">traffic_distribution</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_controller_syncs</div>
	<div class="metric_help">Number of EndpointSlice syncs</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_mirroring_controller_addresses_skipped_per_sync</div>
	<div class="metric_help">Number of addresses skipped on each Endpoints sync due to being invalid or exceeding MaxEndpointsPerSubset</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_mirroring_controller_changes</div>
	<div class="metric_help">Number of EndpointSlice changes</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_mirroring_controller_desired_endpoint_slices</div>
	<div class="metric_help">Number of EndpointSlices that would exist with perfect endpoint allocation</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_mirroring_controller_endpoints_added_per_sync</div>
	<div class="metric_help">Number of endpoints added on each Endpoints sync</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_mirroring_controller_endpoints_desired</div>
	<div class="metric_help">Number of endpoints desired</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_mirroring_controller_endpoints_removed_per_sync</div>
	<div class="metric_help">Number of endpoints removed on each Endpoints sync</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_mirroring_controller_endpoints_sync_duration</div>
	<div class="metric_help">Duration of syncEndpoints() in seconds</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_mirroring_controller_endpoints_updated_per_sync</div>
	<div class="metric_help">Number of endpoints updated on each Endpoints sync</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">endpoint_slice_mirroring_controller_num_endpoint_slices</div>
	<div class="metric_help">Number of EndpointSlices</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">ephemeral_volume_controller_create_failures_total</div>
	<div class="metric_help">Number of PersistenVolumeClaims creation requests</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">ephemeral_volume_controller_create_total</div>
	<div class="metric_help">Number of PersistenVolumeClaims creation requests</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">etcd_bookmark_counts</div>
	<div class="metric_help">Number of etcd bookmarks (progress notify events) split by kind.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">etcd_lease_object_counts</div>
	<div class="metric_help">Number of objects attached to a single etcd lease.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">etcd_request_duration_seconds</div>
	<div class="metric_help">Etcd request latency in seconds for each operation and object type.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">etcd_request_errors_total</div>
	<div class="metric_help">Etcd failed request counts for each operation and object type.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">etcd_requests_total</div>
	<div class="metric_help">Etcd request counts for each operation and object type.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation</span><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">etcd_version_info</div>
	<div class="metric_help">Etcd server's binary version</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">binary_version</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">field_validation_request_duration_seconds</div>
	<div class="metric_help">Response latency distribution in seconds for each field validation value</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">field_validation</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">force_cleaned_failed_volume_operation_errors_total</div>
	<div class="metric_help">The number of volumes that failed force cleanup after their reconstruction failed during kubelet startup.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">force_cleaned_failed_volume_operations_total</div>
	<div class="metric_help">The number of volumes that were force cleaned after their reconstruction failed during kubelet startup. This includes both successful and failed cleanups.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">garbagecollector_controller_resources_sync_error_total</div>
	<div class="metric_help">Number of garbage collector resources sync errors</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">get_token_count</div>
	<div class="metric_help">Counter of total Token() requests to the alternate token source</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">get_token_fail_count</div>
	<div class="metric_help">Counter of failed Token() requests to the alternate token source</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">horizontal_pod_autoscaler_controller_metric_computation_duration_seconds</div>
	<div class="metric_help">The time(seconds) that the HPA controller takes to calculate one metric. The label 'action' should be either 'scale_down', 'scale_up', or 'none'. The label 'error' should be either 'spec', 'internal', or 'none'. The label 'metric_type' corresponds to HPA.spec.metrics[*].type</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">action</span><span class="metric_label">error</span><span class="metric_label">metric_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">horizontal_pod_autoscaler_controller_metric_computation_total</div>
	<div class="metric_help">Number of metric computations. The label 'action' should be either 'scale_down', 'scale_up', or 'none'. Also, the label 'error' should be either 'spec', 'internal', or 'none'. The label 'metric_type' corresponds to HPA.spec.metrics[*].type</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">action</span><span class="metric_label">error</span><span class="metric_label">metric_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">horizontal_pod_autoscaler_controller_reconciliation_duration_seconds</div>
	<div class="metric_help">The time(seconds) that the HPA controller takes to reconcile once. The label 'action' should be either 'scale_down', 'scale_up', or 'none'. Also, the label 'error' should be either 'spec', 'internal', or 'none'. Note that if both spec and internal errors happen during a reconciliation, the first one to occur is reported in `error` label.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">action</span><span class="metric_label">error</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">horizontal_pod_autoscaler_controller_reconciliations_total</div>
	<div class="metric_help">Number of reconciliations of HPA controller. The label 'action' should be either 'scale_down', 'scale_up', or 'none'. Also, the label 'error' should be either 'spec', 'internal', or 'none'. Note that if both spec and internal errors happen during a reconciliation, the first one to occur is reported in `error` label.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">action</span><span class="metric_label">error</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">job_controller_job_finished_indexes_total</div>
	<div class="metric_help">`The number of finished indexes. Possible values for the, 			status label are: "succeeded", "failed". Possible values for the, 			backoffLimit label are: "perIndex" and "global"`</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">backoffLimit</span><span class="metric_label">status</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">job_controller_job_pods_creation_total</div>
	<div class="metric_help">`The number of Pods created by the Job controller labelled with a reason for the Pod creation., This metric also distinguishes between Pods created using different PodReplacementPolicy settings., Possible values of the "reason" label are:, "new", "recreate_terminating_or_failed", "recreate_failed"., Possible values of the "status" label are:, "succeeded", "failed".`</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">reason</span><span class="metric_label">status</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">job_controller_jobs_by_external_controller_total</div>
	<div class="metric_help">The number of Jobs managed by an external controller</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">controller_name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">job_controller_pod_failures_handled_by_failure_policy_total</div>
	<div class="metric_help">`The number of failed Pods handled by failure policy with, 			respect to the failure policy action applied based on the matched, 			rule. Possible values of the action label correspond to the, 			possible values for the failure policy rule action, which are:, 			"FailJob", "Ignore" and "Count".`</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">action</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">job_controller_terminated_pods_tracking_finalizer_total</div>
	<div class="metric_help">`The number of terminated pods (phase=Failed|Succeeded), that have the finalizer batch.kubernetes.io/job-tracking, The event label can be "add" or "delete".`</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">event</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kube_apiserver_clusterip_allocator_allocated_ips</div>
	<div class="metric_help">Gauge measuring the number of allocated IPs for Services</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">cidr</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kube_apiserver_clusterip_allocator_allocation_errors_total</div>
	<div class="metric_help">Number of errors trying to allocate Cluster IPs</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">cidr</span><span class="metric_label">scope</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kube_apiserver_clusterip_allocator_allocation_total</div>
	<div class="metric_help">Number of Cluster IPs allocations</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">cidr</span><span class="metric_label">scope</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kube_apiserver_clusterip_allocator_available_ips</div>
	<div class="metric_help">Gauge measuring the number of available IPs for Services</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">cidr</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kube_apiserver_nodeport_allocator_allocated_ports</div>
	<div class="metric_help">Gauge measuring the number of allocated NodePorts for Services</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kube_apiserver_nodeport_allocator_allocation_errors_total</div>
	<div class="metric_help">Number of errors trying to allocate NodePort</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">scope</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kube_apiserver_nodeport_allocator_allocation_total</div>
	<div class="metric_help">Number of NodePort allocations</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">scope</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kube_apiserver_nodeport_allocator_available_ports</div>
	<div class="metric_help">Gauge measuring the number of available NodePorts for Services</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kube_apiserver_pod_logs_backend_tls_failure_total</div>
	<div class="metric_help">Total number of requests for pods/logs that failed due to kubelet server TLS verification</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kube_apiserver_pod_logs_insecure_backend_total</div>
	<div class="metric_help">Total number of requests for pods/logs sliced by usage type: enforce_tls, skip_tls_allowed, skip_tls_denied</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">usage</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kube_apiserver_pod_logs_pods_logs_backend_tls_failure_total</div>
	<div class="metric_help">Total number of requests for pods/logs that failed due to kubelet server TLS verification</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_deprecated_version"><label class="metric_detail">Deprecated Versions:</label><span>1.27.0</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kube_apiserver_pod_logs_pods_logs_insecure_backend_total</div>
	<div class="metric_help">Total number of requests for pods/logs sliced by usage type: enforce_tls, skip_tls_allowed, skip_tls_denied</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">usage</span></li><li class="metric_deprecated_version"><label class="metric_detail">Deprecated Versions:</label><span>1.27.0</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_active_pods</div>
	<div class="metric_help">The number of pods the kubelet considers active and which are being considered when admitting new pods. static is true if the pod is not from the apiserver.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">static</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_certificate_manager_client_expiration_renew_errors</div>
	<div class="metric_help">Counter of certificate renewal errors.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_certificate_manager_client_ttl_seconds</div>
	<div class="metric_help">Gauge of the TTL (time-to-live) of the Kubelet's client certificate. The value is in seconds until certificate expiry (negative if already expired). If client certificate is invalid or unused, the value will be +INF.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_certificate_manager_server_rotation_seconds</div>
	<div class="metric_help">Histogram of the number of seconds the previous certificate lived before being rotated.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_certificate_manager_server_ttl_seconds</div>
	<div class="metric_help">Gauge of the shortest TTL (time-to-live) of the Kubelet's serving certificate. The value is in seconds until certificate expiry (negative if already expired). If serving certificate is invalid or unused, the value will be +INF.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_cgroup_manager_duration_seconds</div>
	<div class="metric_help">Duration in seconds for cgroup manager operations. Broken down by method.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_container_log_filesystem_used_bytes</div>
	<div class="metric_help">Bytes used by the container's logs on the filesystem.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">uid</span><span class="metric_label">namespace</span><span class="metric_label">pod</span><span class="metric_label">container</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_containers_per_pod_count</div>
	<div class="metric_help">The number of containers per pod.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_cpu_manager_pinning_errors_total</div>
	<div class="metric_help">The number of cpu core allocations which required pinning failed.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_cpu_manager_pinning_requests_total</div>
	<div class="metric_help">The number of cpu core allocations which required pinning.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_credential_provider_plugin_duration</div>
	<div class="metric_help">Duration of execution in seconds for credential provider plugin</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">plugin_name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_credential_provider_plugin_errors</div>
	<div class="metric_help">Number of errors from credential provider plugin</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">plugin_name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_desired_pods</div>
	<div class="metric_help">The number of pods the kubelet is being instructed to run. static is true if the pod is not from the apiserver.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">static</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_device_plugin_alloc_duration_seconds</div>
	<div class="metric_help">Duration in seconds to serve a device plugin Allocation request. Broken down by resource name.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource_name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_device_plugin_registration_total</div>
	<div class="metric_help">Cumulative number of device plugin registrations. Broken down by resource name.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource_name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_evented_pleg_connection_error_count</div>
	<div class="metric_help">The number of errors encountered during the establishment of streaming connection with the CRI runtime.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_evented_pleg_connection_latency_seconds</div>
	<div class="metric_help">The latency of streaming connection with the CRI runtime, measured in seconds.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_evented_pleg_connection_success_count</div>
	<div class="metric_help">The number of times a streaming client was obtained to receive CRI Events.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_eviction_stats_age_seconds</div>
	<div class="metric_help">Time between when stats are collected, and when pod is evicted based on those stats by eviction signal</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">eviction_signal</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_evictions</div>
	<div class="metric_help">Cumulative number of pod evictions by eviction signal</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">eviction_signal</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_graceful_shutdown_end_time_seconds</div>
	<div class="metric_help">Last graceful shutdown start time since unix epoch in seconds</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_graceful_shutdown_start_time_seconds</div>
	<div class="metric_help">Last graceful shutdown start time since unix epoch in seconds</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_http_inflight_requests</div>
	<div class="metric_help">Number of the inflight http requests</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">long_running</span><span class="metric_label">method</span><span class="metric_label">path</span><span class="metric_label">server_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_http_requests_duration_seconds</div>
	<div class="metric_help">Duration in seconds to serve http requests</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">long_running</span><span class="metric_label">method</span><span class="metric_label">path</span><span class="metric_label">server_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_http_requests_total</div>
	<div class="metric_help">Number of the http requests received since the server started</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">long_running</span><span class="metric_label">method</span><span class="metric_label">path</span><span class="metric_label">server_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_image_garbage_collected_total</div>
	<div class="metric_help">Total number of images garbage collected by the kubelet, whether through disk usage or image age.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">reason</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_image_pull_duration_seconds</div>
	<div class="metric_help">Duration in seconds to pull an image.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">image_size_in_bytes</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_lifecycle_handler_http_fallbacks_total</div>
	<div class="metric_help">The number of times lifecycle handlers successfully fell back to http from https.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_managed_ephemeral_containers</div>
	<div class="metric_help">Current number of ephemeral containers in pods managed by this kubelet.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_memory_manager_pinning_errors_total</div>
	<div class="metric_help">The number of memory pages allocations which required pinning that failed.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_memory_manager_pinning_requests_total</div>
	<div class="metric_help">The number of memory pages allocations which required pinning.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_mirror_pods</div>
	<div class="metric_help">The number of mirror pods the kubelet will try to create (one per admitted static pod)</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_node_name</div>
	<div class="metric_help">The node's name. The count is always 1.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">node</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_node_startup_duration_seconds</div>
	<div class="metric_help">Duration in seconds of node startup in total.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_node_startup_post_registration_duration_seconds</div>
	<div class="metric_help">Duration in seconds of node startup after registration.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_node_startup_pre_kubelet_duration_seconds</div>
	<div class="metric_help">Duration in seconds of node startup before kubelet starts.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_node_startup_pre_registration_duration_seconds</div>
	<div class="metric_help">Duration in seconds of node startup before registration.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_node_startup_registration_duration_seconds</div>
	<div class="metric_help">Duration in seconds of node startup during registration.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_orphan_pod_cleaned_volumes</div>
	<div class="metric_help">The total number of orphaned Pods whose volumes were cleaned in the last periodic sweep.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_orphan_pod_cleaned_volumes_errors</div>
	<div class="metric_help">The number of orphaned Pods whose volumes failed to be cleaned in the last periodic sweep.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_orphaned_runtime_pods_total</div>
	<div class="metric_help">Number of pods that have been detected in the container runtime without being already known to the pod worker. This typically indicates the kubelet was restarted while a pod was force deleted in the API or in the local configuration, which is unusual.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_pleg_discard_events</div>
	<div class="metric_help">The number of discard events in PLEG.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_pleg_last_seen_seconds</div>
	<div class="metric_help">Timestamp in seconds when PLEG was last seen active.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_pleg_relist_duration_seconds</div>
	<div class="metric_help">Duration in seconds for relisting pods in PLEG.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_pleg_relist_interval_seconds</div>
	<div class="metric_help">Interval in seconds between relisting in PLEG.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_pod_resources_endpoint_errors_get</div>
	<div class="metric_help">Number of requests to the PodResource Get endpoint which returned error. Broken down by server api version.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">server_api_version</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_pod_resources_endpoint_errors_get_allocatable</div>
	<div class="metric_help">Number of requests to the PodResource GetAllocatableResources endpoint which returned error. Broken down by server api version.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">server_api_version</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_pod_resources_endpoint_errors_list</div>
	<div class="metric_help">Number of requests to the PodResource List endpoint which returned error. Broken down by server api version.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">server_api_version</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_pod_resources_endpoint_requests_get</div>
	<div class="metric_help">Number of requests to the PodResource Get endpoint. Broken down by server api version.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">server_api_version</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_pod_resources_endpoint_requests_get_allocatable</div>
	<div class="metric_help">Number of requests to the PodResource GetAllocatableResources endpoint. Broken down by server api version.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">server_api_version</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_pod_resources_endpoint_requests_list</div>
	<div class="metric_help">Number of requests to the PodResource List endpoint. Broken down by server api version.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">server_api_version</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_pod_resources_endpoint_requests_total</div>
	<div class="metric_help">Cumulative number of requests to the PodResource endpoint. Broken down by server api version.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">server_api_version</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_pod_start_duration_seconds</div>
	<div class="metric_help">Duration in seconds from kubelet seeing a pod for the first time to the pod starting to run</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_pod_start_sli_duration_seconds</div>
	<div class="metric_help">Duration in seconds to start a pod, excluding time to pull images and run init containers, measured from pod creation timestamp to when all its containers are reported as started and observed via watch</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_pod_start_total_duration_seconds</div>
	<div class="metric_help">Duration in seconds to start a pod since creation, including time to pull images and run init containers, measured from pod creation timestamp to when all its containers are reported as started and observed via watch</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_pod_status_sync_duration_seconds</div>
	<div class="metric_help">Duration in seconds to sync a pod status update. Measures time from detection of a change to pod status until the API is successfully updated for that pod, even if multiple intevening changes to pod status occur.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_pod_worker_duration_seconds</div>
	<div class="metric_help">Duration in seconds to sync a single pod. Broken down by operation type: create, update, or sync</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_pod_worker_start_duration_seconds</div>
	<div class="metric_help">Duration in seconds from kubelet seeing a pod to starting a worker.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_preemptions</div>
	<div class="metric_help">Cumulative number of pod preemptions by preemption resource</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">preemption_signal</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_restarted_pods_total</div>
	<div class="metric_help">Number of pods that have been restarted because they were deleted and recreated with the same UID while the kubelet was watching them (common for static pods, extremely uncommon for API pods)</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">static</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_run_podsandbox_duration_seconds</div>
	<div class="metric_help">Duration in seconds of the run_podsandbox operations. Broken down by RuntimeClass.Handler.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">runtime_handler</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_run_podsandbox_errors_total</div>
	<div class="metric_help">Cumulative number of the run_podsandbox operation errors by RuntimeClass.Handler.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">runtime_handler</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_running_containers</div>
	<div class="metric_help">Number of containers currently running</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">container_state</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_running_pods</div>
	<div class="metric_help">Number of pods that have a running pod sandbox</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_runtime_operations_duration_seconds</div>
	<div class="metric_help">Duration in seconds of runtime operations. Broken down by operation type.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_runtime_operations_errors_total</div>
	<div class="metric_help">Cumulative number of runtime operation errors by operation type.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_runtime_operations_total</div>
	<div class="metric_help">Cumulative number of runtime operations by operation type.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_server_expiration_renew_errors</div>
	<div class="metric_help">Counter of certificate renewal errors.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_sleep_action_terminated_early_total</div>
	<div class="metric_help">The number of times lifecycle sleep handler got terminated before it finishes</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_started_containers_errors_total</div>
	<div class="metric_help">Cumulative number of errors when starting containers</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span><span class="metric_label">container_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_started_containers_total</div>
	<div class="metric_help">Cumulative number of containers started</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">container_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_started_host_process_containers_errors_total</div>
	<div class="metric_help">Cumulative number of errors when starting hostprocess containers. This metric will only be collected on Windows.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span><span class="metric_label">container_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_started_host_process_containers_total</div>
	<div class="metric_help">Cumulative number of hostprocess containers started. This metric will only be collected on Windows.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">container_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_started_pods_errors_total</div>
	<div class="metric_help">Cumulative number of errors when starting pods</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_started_pods_total</div>
	<div class="metric_help">Cumulative number of pods started</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_topology_manager_admission_duration_ms</div>
	<div class="metric_help">Duration in milliseconds to serve a pod admission request.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_topology_manager_admission_errors_total</div>
	<div class="metric_help">The number of admission request failures where resources could not be aligned.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_topology_manager_admission_requests_total</div>
	<div class="metric_help">The number of admission requests where resources have to be aligned.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_volume_metric_collection_duration_seconds</div>
	<div class="metric_help">Duration in seconds to calculate volume stats</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">metric_source</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_volume_stats_available_bytes</div>
	<div class="metric_help">Number of available bytes in the volume</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">namespace</span><span class="metric_label">persistentvolumeclaim</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_volume_stats_capacity_bytes</div>
	<div class="metric_help">Capacity in bytes of the volume</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">namespace</span><span class="metric_label">persistentvolumeclaim</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_volume_stats_health_status_abnormal</div>
	<div class="metric_help">Abnormal volume health status. The count is either 1 or 0. 1 indicates the volume is unhealthy, 0 indicates volume is healthy</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">namespace</span><span class="metric_label">persistentvolumeclaim</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_volume_stats_inodes</div>
	<div class="metric_help">Maximum number of inodes in the volume</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">namespace</span><span class="metric_label">persistentvolumeclaim</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_volume_stats_inodes_free</div>
	<div class="metric_help">Number of free inodes in the volume</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">namespace</span><span class="metric_label">persistentvolumeclaim</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_volume_stats_inodes_used</div>
	<div class="metric_help">Number of used inodes in the volume</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">namespace</span><span class="metric_label">persistentvolumeclaim</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_volume_stats_used_bytes</div>
	<div class="metric_help">Number of used bytes in the volume</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">namespace</span><span class="metric_label">persistentvolumeclaim</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubelet_working_pods</div>
	<div class="metric_help">Number of pods the kubelet is actually running, broken down by lifecycle phase, whether the pod is desired, orphaned, or runtime only (also orphaned), and whether the pod is static. An orphaned pod has been removed from local configuration or force deleted in the API and consumes resources that are not otherwise visible.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">config</span><span class="metric_label">lifecycle</span><span class="metric_label">static</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubeproxy_network_programming_duration_seconds</div>
	<div class="metric_help">In Cluster Network Programming Latency in seconds</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubeproxy_proxy_healthz_total</div>
	<div class="metric_help">Cumulative proxy healthz HTTP status</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubeproxy_proxy_livez_total</div>
	<div class="metric_help">Cumulative proxy livez HTTP status</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubeproxy_sync_full_proxy_rules_duration_seconds</div>
	<div class="metric_help">SyncProxyRules latency in seconds for full resyncs</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubeproxy_sync_partial_proxy_rules_duration_seconds</div>
	<div class="metric_help">SyncProxyRules latency in seconds for partial resyncs</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubeproxy_sync_proxy_rules_duration_seconds</div>
	<div class="metric_help">SyncProxyRules latency in seconds</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubeproxy_sync_proxy_rules_endpoint_changes_pending</div>
	<div class="metric_help">Pending proxy rules Endpoint changes</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubeproxy_sync_proxy_rules_endpoint_changes_total</div>
	<div class="metric_help">Cumulative proxy rules Endpoint changes</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubeproxy_sync_proxy_rules_iptables_last</div>
	<div class="metric_help">Number of iptables rules written by kube-proxy in last sync</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">table</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubeproxy_sync_proxy_rules_iptables_partial_restore_failures_total</div>
	<div class="metric_help">Cumulative proxy iptables partial restore failures</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubeproxy_sync_proxy_rules_iptables_restore_failures_total</div>
	<div class="metric_help">Cumulative proxy iptables restore failures</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubeproxy_sync_proxy_rules_iptables_total</div>
	<div class="metric_help">Total number of iptables rules owned by kube-proxy</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">table</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubeproxy_sync_proxy_rules_last_queued_timestamp_seconds</div>
	<div class="metric_help">The last time a sync of proxy rules was queued</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubeproxy_sync_proxy_rules_last_timestamp_seconds</div>
	<div class="metric_help">The last time proxy rules were successfully synced</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubeproxy_sync_proxy_rules_no_local_endpoints_total</div>
	<div class="metric_help">Number of services with a Local traffic policy and no endpoints</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">traffic_policy</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubeproxy_sync_proxy_rules_service_changes_pending</div>
	<div class="metric_help">Pending proxy rules Service changes</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubeproxy_sync_proxy_rules_service_changes_total</div>
	<div class="metric_help">Cumulative proxy rules Service changes</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">kubernetes_build_info</div>
	<div class="metric_help">A metric with a constant '1' value labeled by major, minor, git version, git commit, git tree state, build date, Go version, and compiler from which Kubernetes was built, and platform on which it is running.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">build_date</span><span class="metric_label">compiler</span><span class="metric_label">git_commit</span><span class="metric_label">git_tree_state</span><span class="metric_label">git_version</span><span class="metric_label">go_version</span><span class="metric_label">major</span><span class="metric_label">minor</span><span class="metric_label">platform</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">leader_election_master_status</div>
	<div class="metric_help">Gauge of if the reporting system is master of the relevant lease, 0 indicates backup, 1 indicates master. 'name' is the string used to identify the lease. Please make sure to group by name.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">leader_election_slowpath_total</div>
	<div class="metric_help">Total number of slow path exercised in renewing leader leases. 'name' is the string used to identify the lease. Please make sure to group by name.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">node_authorizer_graph_actions_duration_seconds</div>
	<div class="metric_help">Histogram of duration of graph actions in node authorizer.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">node_collector_unhealthy_nodes_in_zone</div>
	<div class="metric_help">Gauge measuring number of not Ready Nodes per zones.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">zone</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">node_collector_update_all_nodes_health_duration_seconds</div>
	<div class="metric_help">Duration in seconds for NodeController to update the health of all nodes.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">node_collector_update_node_health_duration_seconds</div>
	<div class="metric_help">Duration in seconds for NodeController to update the health of a single node.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">node_collector_zone_health</div>
	<div class="metric_help">Gauge measuring percentage of healthy nodes per zone.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">zone</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">node_collector_zone_size</div>
	<div class="metric_help">Gauge measuring number of registered Nodes per zones.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">zone</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">node_controller_cloud_provider_taint_removal_delay_seconds</div>
	<div class="metric_help">Number of seconds after node creation when NodeController removed the cloud-provider taint of a single node.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">node_controller_initial_node_sync_delay_seconds</div>
	<div class="metric_help">Number of seconds after node creation when NodeController finished the initial synchronization of a single node.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">node_ipam_controller_cidrset_allocation_tries_per_request</div>
	<div class="metric_help">Number of endpoints added on each Service sync</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">clusterCIDR</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">node_ipam_controller_cidrset_cidrs_allocations_total</div>
	<div class="metric_help">Counter measuring total number of CIDR allocations.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">clusterCIDR</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">node_ipam_controller_cidrset_cidrs_releases_total</div>
	<div class="metric_help">Counter measuring total number of CIDR releases.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">clusterCIDR</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">node_ipam_controller_cidrset_usage_cidrs</div>
	<div class="metric_help">Gauge measuring percentage of allocated CIDRs.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">clusterCIDR</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">node_ipam_controller_cirdset_max_cidrs</div>
	<div class="metric_help">Maximum number of CIDRs that can be allocated.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">clusterCIDR</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">node_swap_usage_bytes</div>
	<div class="metric_help">Current swap usage of the node in bytes. Reported only on non-windows systems</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">number_of_l4_ilbs</div>
	<div class="metric_help">Number of L4 ILBs</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">feature</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">plugin_manager_total_plugins</div>
	<div class="metric_help">Number of plugins in Plugin Manager</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">socket_path</span><span class="metric_label">state</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">pod_gc_collector_force_delete_pod_errors_total</div>
	<div class="metric_help">Number of errors encountered when forcefully deleting the pods since the Pod GC Controller started.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">namespace</span><span class="metric_label">reason</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">pod_gc_collector_force_delete_pods_total</div>
	<div class="metric_help">Number of pods that are being forcefully deleted since the Pod GC Controller started.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">namespace</span><span class="metric_label">reason</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">pod_security_errors_total</div>
	<div class="metric_help">Number of errors preventing normal evaluation. Non-fatal errors may result in the latest restricted profile being used for evaluation.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">fatal</span><span class="metric_label">request_operation</span><span class="metric_label">resource</span><span class="metric_label">subresource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">pod_security_evaluations_total</div>
	<div class="metric_help">Number of policy evaluations that occurred, not counting ignored or exempt requests.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">decision</span><span class="metric_label">mode</span><span class="metric_label">policy_level</span><span class="metric_label">policy_version</span><span class="metric_label">request_operation</span><span class="metric_label">resource</span><span class="metric_label">subresource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">pod_security_exemptions_total</div>
	<div class="metric_help">Number of exempt requests, not counting ignored or out of scope requests.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">request_operation</span><span class="metric_label">resource</span><span class="metric_label">subresource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">pod_swap_usage_bytes</div>
	<div class="metric_help">Current amount of the pod swap usage in bytes. Reported only on non-windows systems</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">pod</span><span class="metric_label">namespace</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">prober_probe_duration_seconds</div>
	<div class="metric_help">Duration in seconds for a probe response.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">container</span><span class="metric_label">namespace</span><span class="metric_label">pod</span><span class="metric_label">probe_type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">prober_probe_total</div>
	<div class="metric_help">Cumulative number of a liveness, readiness or startup probe for a container by result.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">container</span><span class="metric_label">namespace</span><span class="metric_label">pod</span><span class="metric_label">pod_uid</span><span class="metric_label">probe_type</span><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">pv_collector_bound_pv_count</div>
	<div class="metric_help">Gauge measuring number of persistent volume currently bound</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">storage_class</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">pv_collector_bound_pvc_count</div>
	<div class="metric_help">Gauge measuring number of persistent volume claim currently bound</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">namespace</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">pv_collector_total_pv_count</div>
	<div class="metric_help">Gauge measuring total number of persistent volumes</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">plugin_name</span><span class="metric_label">volume_mode</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">pv_collector_unbound_pv_count</div>
	<div class="metric_help">Gauge measuring number of persistent volume currently unbound</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">storage_class</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">pv_collector_unbound_pvc_count</div>
	<div class="metric_help">Gauge measuring number of persistent volume claim currently unbound</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">namespace</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">reconstruct_volume_operations_errors_total</div>
	<div class="metric_help">The number of volumes that failed reconstruction from the operating system during kubelet startup.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">reconstruct_volume_operations_total</div>
	<div class="metric_help">The number of volumes that were attempted to be reconstructed from the operating system during kubelet startup. This includes both successful and failed reconstruction.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">replicaset_controller_sorting_deletion_age_ratio</div>
	<div class="metric_help">The ratio of chosen deleted pod's ages to the current youngest pod's age (at the time). Should be <2. The intent of this metric is to measure the rough efficacy of the LogarithmicScaleDown feature gate's effect on the sorting (and deletion) of pods when a replicaset scales down. This only considers Ready pods when calculating and reporting.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">resourceclaim_controller_create_attempts_total</div>
	<div class="metric_help">Number of ResourceClaims creation requests</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">resourceclaim_controller_create_failures_total</div>
	<div class="metric_help">Number of ResourceClaims creation request failures</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">rest_client_dns_resolution_duration_seconds</div>
	<div class="metric_help">DNS resolver latency in seconds. Broken down by host.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">host</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">rest_client_exec_plugin_call_total</div>
	<div class="metric_help">Number of calls to an exec plugin, partitioned by the type of event encountered (no_error, plugin_execution_error, plugin_not_found_error, client_internal_error) and an optional exit code. The exit code will be set to 0 if and only if the plugin call was successful.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">call_status</span><span class="metric_label">code</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">rest_client_exec_plugin_certificate_rotation_age</div>
	<div class="metric_help">Histogram of the number of seconds the last auth exec plugin client certificate lived before being rotated. If auth exec plugin client certificates are unused, histogram will contain no data.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">rest_client_exec_plugin_ttl_seconds</div>
	<div class="metric_help">Gauge of the shortest TTL (time-to-live) of the client certificate(s) managed by the auth exec plugin. The value is in seconds until certificate expiry (negative if already expired). If auth exec plugins are unused or manage no TLS certificates, the value will be +INF.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">rest_client_rate_limiter_duration_seconds</div>
	<div class="metric_help">Client side rate limiter latency in seconds. Broken down by verb, and host.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">host</span><span class="metric_label">verb</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">rest_client_request_duration_seconds</div>
	<div class="metric_help">Request latency in seconds. Broken down by verb, and host.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">host</span><span class="metric_label">verb</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">rest_client_request_retries_total</div>
	<div class="metric_help">Number of request retries, partitioned by status code, verb, and host.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span><span class="metric_label">host</span><span class="metric_label">verb</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">rest_client_request_size_bytes</div>
	<div class="metric_help">Request size in bytes. Broken down by verb and host.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">host</span><span class="metric_label">verb</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">rest_client_requests_total</div>
	<div class="metric_help">Number of HTTP requests, partitioned by status code, method, and host.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span><span class="metric_label">host</span><span class="metric_label">method</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">rest_client_response_size_bytes</div>
	<div class="metric_help">Response size in bytes. Broken down by verb and host.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">host</span><span class="metric_label">verb</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">rest_client_transport_cache_entries</div>
	<div class="metric_help">Number of transport entries in the internal cache.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">rest_client_transport_create_calls_total</div>
	<div class="metric_help">Number of calls to get a new transport, partitioned by the result of the operation hit: obtained from the cache, miss: created and added to the cache, uncacheable: created and not cached</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">retroactive_storageclass_errors_total</div>
	<div class="metric_help">Total number of failed retroactive StorageClass assignments to persistent volume claim</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">retroactive_storageclass_total</div>
	<div class="metric_help">Total number of retroactive StorageClass assignments to persistent volume claim</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">root_ca_cert_publisher_sync_duration_seconds</div>
	<div class="metric_help">Number of namespace syncs happened in root ca cert publisher.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">root_ca_cert_publisher_sync_total</div>
	<div class="metric_help">Number of namespace syncs happened in root ca cert publisher.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">code</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">running_managed_controllers</div>
	<div class="metric_help">Indicates where instances of a controller are currently running</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">manager</span><span class="metric_label">name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">scheduler_goroutines</div>
	<div class="metric_help">Number of running goroutines split by the work they do such as binding.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">scheduler_permit_wait_duration_seconds</div>
	<div class="metric_help">Duration of waiting on permit.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">result</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">scheduler_plugin_evaluation_total</div>
	<div class="metric_help">Number of attempts to schedule pods by each plugin and the extension point (available only in PreFilter, Filter, PreScore, and Score).</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">extension_point</span><span class="metric_label">plugin</span><span class="metric_label">profile</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">scheduler_plugin_execution_duration_seconds</div>
	<div class="metric_help">Duration for running a plugin at a specific extension point.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">extension_point</span><span class="metric_label">plugin</span><span class="metric_label">status</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">scheduler_scheduler_cache_size</div>
	<div class="metric_help">Number of nodes, pods, and assumed (bound) pods in the scheduler cache.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">type</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">scheduler_scheduling_algorithm_duration_seconds</div>
	<div class="metric_help">Scheduling algorithm latency in seconds</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">scheduler_unschedulable_pods</div>
	<div class="metric_help">The number of unschedulable pods broken down by plugin name. A pod will increment the gauge for all plugins that caused it to not schedule and so this metric have meaning only when broken down by plugin.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">plugin</span><span class="metric_label">profile</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">scheduler_volume_binder_cache_requests_total</div>
	<div class="metric_help">Total number for request volume binding cache</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">scheduler_volume_scheduling_stage_error_total</div>
	<div class="metric_help">Volume scheduling stage error count</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">scrape_error</div>
	<div class="metric_help">1 if there was an error while getting container metrics, 0 otherwise</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_deprecated_version"><label class="metric_detail">Deprecated Versions:</label><span>1.29.0</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">service_controller_loadbalancer_sync_total</div>
	<div class="metric_help">A metric counting the amount of times any load balancer has been configured, as an effect of service/node changes on the cluster</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">service_controller_nodesync_error_total</div>
	<div class="metric_help">A metric counting the amount of times any load balancer has been configured and errored, as an effect of node changes on the cluster</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">service_controller_nodesync_latency_seconds</div>
	<div class="metric_help">A metric measuring the latency for nodesync which updates loadbalancer hosts on cluster node updates.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">service_controller_update_loadbalancer_host_latency_seconds</div>
	<div class="metric_help">A metric measuring the latency for updating each load balancer hosts.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">serviceaccount_invalid_legacy_auto_token_uses_total</div>
	<div class="metric_help">Cumulative invalid auto-generated legacy tokens used</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">serviceaccount_legacy_auto_token_uses_total</div>
	<div class="metric_help">Cumulative auto-generated legacy tokens used</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">serviceaccount_legacy_manual_token_uses_total</div>
	<div class="metric_help">Cumulative manually created legacy tokens used</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">serviceaccount_legacy_tokens_total</div>
	<div class="metric_help">Cumulative legacy service account tokens used</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">serviceaccount_stale_tokens_total</div>
	<div class="metric_help">Cumulative stale projected service account tokens used</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">serviceaccount_valid_tokens_total</div>
	<div class="metric_help">Cumulative valid projected service account tokens used</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">storage_count_attachable_volumes_in_use</div>
	<div class="metric_help">Measure number of volumes in use</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">node</span><span class="metric_label">volume_plugin</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">storage_operation_duration_seconds</div>
	<div class="metric_help">Storage operation duration</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">migrated</span><span class="metric_label">operation_name</span><span class="metric_label">status</span><span class="metric_label">volume_plugin</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">taint_eviction_controller_pod_deletion_duration_seconds</div>
	<div class="metric_help">Latency, in seconds, between the time when a taint effect has been activated for the Pod and its deletion via TaintEvictionController.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">taint_eviction_controller_pod_deletions_total</div>
	<div class="metric_help">Total number of Pods deleted by TaintEvictionController since its start.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">ttl_after_finished_controller_job_deletion_duration_seconds</div>
	<div class="metric_help">The time it took to delete the job since it became eligible for deletion</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	</ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">volume_manager_selinux_container_errors_total</div>
	<div class="metric_help">Number of errors when kubelet cannot compute SELinux context for a container. Kubelet can't start such a Pod then and it will retry, therefore value of this metric may not represent the actual nr. of containers.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">access_mode</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">volume_manager_selinux_container_warnings_total</div>
	<div class="metric_help">Number of errors when kubelet cannot compute SELinux context for a container that are ignored. They will become real errors when SELinuxMountReadWriteOncePod feature is expanded to all volume access modes.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">access_mode</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">volume_manager_selinux_pod_context_mismatch_errors_total</div>
	<div class="metric_help">Number of errors when a Pod defines different SELinux contexts for its containers that use the same volume. Kubelet can't start such a Pod then and it will retry, therefore value of this metric may not represent the actual nr. of Pods.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">access_mode</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">volume_manager_selinux_pod_context_mismatch_warnings_total</div>
	<div class="metric_help">Number of errors when a Pod defines different SELinux contexts for its containers that use the same volume. They are not errors yet, but they will become real errors when SELinuxMountReadWriteOncePod feature is expanded to all volume access modes.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">access_mode</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">volume_manager_selinux_volume_context_mismatch_errors_total</div>
	<div class="metric_help">Number of errors when a Pod uses a volume that is already mounted with a different SELinux context than the Pod needs. Kubelet can't start such a Pod then and it will retry, therefore value of this metric may not represent the actual nr. of Pods.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">access_mode</span><span class="metric_label">volume_plugin</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">volume_manager_selinux_volume_context_mismatch_warnings_total</div>
	<div class="metric_help">Number of errors when a Pod uses a volume that is already mounted with a different SELinux context than the Pod needs. They are not errors yet, but they will become real errors when SELinuxMountReadWriteOncePod feature is expanded to all volume access modes.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">access_mode</span><span class="metric_label">volume_plugin</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">volume_manager_selinux_volumes_admitted_total</div>
	<div class="metric_help">Number of volumes whose SELinux context was fine and will be mounted with mount -o context option.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">access_mode</span><span class="metric_label">volume_plugin</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">volume_manager_total_volumes</div>
	<div class="metric_help">Number of volumes in Volume Manager</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="custom"><label class="metric_detail">Type:</label> <span class="metric_type">Custom</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">plugin_name</span><span class="metric_label">state</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">volume_operation_total_errors</div>
	<div class="metric_help">Total volume operation errors</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation_name</span><span class="metric_label">plugin_name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">volume_operation_total_seconds</div>
	<div class="metric_help">Storage operation end to end duration in seconds</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">operation_name</span><span class="metric_label">plugin_name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">watch_cache_capacity</div>
	<div class="metric_help">Total capacity of watch cache broken by resource type.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">watch_cache_capacity_decrease_total</div>
	<div class="metric_help">Total number of watch cache capacity decrease events broken by resource type.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">watch_cache_capacity_increase_total</div>
	<div class="metric_help">Total number of watch cache capacity increase events broken by resource type.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">resource</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">workqueue_adds_total</div>
	<div class="metric_help">Total number of adds handled by workqueue</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">workqueue_depth</div>
	<div class="metric_help">Current depth of workqueue</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">workqueue_longest_running_processor_seconds</div>
	<div class="metric_help">How many seconds has the longest running processor for workqueue been running.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">workqueue_queue_duration_seconds</div>
	<div class="metric_help">How long in seconds an item stays in workqueue before being requested.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">workqueue_retries_total</div>
	<div class="metric_help">Total number of retries handled by workqueue</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="counter"><label class="metric_detail">Type:</label> <span class="metric_type">Counter</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">workqueue_unfinished_work_seconds</div>
	<div class="metric_help">How many seconds of work has done that is in progress and hasn't been observed by work_duration. Large values indicate stuck threads. One can deduce the number of stuck threads by observing the rate at which this increases.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="gauge"><label class="metric_detail">Type:</label> <span class="metric_type">Gauge</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span></li></ul>
	</div><div class="metric" data-stability="alpha">
	<div class="metric_name">workqueue_work_duration_seconds</div>
	<div class="metric_help">How long in seconds processing an item from workqueue takes.</div>
	<ul>
	<li><label class="metric_detail">Stability Level:</label><span class="metric_stability_level">ALPHA</span></li>
	<li data-type="histogram"><label class="metric_detail">Type:</label> <span class="metric_type">Histogram</span></li>
	<li class="metric_labels_varying"><label class="metric_detail">Labels:</label><span class="metric_label">name</span></li></ul>
	</div>
</div>
