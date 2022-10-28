---
title: Kubernetes Metrics Reference
content_type: reference
description: >-
  Details of the metric data that Kubernetes components export.
---


## Metrics (auto-generated 2022 Oct 27)

This page details the metrics that different Kubernetes components export. You can query the metrics endpoint for these 
components using an HTTP scrape, and fetch the metrics in Prometheus format.

### List of Stable Kubernetes Metrics

<table class="table" caption="This is the list of STABLE metrics emitted from core Kubernetes components">
<thead>
	<tr>
		<th class="metric_name">Name</th>
		<th class="metric_stability_level">Stability Level</th>
		<th class="metric_type">Type</th>
		<th class="metric_help">Help</th>
		<th class="metric_labels">Labels</th>
		<th class="metric_const_labels">Const Labels</th>
	</tr>
</thead>
<tbody>

<tr class="metric"><td class="metric_name">apiserver_admission_controller_admission_duration_seconds</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Admission controller latency histogram in seconds, identified by name and broken out for each operation and API resource and type (validate or admit).</td>
<td class="metric_labels_varying"><div class="metric_label">name</div><div class="metric_label">operation</div><div class="metric_label">rejected</div><div class="metric_label">type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_admission_step_admission_duration_seconds</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Admission sub-step latency histogram in seconds, broken out for each operation and API resource and step type (validate or admit).</td>
<td class="metric_labels_varying"><div class="metric_label">operation</div><div class="metric_label">rejected</div><div class="metric_label">type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_admission_webhook_admission_duration_seconds</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Admission webhook latency histogram in seconds, identified by name and broken out for each operation and API resource and type (validate or admit).</td>
<td class="metric_labels_varying"><div class="metric_label">name</div><div class="metric_label">operation</div><div class="metric_label">rejected</div><div class="metric_label">type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_current_inflight_requests</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Maximal number of currently used inflight request limit of this apiserver per request kind in last second.</td>
<td class="metric_labels_varying"><div class="metric_label">request_kind</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_longrunning_requests</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Gauge of all active long-running apiserver requests broken out by verb, group, version, resource, scope and component. Not all requests are tracked this way.</td>
<td class="metric_labels_varying"><div class="metric_label">component</div><div class="metric_label">group</div><div class="metric_label">resource</div><div class="metric_label">scope</div><div class="metric_label">subresource</div><div class="metric_label">verb</div><div class="metric_label">version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_request_duration_seconds</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Response latency distribution in seconds for each verb, dry run value, group, version, resource, subresource, scope and component.</td>
<td class="metric_labels_varying"><div class="metric_label">component</div><div class="metric_label">dry_run</div><div class="metric_label">group</div><div class="metric_label">resource</div><div class="metric_label">scope</div><div class="metric_label">subresource</div><div class="metric_label">verb</div><div class="metric_label">version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_request_total</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of apiserver requests broken out for each verb, dry run value, group, version, resource, scope, component, and HTTP response code.</td>
<td class="metric_labels_varying"><div class="metric_label">code</div><div class="metric_label">component</div><div class="metric_label">dry_run</div><div class="metric_label">group</div><div class="metric_label">resource</div><div class="metric_label">scope</div><div class="metric_label">subresource</div><div class="metric_label">verb</div><div class="metric_label">version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_requested_deprecated_apis</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Gauge of deprecated APIs that have been requested, broken out by API group, version, resource, subresource, and removed_release.</td>
<td class="metric_labels_varying"><div class="metric_label">group</div><div class="metric_label">removed_release</div><div class="metric_label">resource</div><div class="metric_label">subresource</div><div class="metric_label">version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_response_sizes</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Response size distribution in bytes for each group, version, verb, resource, subresource, scope and component.</td>
<td class="metric_labels_varying"><div class="metric_label">component</div><div class="metric_label">group</div><div class="metric_label">resource</div><div class="metric_label">scope</div><div class="metric_label">subresource</div><div class="metric_label">verb</div><div class="metric_label">version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_storage_objects</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of stored objects at the time of last check split by kind.</td>
<td class="metric_labels_varying"><div class="metric_label">resource</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">node_collector_evictions_total</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of Node evictions that happened since current instance of NodeController started.</td>
<td class="metric_labels_varying"><div class="metric_label">zone</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_framework_extension_point_duration_seconds</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Latency for running all plugins of a specific extension point.</td>
<td class="metric_labels_varying"><div class="metric_label">extension_point</div><div class="metric_label">profile</div><div class="metric_label">status</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_pending_pods</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulablePods.</td>
<td class="metric_labels_varying"><div class="metric_label">queue</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_pod_scheduling_attempts</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Number of attempts to successfully schedule a pod.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_pod_scheduling_duration_seconds</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">E2e latency for a pod being scheduled which may include multiple scheduling attempts.</td>
<td class="metric_labels_varying"><div class="metric_label">attempts</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_preemption_attempts_total</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Total preemption attempts in the cluster till now</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_preemption_victims</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Number of selected preemption victims</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_queue_incoming_pods_total</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of pods added to scheduling queues by event and queue type.</td>
<td class="metric_labels_varying"><div class="metric_label">event</div><div class="metric_label">queue</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_schedule_attempts_total</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of attempts to schedule pods, by the result. 'unschedulable' means a pod could not be scheduled, while 'error' means an internal scheduler problem.</td>
<td class="metric_labels_varying"><div class="metric_label">profile</div><div class="metric_label">result</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_scheduling_attempt_duration_seconds</td>
<td class="metric_stability_level" data-stability="stable">STABLE</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Scheduling attempt latency in seconds (scheduling algorithm + binding)</td>
<td class="metric_labels_varying"><div class="metric_label">profile</div><div class="metric_label">result</div></td>
<td class="metric_labels_constant">None</td></tr>
</tbody>
</table>

### List of Alpha Kubernetes Metrics

<table class="table" caption="This is the list of ALPHA metrics emitted from core Kubernetes components">
<thead>
	<tr>
		<th class="metric_name">Name</th>
		<th class="metric_stability_level">Stability Level</th>
		<th class="metric_type">Type</th>
		<th class="metric_help">Help</th>
		<th class="metric_labels">Labels</th>
		<th class="metric_const_labels">Const Labels</th>
	</tr>
</thead>
<tbody>

<tr class="metric"><td class="metric_name">aggregator_openapi_v2_regeneration_count</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of OpenAPI v2 spec regeneration count broken down by causing APIService name and reason.</td>
<td class="metric_labels_varying"><div class="metric_label">apiservice</div><div class="metric_label">reason</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">aggregator_openapi_v2_regeneration_duration</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Gauge of OpenAPI v2 spec regeneration duration in seconds.</td>
<td class="metric_labels_varying"><div class="metric_label">reason</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">aggregator_unavailable_apiservice_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of APIServices which are marked as unavailable broken down by APIService name and reason.</td>
<td class="metric_labels_varying"><div class="metric_label">name</div><div class="metric_label">reason</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiextensions_openapi_v2_regeneration_count</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of OpenAPI v2 spec regeneration count broken down by causing CRD name and reason.</td>
<td class="metric_labels_varying"><div class="metric_label">crd</div><div class="metric_label">reason</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiextensions_openapi_v3_regeneration_count</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of OpenAPI v3 spec regeneration count broken down by group, version, causing CRD and reason.</td>
<td class="metric_labels_varying"><div class="metric_label">crd</div><div class="metric_label">group</div><div class="metric_label">reason</div><div class="metric_label">version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_admission_step_admission_duration_seconds_summary</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="summary">Summary</td>
<td class="metric_description">Admission sub-step latency summary in seconds, broken out for each operation and API resource and step type (validate or admit).</td>
<td class="metric_labels_varying"><div class="metric_label">operation</div><div class="metric_label">rejected</div><div class="metric_label">type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_admission_webhook_fail_open_count</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Admission webhook fail open count, identified by name and broken out for each admission type (validating or mutating).</td>
<td class="metric_labels_varying"><div class="metric_label">name</div><div class="metric_label">type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_admission_webhook_rejection_count</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Admission webhook rejection count, identified by name and broken out for each admission type (validating or admit) and operation. Additional labels specify an error type (calling_webhook_error or apiserver_internal_error if an error occurred; no_error otherwise) and optionally a non-zero rejection code if the webhook rejects the request with an HTTP status code (honored by the apiserver when the code is greater or equal to 400). Codes greater than 600 are truncated to 600, to keep the metrics cardinality bounded.</td>
<td class="metric_labels_varying"><div class="metric_label">error_type</div><div class="metric_label">name</div><div class="metric_label">operation</div><div class="metric_label">rejection_code</div><div class="metric_label">type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_admission_webhook_request_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Admission webhook request total, identified by name and broken out for each admission type (validating or mutating) and operation. Additional labels specify whether the request was rejected or not and an HTTP status code. Codes greater than 600 are truncated to 600, to keep the metrics cardinality bounded.</td>
<td class="metric_labels_varying"><div class="metric_label">code</div><div class="metric_label">name</div><div class="metric_label">operation</div><div class="metric_label">rejected</div><div class="metric_label">type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_audit_error_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of audit events that failed to be audited properly. Plugin identifies the plugin affected by the error.</td>
<td class="metric_labels_varying"><div class="metric_label">plugin</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_audit_event_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of audit events generated and sent to the audit backend.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_audit_level_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of policy levels for audit events (1 per request).</td>
<td class="metric_labels_varying"><div class="metric_label">level</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_audit_requests_rejected_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of apiserver requests rejected due to an error in audit logging backend.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_cache_list_fetched_objects_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of objects read from watch cache in the course of serving a LIST request</td>
<td class="metric_labels_varying"><div class="metric_label">index</div><div class="metric_label">resource_prefix</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_cache_list_returned_objects_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of objects returned for a LIST request from watch cache</td>
<td class="metric_labels_varying"><div class="metric_label">resource_prefix</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_cache_list_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of LIST requests served from watch cache</td>
<td class="metric_labels_varying"><div class="metric_label">index</div><div class="metric_label">resource_prefix</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_cel_compilation_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description"></td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_cel_evaluation_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description"></td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_certificates_registry_csr_honored_duration_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Total number of issued CSRs with a requested duration that was honored, sliced by signer (only kubernetes.io signer names are specifically identified)</td>
<td class="metric_labels_varying"><div class="metric_label">signerName</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_certificates_registry_csr_requested_duration_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Total number of issued CSRs with a requested duration, sliced by signer (only kubernetes.io signer names are specifically identified)</td>
<td class="metric_labels_varying"><div class="metric_label">signerName</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_client_certificate_expiration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Distribution of the remaining lifetime on the certificate used to authenticate a request.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_crd_webhook_conversion_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">CRD webhook conversion duration in seconds</td>
<td class="metric_labels_varying"><div class="metric_label">crd_name</div><div class="metric_label">from_version</div><div class="metric_label">succeeded</div><div class="metric_label">to_version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_current_inqueue_requests</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Maximal number of queued requests in this apiserver per request kind in last second.</td>
<td class="metric_labels_varying"><div class="metric_label">request_kind</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_delegated_authn_request_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Request latency in seconds. Broken down by status code.</td>
<td class="metric_labels_varying"><div class="metric_label">code</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_delegated_authn_request_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of HTTP requests partitioned by status code.</td>
<td class="metric_labels_varying"><div class="metric_label">code</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_delegated_authz_request_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Request latency in seconds. Broken down by status code.</td>
<td class="metric_labels_varying"><div class="metric_label">code</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_delegated_authz_request_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of HTTP requests partitioned by status code.</td>
<td class="metric_labels_varying"><div class="metric_label">code</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_egress_dialer_dial_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Dial latency histogram in seconds, labeled by the protocol (http-connect or grpc), transport (tcp or uds)</td>
<td class="metric_labels_varying"><div class="metric_label">protocol</div><div class="metric_label">transport</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_egress_dialer_dial_failure_count</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Dial failure count, labeled by the protocol (http-connect or grpc), transport (tcp or uds), and stage (connect or proxy). The stage indicates at which stage the dial failed</td>
<td class="metric_labels_varying"><div class="metric_label">protocol</div><div class="metric_label">stage</div><div class="metric_label">transport</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_envelope_encryption_dek_cache_fill_percent</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Percent of the cache slots currently occupied by cached DEKs.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_envelope_encryption_dek_cache_inter_arrival_time_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Time (in seconds) of inter arrival of transformation requests.</td>
<td class="metric_labels_varying"><div class="metric_label">transformation_type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_current_executing_requests</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of requests in initial (for a WATCH) or any (for a non-WATCH) execution stage in the API Priority and Fairness subsystem</td>
<td class="metric_labels_varying"><div class="metric_label">flow_schema</div><div class="metric_label">priority_level</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_current_inqueue_requests</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of requests currently pending in queues of the API Priority and Fairness subsystem</td>
<td class="metric_labels_varying"><div class="metric_label">flow_schema</div><div class="metric_label">priority_level</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_current_r</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">R(time of last change)</td>
<td class="metric_labels_varying"><div class="metric_label">priority_level</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_dispatch_r</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">R(time of last dispatch)</td>
<td class="metric_labels_varying"><div class="metric_label">priority_level</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_dispatched_requests_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of requests executed by API Priority and Fairness subsystem</td>
<td class="metric_labels_varying"><div class="metric_label">flow_schema</div><div class="metric_label">priority_level</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_epoch_advance_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of times the queueset's progress meter jumped backward</td>
<td class="metric_labels_varying"><div class="metric_label">priority_level</div><div class="metric_label">success</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_latest_s</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">S(most recently dispatched request)</td>
<td class="metric_labels_varying"><div class="metric_label">priority_level</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_next_discounted_s_bounds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">min and max, over queues, of S(oldest waiting request in queue) - estimated work in progress</td>
<td class="metric_labels_varying"><div class="metric_label">bound</div><div class="metric_label">priority_level</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_next_s_bounds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">min and max, over queues, of S(oldest waiting request in queue)</td>
<td class="metric_labels_varying"><div class="metric_label">bound</div><div class="metric_label">priority_level</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_priority_level_request_utilization</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type=""></td>
<td class="metric_description">Observations, at the end of every nanosecond, of number of requests (as a fraction of the relevant limit) waiting or in any stage of execution (but only initial stage for WATCHes)</td>
<td class="metric_labels_varying"><div class="metric_label">phase</div><div class="metric_label">priority_level</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_priority_level_seat_utilization</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type=""></td>
<td class="metric_description">Observations, at the end of every nanosecond, of utilization of seats for any stage of execution (but only initial stage for WATCHes)</td>
<td class="metric_labels_varying"><div class="metric_label">priority_level</div></td>
<td class="metric_labels_constant">map[phase:executing]</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_read_vs_write_current_requests</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type=""></td>
<td class="metric_description">Observations, at the end of every nanosecond, of the number of requests (as a fraction of the relevant limit) waiting or in regular stage of execution</td>
<td class="metric_labels_varying"><div class="metric_label">phase</div><div class="metric_label">request_kind</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_rejected_requests_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of requests rejected by API Priority and Fairness subsystem</td>
<td class="metric_labels_varying"><div class="metric_label">flow_schema</div><div class="metric_label">priority_level</div><div class="metric_label">reason</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_request_concurrency_in_use</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Concurrency (number of seats) occupied by the currently executing (initial stage for a WATCH, any stage otherwise) requests in the API Priority and Fairness subsystem</td>
<td class="metric_labels_varying"><div class="metric_label">flow_schema</div><div class="metric_label">priority_level</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_request_concurrency_limit</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Shared concurrency limit in the API Priority and Fairness subsystem</td>
<td class="metric_labels_varying"><div class="metric_label">priority_level</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_request_dispatch_no_accommodation_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of times a dispatch attempt resulted in a non accommodation due to lack of available seats</td>
<td class="metric_labels_varying"><div class="metric_label">flow_schema</div><div class="metric_label">priority_level</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_request_execution_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Duration of initial stage (for a WATCH) or any (for a non-WATCH) stage of request execution in the API Priority and Fairness subsystem</td>
<td class="metric_labels_varying"><div class="metric_label">flow_schema</div><div class="metric_label">priority_level</div><div class="metric_label">type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_request_queue_length_after_enqueue</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Length of queue in the API Priority and Fairness subsystem, as seen by each request after it is enqueued</td>
<td class="metric_labels_varying"><div class="metric_label">flow_schema</div><div class="metric_label">priority_level</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_request_wait_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Length of time a request spent waiting in its queue</td>
<td class="metric_labels_varying"><div class="metric_label">execute</div><div class="metric_label">flow_schema</div><div class="metric_label">priority_level</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_watch_count_samples</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">count of watchers for mutating requests in API Priority and Fairness</td>
<td class="metric_labels_varying"><div class="metric_label">flow_schema</div><div class="metric_label">priority_level</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_flowcontrol_work_estimated_seats</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Number of estimated seats (maximum of initial and final seats) associated with requests in API Priority and Fairness</td>
<td class="metric_labels_varying"><div class="metric_label">flow_schema</div><div class="metric_label">priority_level</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_init_events_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of init events processed in watch cache broken by resource type.</td>
<td class="metric_labels_varying"><div class="metric_label">resource</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_kube_aggregator_x509_insecure_sha1_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counts the number of requests to servers with insecure SHA1 signatures in their serving certificate OR the number of connection failures due to the insecure SHA1 signatures (either/or, based on the runtime environment)</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_kube_aggregator_x509_missing_san_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counts the number of requests to servers missing SAN extension in their serving certificate OR the number of connection failures due to the lack of x509 certificate SAN extension missing (either/or, based on the runtime environment)</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_request_aborts_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of requests which apiserver aborted possibly due to a timeout, for each group, version, verb, resource, subresource and scope</td>
<td class="metric_labels_varying"><div class="metric_label">group</div><div class="metric_label">resource</div><div class="metric_label">scope</div><div class="metric_label">subresource</div><div class="metric_label">verb</div><div class="metric_label">version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_request_body_sizes</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Apiserver request body sizes broken out by size.</td>
<td class="metric_labels_varying"><div class="metric_label">resource</div><div class="metric_label">verb</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_request_filter_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Request filter latency distribution in seconds, for each filter type</td>
<td class="metric_labels_varying"><div class="metric_label">filter</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_request_post_timeout_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Tracks the activity of the request handlers after the associated requests have been timed out by the apiserver</td>
<td class="metric_labels_varying"><div class="metric_label">source</div><div class="metric_label">status</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_request_slo_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Response latency distribution (not counting webhook duration) in seconds for each verb, group, version, resource, subresource, scope and component.</td>
<td class="metric_labels_varying"><div class="metric_label">component</div><div class="metric_label">group</div><div class="metric_label">resource</div><div class="metric_label">scope</div><div class="metric_label">subresource</div><div class="metric_label">verb</div><div class="metric_label">version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_request_terminations_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of requests which apiserver terminated in self-defense.</td>
<td class="metric_labels_varying"><div class="metric_label">code</div><div class="metric_label">component</div><div class="metric_label">group</div><div class="metric_label">resource</div><div class="metric_label">scope</div><div class="metric_label">subresource</div><div class="metric_label">verb</div><div class="metric_label">version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_request_timestamp_comparison_time</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Time taken for comparison of old vs new objects in UPDATE or PATCH requests</td>
<td class="metric_labels_varying"><div class="metric_label">code_path</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_selfrequest_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of apiserver self-requests broken out for each verb, API resource and subresource.</td>
<td class="metric_labels_varying"><div class="metric_label">resource</div><div class="metric_label">subresource</div><div class="metric_label">verb</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_storage_data_key_generation_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Latencies in seconds of data encryption key(DEK) generation operations.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_storage_data_key_generation_failures_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Total number of failed data encryption key(DEK) generation operations.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_storage_envelope_transformation_cache_misses_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Total number of cache misses while accessing key decryption key(KEK).</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_storage_list_evaluated_objects_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of objects tested in the course of serving a LIST request from storage</td>
<td class="metric_labels_varying"><div class="metric_label">resource</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_storage_list_fetched_objects_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of objects read from storage in the course of serving a LIST request</td>
<td class="metric_labels_varying"><div class="metric_label">resource</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_storage_list_returned_objects_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of objects returned for a LIST request from storage</td>
<td class="metric_labels_varying"><div class="metric_label">resource</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_storage_list_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of LIST requests served from storage</td>
<td class="metric_labels_varying"><div class="metric_label">resource</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_storage_transformation_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Latencies in seconds of value transformation operations.</td>
<td class="metric_labels_varying"><div class="metric_label">transformation_type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_storage_transformation_operations_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Total number of transformations.</td>
<td class="metric_labels_varying"><div class="metric_label">status</div><div class="metric_label">transformation_type</div><div class="metric_label">transformer_prefix</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_terminated_watchers_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of watchers closed due to unresponsiveness broken by resource type.</td>
<td class="metric_labels_varying"><div class="metric_label">resource</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_tls_handshake_errors_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of requests dropped with 'TLS handshake error from' error</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_watch_cache_events_dispatched_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of events dispatched in watch cache broken by resource type.</td>
<td class="metric_labels_varying"><div class="metric_label">resource</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_watch_cache_initializations_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of watch cache initializations broken by resource type.</td>
<td class="metric_labels_varying"><div class="metric_label">resource</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_watch_events_sizes</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Watch event size distribution in bytes</td>
<td class="metric_labels_varying"><div class="metric_label">group</div><div class="metric_label">kind</div><div class="metric_label">version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_watch_events_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of events sent in watch clients</td>
<td class="metric_labels_varying"><div class="metric_label">group</div><div class="metric_label">kind</div><div class="metric_label">version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_webhooks_x509_insecure_sha1_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counts the number of requests to servers with insecure SHA1 signatures in their serving certificate OR the number of connection failures due to the insecure SHA1 signatures (either/or, based on the runtime environment)</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">apiserver_webhooks_x509_missing_san_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counts the number of requests to servers missing SAN extension in their serving certificate OR the number of connection failures due to the lack of x509 certificate SAN extension missing (either/or, based on the runtime environment)</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">attachdetach_controller_forced_detaches</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of times the A/D Controller performed a forced detach</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">authenticated_user_requests</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of authenticated requests broken out by username.</td>
<td class="metric_labels_varying"><div class="metric_label">username</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">authentication_attempts</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of authenticated attempts.</td>
<td class="metric_labels_varying"><div class="metric_label">result</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">authentication_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Authentication duration in seconds broken out by result.</td>
<td class="metric_labels_varying"><div class="metric_label">result</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">authentication_token_cache_active_fetch_count</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description"></td>
<td class="metric_labels_varying"><div class="metric_label">status</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">authentication_token_cache_fetch_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description"></td>
<td class="metric_labels_varying"><div class="metric_label">status</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">authentication_token_cache_request_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description"></td>
<td class="metric_labels_varying"><div class="metric_label">status</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">authentication_token_cache_request_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description"></td>
<td class="metric_labels_varying"><div class="metric_label">status</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">cloudprovider_aws_api_request_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Latency of AWS API calls</td>
<td class="metric_labels_varying"><div class="metric_label">request</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">cloudprovider_aws_api_request_errors</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">AWS API errors</td>
<td class="metric_labels_varying"><div class="metric_label">request</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">cloudprovider_aws_api_throttled_requests_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">AWS API throttled requests</td>
<td class="metric_labels_varying"><div class="metric_label">operation_name</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">cloudprovider_azure_api_request_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Latency of an Azure API call</td>
<td class="metric_labels_varying"><div class="metric_label">request</div><div class="metric_label">resource_group</div><div class="metric_label">source</div><div class="metric_label">subscription_id</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">cloudprovider_azure_api_request_errors</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of errors for an Azure API call</td>
<td class="metric_labels_varying"><div class="metric_label">request</div><div class="metric_label">resource_group</div><div class="metric_label">source</div><div class="metric_label">subscription_id</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">cloudprovider_azure_api_request_ratelimited_count</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of rate limited Azure API calls</td>
<td class="metric_labels_varying"><div class="metric_label">request</div><div class="metric_label">resource_group</div><div class="metric_label">source</div><div class="metric_label">subscription_id</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">cloudprovider_azure_api_request_throttled_count</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of throttled Azure API calls</td>
<td class="metric_labels_varying"><div class="metric_label">request</div><div class="metric_label">resource_group</div><div class="metric_label">source</div><div class="metric_label">subscription_id</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">cloudprovider_azure_op_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Latency of an Azure service operation</td>
<td class="metric_labels_varying"><div class="metric_label">request</div><div class="metric_label">resource_group</div><div class="metric_label">source</div><div class="metric_label">subscription_id</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">cloudprovider_azure_op_failure_count</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of failed Azure service operations</td>
<td class="metric_labels_varying"><div class="metric_label">request</div><div class="metric_label">resource_group</div><div class="metric_label">source</div><div class="metric_label">subscription_id</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">cloudprovider_gce_api_request_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Latency of a GCE API call</td>
<td class="metric_labels_varying"><div class="metric_label">region</div><div class="metric_label">request</div><div class="metric_label">version</div><div class="metric_label">zone</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">cloudprovider_gce_api_request_errors</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of errors for an API call</td>
<td class="metric_labels_varying"><div class="metric_label">region</div><div class="metric_label">request</div><div class="metric_label">version</div><div class="metric_label">zone</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">cloudprovider_vsphere_api_request_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Latency of vsphere api call</td>
<td class="metric_labels_varying"><div class="metric_label">request</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">cloudprovider_vsphere_api_request_errors</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">vsphere Api errors</td>
<td class="metric_labels_varying"><div class="metric_label">request</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">cloudprovider_vsphere_operation_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Latency of vsphere operation call</td>
<td class="metric_labels_varying"><div class="metric_label">operation</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">cloudprovider_vsphere_operation_errors</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">vsphere operation errors</td>
<td class="metric_labels_varying"><div class="metric_label">operation</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">cronjob_controller_cronjob_job_creation_skew_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Time between when a cronjob is scheduled to be run, and when the corresponding job is created</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">csi_operations_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Container Storage Interface operation duration with gRPC error code status total</td>
<td class="metric_labels_varying"><div class="metric_label">driver_name</div><div class="metric_label">grpc_status_code</div><div class="metric_label">method_name</div><div class="metric_label">migrated</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">endpoint_slice_controller_changes</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of EndpointSlice changes</td>
<td class="metric_labels_varying"><div class="metric_label">operation</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">endpoint_slice_controller_desired_endpoint_slices</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of EndpointSlices that would exist with perfect endpoint allocation</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">endpoint_slice_controller_endpoints_added_per_sync</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Number of endpoints added on each Service sync</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">endpoint_slice_controller_endpoints_desired</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of endpoints desired</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">endpoint_slice_controller_endpoints_removed_per_sync</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Number of endpoints removed on each Service sync</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">endpoint_slice_controller_endpointslices_changed_per_sync</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Number of EndpointSlices changed on each Service sync</td>
<td class="metric_labels_varying"><div class="metric_label">topology</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">endpoint_slice_controller_num_endpoint_slices</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of EndpointSlices</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">endpoint_slice_controller_syncs</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of EndpointSlice syncs</td>
<td class="metric_labels_varying"><div class="metric_label">result</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">endpoint_slice_mirroring_controller_addresses_skipped_per_sync</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Number of addresses skipped on each Endpoints sync due to being invalid or exceeding MaxEndpointsPerSubset</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">endpoint_slice_mirroring_controller_changes</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of EndpointSlice changes</td>
<td class="metric_labels_varying"><div class="metric_label">operation</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">endpoint_slice_mirroring_controller_desired_endpoint_slices</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of EndpointSlices that would exist with perfect endpoint allocation</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">endpoint_slice_mirroring_controller_endpoints_added_per_sync</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Number of endpoints added on each Endpoints sync</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">endpoint_slice_mirroring_controller_endpoints_desired</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of endpoints desired</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">endpoint_slice_mirroring_controller_endpoints_removed_per_sync</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Number of endpoints removed on each Endpoints sync</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">endpoint_slice_mirroring_controller_endpoints_sync_duration</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Duration of syncEndpoints() in seconds</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">endpoint_slice_mirroring_controller_endpoints_updated_per_sync</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Number of endpoints updated on each Endpoints sync</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">endpoint_slice_mirroring_controller_num_endpoint_slices</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of EndpointSlices</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">ephemeral_volume_controller_create_failures_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of PersistenVolumeClaims creation requests</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">ephemeral_volume_controller_create_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of PersistenVolumeClaims creation requests</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">etcd_bookmark_counts</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of etcd bookmarks (progress notify events) split by kind.</td>
<td class="metric_labels_varying"><div class="metric_label">resource</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">etcd_db_total_size_in_bytes</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Total size of the etcd database file physically allocated in bytes.</td>
<td class="metric_labels_varying"><div class="metric_label">endpoint</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">etcd_lease_object_counts</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Number of objects attached to a single etcd lease.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">etcd_request_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Etcd request latency in seconds for each operation and object type.</td>
<td class="metric_labels_varying"><div class="metric_label">operation</div><div class="metric_label">type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">etcd_version_info</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Etcd server's binary version</td>
<td class="metric_labels_varying"><div class="metric_label">binary_version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">field_validation_request_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Response latency distribution in seconds for each field validation value and whether field validation is enabled or not</td>
<td class="metric_labels_varying"><div class="metric_label">enabled</div><div class="metric_label">field_validation</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">garbagecollector_controller_resources_sync_error_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of garbage collector resources sync errors</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">get_token_count</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of total Token() requests to the alternate token source</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">get_token_fail_count</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of failed Token() requests to the alternate token source</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">job_controller_job_finished_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">The number of finished job</td>
<td class="metric_labels_varying"><div class="metric_label">completion_mode</div><div class="metric_label">result</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">job_controller_job_pods_finished_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">The number of finished Pods that are fully tracked</td>
<td class="metric_labels_varying"><div class="metric_label">completion_mode</div><div class="metric_label">result</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">job_controller_job_sync_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">The time it took to sync a job</td>
<td class="metric_labels_varying"><div class="metric_label">action</div><div class="metric_label">completion_mode</div><div class="metric_label">result</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">job_controller_job_sync_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">The number of job syncs</td>
<td class="metric_labels_varying"><div class="metric_label">action</div><div class="metric_label">completion_mode</div><div class="metric_label">result</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">job_controller_terminated_pods_tracking_finalizer_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">`The number of terminated pods (phase=Failed|Succeeded), that have the finalizer batch.kubernetes.io/job-tracking, The event label can be "add" or "delete".`</td>
<td class="metric_labels_varying"><div class="metric_label">event</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kube_apiserver_clusterip_allocator_allocated_ips</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Gauge measuring the number of allocated IPs for Services</td>
<td class="metric_labels_varying"><div class="metric_label">cidr</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kube_apiserver_clusterip_allocator_allocation_errors_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of errors trying to allocate Cluster IPs</td>
<td class="metric_labels_varying"><div class="metric_label">cidr</div><div class="metric_label">scope</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kube_apiserver_clusterip_allocator_allocation_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of Cluster IPs allocations</td>
<td class="metric_labels_varying"><div class="metric_label">cidr</div><div class="metric_label">scope</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kube_apiserver_clusterip_allocator_available_ips</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Gauge measuring the number of available IPs for Services</td>
<td class="metric_labels_varying"><div class="metric_label">cidr</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kube_apiserver_pod_logs_pods_logs_backend_tls_failure_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Total number of requests for pods/logs that failed due to kubelet server TLS verification</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kube_apiserver_pod_logs_pods_logs_insecure_backend_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Total number of requests for pods/logs sliced by usage type: enforce_tls, skip_tls_allowed, skip_tls_denied</td>
<td class="metric_labels_varying"><div class="metric_label">usage</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_certificate_manager_client_expiration_renew_errors</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of certificate renewal errors.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_certificate_manager_client_ttl_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Gauge of the TTL (time-to-live) of the Kubelet's client certificate. The value is in seconds until certificate expiry (negative if already expired). If client certificate is invalid or unused, the value will be +INF.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_certificate_manager_server_rotation_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Histogram of the number of seconds the previous certificate lived before being rotated.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_certificate_manager_server_ttl_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Gauge of the shortest TTL (time-to-live) of the Kubelet's serving certificate. The value is in seconds until certificate expiry (negative if already expired). If serving certificate is invalid or unused, the value will be +INF.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_cgroup_manager_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Duration in seconds for cgroup manager operations. Broken down by method.</td>
<td class="metric_labels_varying"><div class="metric_label">operation_type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_containers_per_pod_count</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">The number of containers per pod.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_device_plugin_alloc_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Duration in seconds to serve a device plugin Allocation request. Broken down by resource name.</td>
<td class="metric_labels_varying"><div class="metric_label">resource_name</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_device_plugin_registration_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative number of device plugin registrations. Broken down by resource name.</td>
<td class="metric_labels_varying"><div class="metric_label">resource_name</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_eviction_stats_age_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Time between when stats are collected, and when pod is evicted based on those stats by eviction signal</td>
<td class="metric_labels_varying"><div class="metric_label">eviction_signal</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_evictions</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative number of pod evictions by eviction signal</td>
<td class="metric_labels_varying"><div class="metric_label">eviction_signal</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_graceful_shutdown_end_time_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Last graceful shutdown start time since unix epoch in seconds</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_graceful_shutdown_start_time_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Last graceful shutdown start time since unix epoch in seconds</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_http_inflight_requests</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of the inflight http requests</td>
<td class="metric_labels_varying"><div class="metric_label">long_running</div><div class="metric_label">method</div><div class="metric_label">path</div><div class="metric_label">server_type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_http_requests_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Duration in seconds to serve http requests</td>
<td class="metric_labels_varying"><div class="metric_label">long_running</div><div class="metric_label">method</div><div class="metric_label">path</div><div class="metric_label">server_type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_http_requests_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of the http requests received since the server started</td>
<td class="metric_labels_varying"><div class="metric_label">long_running</div><div class="metric_label">method</div><div class="metric_label">path</div><div class="metric_label">server_type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_kubelet_credential_provider_plugin_duration</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Duration of execution in seconds for credential provider plugin</td>
<td class="metric_labels_varying"><div class="metric_label">plugin_name</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_kubelet_credential_provider_plugin_errors</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of errors from credential provider plugin</td>
<td class="metric_labels_varying"><div class="metric_label">plugin_name</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_lifecycle_handler_http_fallbacks_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">The number of times lifecycle handlers successfully fell back to http from https.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_managed_ephemeral_containers</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Current number of ephemeral containers in pods managed by this kubelet. Ephemeral containers will be ignored if disabled by the EphemeralContainers feature gate, and this number will be 0.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_node_name</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">The node's name. The count is always 1.</td>
<td class="metric_labels_varying"><div class="metric_label">node</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_pleg_discard_events</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">The number of discard events in PLEG.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_pleg_last_seen_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Timestamp in seconds when PLEG was last seen active.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_pleg_relist_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Duration in seconds for relisting pods in PLEG.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_pleg_relist_interval_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Interval in seconds between relisting in PLEG.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_pod_resources_endpoint_errors_get_allocatable</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of requests to the PodResource GetAllocatableResources endpoint which returned error. Broken down by server api version.</td>
<td class="metric_labels_varying"><div class="metric_label">server_api_version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_pod_resources_endpoint_errors_list</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of requests to the PodResource List endpoint which returned error. Broken down by server api version.</td>
<td class="metric_labels_varying"><div class="metric_label">server_api_version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_pod_resources_endpoint_requests_get_allocatable</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of requests to the PodResource GetAllocatableResources endpoint. Broken down by server api version.</td>
<td class="metric_labels_varying"><div class="metric_label">server_api_version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_pod_resources_endpoint_requests_list</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of requests to the PodResource List endpoint. Broken down by server api version.</td>
<td class="metric_labels_varying"><div class="metric_label">server_api_version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_pod_resources_endpoint_requests_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative number of requests to the PodResource endpoint. Broken down by server api version.</td>
<td class="metric_labels_varying"><div class="metric_label">server_api_version</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_pod_start_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Duration in seconds from kubelet seeing a pod for the first time to the pod starting to run</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_pod_status_sync_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Duration in seconds to sync a pod status update. Measures time from detection of a change to pod status until the API is successfully updated for that pod, even if multiple intevening changes to pod status occur.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_pod_worker_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Duration in seconds to sync a single pod. Broken down by operation type: create, update, or sync</td>
<td class="metric_labels_varying"><div class="metric_label">operation_type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_pod_worker_start_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Duration in seconds from kubelet seeing a pod to starting a worker.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_preemptions</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative number of pod preemptions by preemption resource</td>
<td class="metric_labels_varying"><div class="metric_label">preemption_signal</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_run_podsandbox_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Duration in seconds of the run_podsandbox operations. Broken down by RuntimeClass.Handler.</td>
<td class="metric_labels_varying"><div class="metric_label">runtime_handler</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_run_podsandbox_errors_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative number of the run_podsandbox operation errors by RuntimeClass.Handler.</td>
<td class="metric_labels_varying"><div class="metric_label">runtime_handler</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_running_containers</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of containers currently running</td>
<td class="metric_labels_varying"><div class="metric_label">container_state</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_running_pods</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of pods that have a running pod sandbox</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_runtime_operations_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Duration in seconds of runtime operations. Broken down by operation type.</td>
<td class="metric_labels_varying"><div class="metric_label">operation_type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_runtime_operations_errors_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative number of runtime operation errors by operation type.</td>
<td class="metric_labels_varying"><div class="metric_label">operation_type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_runtime_operations_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative number of runtime operations by operation type.</td>
<td class="metric_labels_varying"><div class="metric_label">operation_type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_server_expiration_renew_errors</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter of certificate renewal errors.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_started_containers_errors_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative number of errors when starting containers</td>
<td class="metric_labels_varying"><div class="metric_label">code</div><div class="metric_label">container_type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_started_containers_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative number of containers started</td>
<td class="metric_labels_varying"><div class="metric_label">container_type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_started_host_process_containers_errors_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative number of errors when starting hostprocess containers. This metric will only be collected on Windows and requires WindowsHostProcessContainers feature gate to be enabled.</td>
<td class="metric_labels_varying"><div class="metric_label">code</div><div class="metric_label">container_type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_started_host_process_containers_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative number of hostprocess containers started. This metric will only be collected on Windows and requires WindowsHostProcessContainers feature gate to be enabled.</td>
<td class="metric_labels_varying"><div class="metric_label">container_type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_started_pods_errors_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative number of errors when starting pods</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_started_pods_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative number of pods started</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubelet_volume_metric_collection_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Duration in seconds to calculate volume stats</td>
<td class="metric_labels_varying"><div class="metric_label">metric_source</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubeproxy_network_programming_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">In Cluster Network Programming Latency in seconds</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubeproxy_sync_proxy_rules_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">SyncProxyRules latency in seconds</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubeproxy_sync_proxy_rules_endpoint_changes_pending</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Pending proxy rules Endpoint changes</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubeproxy_sync_proxy_rules_endpoint_changes_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative proxy rules Endpoint changes</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubeproxy_sync_proxy_rules_iptables_restore_failures_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative proxy iptables restore failures</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubeproxy_sync_proxy_rules_iptables_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of proxy iptables rules programmed</td>
<td class="metric_labels_varying"><div class="metric_label">table</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubeproxy_sync_proxy_rules_last_queued_timestamp_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">The last time a sync of proxy rules was queued</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubeproxy_sync_proxy_rules_last_timestamp_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">The last time proxy rules were successfully synced</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubeproxy_sync_proxy_rules_no_local_endpoints_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of services with a Local traffic policy and no endpoints</td>
<td class="metric_labels_varying"><div class="metric_label">traffic_policy</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubeproxy_sync_proxy_rules_service_changes_pending</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Pending proxy rules Service changes</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubeproxy_sync_proxy_rules_service_changes_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative proxy rules Service changes</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubernetes_build_info</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">A metric with a constant '1' value labeled by major, minor, git version, git commit, git tree state, build date, Go version, and compiler from which Kubernetes was built, and platform on which it is running.</td>
<td class="metric_labels_varying"><div class="metric_label">build_date</div><div class="metric_label">compiler</div><div class="metric_label">git_commit</div><div class="metric_label">git_tree_state</div><div class="metric_label">git_version</div><div class="metric_label">go_version</div><div class="metric_label">major</div><div class="metric_label">minor</div><div class="metric_label">platform</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubernetes_feature_enabled</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">This metric records the data about the stage and enablement of a k8s feature.</td>
<td class="metric_labels_varying"><div class="metric_label">name</div><div class="metric_label">stage</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubernetes_healthcheck</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">This metric records the result of a single healthcheck.</td>
<td class="metric_labels_varying"><div class="metric_label">name</div><div class="metric_label">type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">kubernetes_healthchecks_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">This metric records the results of all healthcheck.</td>
<td class="metric_labels_varying"><div class="metric_label">name</div><div class="metric_label">status</div><div class="metric_label">type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">leader_election_master_status</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Gauge of if the reporting system is master of the relevant lease, 0 indicates backup, 1 indicates master. 'name' is the string used to identify the lease. Please make sure to group by name.</td>
<td class="metric_labels_varying"><div class="metric_label">name</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">node_authorizer_graph_actions_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Histogram of duration of graph actions in node authorizer.</td>
<td class="metric_labels_varying"><div class="metric_label">operation</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">node_collector_evictions_number</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of Node evictions that happened since current instance of NodeController started, This metric is replaced by node_collector_evictions_total.</td>
<td class="metric_labels_varying"><div class="metric_label">zone</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">node_collector_unhealthy_nodes_in_zone</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Gauge measuring number of not Ready Nodes per zones.</td>
<td class="metric_labels_varying"><div class="metric_label">zone</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">node_collector_zone_health</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Gauge measuring percentage of healthy nodes per zone.</td>
<td class="metric_labels_varying"><div class="metric_label">zone</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">node_collector_zone_size</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Gauge measuring number of registered Nodes per zones.</td>
<td class="metric_labels_varying"><div class="metric_label">zone</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">node_ipam_controller_cidrset_allocation_tries_per_request</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Number of endpoints added on each Service sync</td>
<td class="metric_labels_varying"><div class="metric_label">clusterCIDR</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">node_ipam_controller_cidrset_cidrs_allocations_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter measuring total number of CIDR allocations.</td>
<td class="metric_labels_varying"><div class="metric_label">clusterCIDR</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">node_ipam_controller_cidrset_cidrs_releases_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter measuring total number of CIDR releases.</td>
<td class="metric_labels_varying"><div class="metric_label">clusterCIDR</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">node_ipam_controller_cidrset_usage_cidrs</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Gauge measuring percentage of allocated CIDRs.</td>
<td class="metric_labels_varying"><div class="metric_label">clusterCIDR</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">node_ipam_controller_multicidrset_allocation_tries_per_request</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Histogram measuring CIDR allocation tries per request.</td>
<td class="metric_labels_varying"><div class="metric_label">clusterCIDR</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">node_ipam_controller_multicidrset_cidrs_allocations_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter measuring total number of CIDR allocations.</td>
<td class="metric_labels_varying"><div class="metric_label">clusterCIDR</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">node_ipam_controller_multicidrset_cidrs_releases_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Counter measuring total number of CIDR releases.</td>
<td class="metric_labels_varying"><div class="metric_label">clusterCIDR</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">node_ipam_controller_multicidrset_usage_cidrs</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Gauge measuring percentage of allocated CIDRs.</td>
<td class="metric_labels_varying"><div class="metric_label">clusterCIDR</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">number_of_l4_ilbs</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of L4 ILBs</td>
<td class="metric_labels_varying"><div class="metric_label">feature</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">pod_security_errors_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of errors preventing normal evaluation. Non-fatal errors may result in the latest restricted profile being used for evaluation.</td>
<td class="metric_labels_varying"><div class="metric_label">fatal</div><div class="metric_label">request_operation</div><div class="metric_label">resource</div><div class="metric_label">subresource</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">pod_security_evaluations_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of policy evaluations that occurred, not counting ignored or exempt requests.</td>
<td class="metric_labels_varying"><div class="metric_label">decision</div><div class="metric_label">mode</div><div class="metric_label">policy_level</div><div class="metric_label">policy_version</div><div class="metric_label">request_operation</div><div class="metric_label">resource</div><div class="metric_label">subresource</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">pod_security_exemptions_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of exempt requests, not counting ignored or out of scope requests.</td>
<td class="metric_labels_varying"><div class="metric_label">request_operation</div><div class="metric_label">resource</div><div class="metric_label">subresource</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">prober_probe_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Duration in seconds for a probe response.</td>
<td class="metric_labels_varying"><div class="metric_label">container</div><div class="metric_label">namespace</div><div class="metric_label">pod</div><div class="metric_label">probe_type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">prober_probe_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative number of a liveness, readiness or startup probe for a container by result.</td>
<td class="metric_labels_varying"><div class="metric_label">container</div><div class="metric_label">namespace</div><div class="metric_label">pod</div><div class="metric_label">pod_uid</div><div class="metric_label">probe_type</div><div class="metric_label">result</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">replicaset_controller_sorting_deletion_age_ratio</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">The ratio of chosen deleted pod's ages to the current youngest pod's age (at the time). Should be <2.The intent of this metric is to measure the rough efficacy of the LogarithmicScaleDown feature gate's effect onthe sorting (and deletion) of pods when a replicaset scales down. This only considers Ready pods when calculating and reporting.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">rest_client_exec_plugin_call_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of calls to an exec plugin, partitioned by the type of event encountered (no_error, plugin_execution_error, plugin_not_found_error, client_internal_error) and an optional exit code. The exit code will be set to 0 if and only if the plugin call was successful.</td>
<td class="metric_labels_varying"><div class="metric_label">call_status</div><div class="metric_label">code</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">rest_client_exec_plugin_certificate_rotation_age</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Histogram of the number of seconds the last auth exec plugin client certificate lived before being rotated. If auth exec plugin client certificates are unused, histogram will contain no data.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">rest_client_exec_plugin_ttl_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Gauge of the shortest TTL (time-to-live) of the client certificate(s) managed by the auth exec plugin. The value is in seconds until certificate expiry (negative if already expired). If auth exec plugins are unused or manage no TLS certificates, the value will be +INF.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">rest_client_rate_limiter_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Client side rate limiter latency in seconds. Broken down by verb, and host.</td>
<td class="metric_labels_varying"><div class="metric_label">host</div><div class="metric_label">verb</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">rest_client_request_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Request latency in seconds. Broken down by verb, and host.</td>
<td class="metric_labels_varying"><div class="metric_label">host</div><div class="metric_label">verb</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">rest_client_request_size_bytes</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Request size in bytes. Broken down by verb and host.</td>
<td class="metric_labels_varying"><div class="metric_label">host</div><div class="metric_label">verb</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">rest_client_requests_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of HTTP requests, partitioned by status code, method, and host.</td>
<td class="metric_labels_varying"><div class="metric_label">code</div><div class="metric_label">host</div><div class="metric_label">method</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">rest_client_response_size_bytes</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Response size in bytes. Broken down by verb and host.</td>
<td class="metric_labels_varying"><div class="metric_label">host</div><div class="metric_label">verb</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">root_ca_cert_publisher_sync_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Number of namespace syncs happened in root ca cert publisher.</td>
<td class="metric_labels_varying"><div class="metric_label">code</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">root_ca_cert_publisher_sync_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Number of namespace syncs happened in root ca cert publisher.</td>
<td class="metric_labels_varying"><div class="metric_label">code</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">running_managed_controllers</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Indicates where instances of a controller are currently running</td>
<td class="metric_labels_varying"><div class="metric_label">manager</div><div class="metric_label">name</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_e2e_scheduling_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">E2e scheduling latency in seconds (scheduling algorithm + binding). This metric is replaced by scheduling_attempt_duration_seconds.</td>
<td class="metric_labels_varying"><div class="metric_label">profile</div><div class="metric_label">result</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_goroutines</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of running goroutines split by the work they do such as binding.</td>
<td class="metric_labels_varying"><div class="metric_label">operation</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_permit_wait_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Duration of waiting on permit.</td>
<td class="metric_labels_varying"><div class="metric_label">result</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_plugin_execution_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Duration for running a plugin at a specific extension point.</td>
<td class="metric_labels_varying"><div class="metric_label">extension_point</div><div class="metric_label">plugin</div><div class="metric_label">status</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_scheduler_cache_size</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of nodes, pods, and assumed (bound) pods in the scheduler cache.</td>
<td class="metric_labels_varying"><div class="metric_label">type</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_scheduler_goroutines</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of running goroutines split by the work they do such as binding. This metric is replaced by the \"goroutines\" metric.</td>
<td class="metric_labels_varying"><div class="metric_label">work</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_scheduling_algorithm_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Scheduling algorithm latency in seconds</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_unschedulable_pods</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">The number of unschedulable pods broken down by plugin name. A pod will increment the gauge for all plugins that caused it to not schedule and so this metric have meaning only when broken down by plugin.</td>
<td class="metric_labels_varying"><div class="metric_label">plugin</div><div class="metric_label">profile</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_volume_binder_cache_requests_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Total number for request volume binding cache</td>
<td class="metric_labels_varying"><div class="metric_label">operation</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">scheduler_volume_scheduling_stage_error_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Volume scheduling stage error count</td>
<td class="metric_labels_varying"><div class="metric_label">operation</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">service_controller_nodesync_latency_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">A metric measuring the latency for nodesync which updates loadbalancer hosts on cluster node updates.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">service_controller_update_loadbalancer_host_latency_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">A metric measuring the latency for updating each load balancer hosts.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">serviceaccount_legacy_tokens_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative legacy service account tokens used</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">serviceaccount_stale_tokens_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative stale projected service account tokens used</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">serviceaccount_valid_tokens_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Cumulative valid projected service account tokens used</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">storage_operation_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Storage operation duration</td>
<td class="metric_labels_varying"><div class="metric_label">migrated</div><div class="metric_label">operation_name</div><div class="metric_label">status</div><div class="metric_label">volume_plugin</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">ttl_after_finished_controller_job_deletion_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">The time it took to delete the job since it became eligible for deletion</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">volume_manager_selinux_container_errors_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of errors when kubelet cannot compute SELinux context for a container. Kubelet can't start such a Pod then and it will retry, therefore value of this metric may not represent the actual nr. of containers.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">volume_manager_selinux_container_warnings_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of errors when kubelet cannot compute SELinux context for a container that are ignored. They will become real errors when SELinuxMountReadWriteOncePod feature is expanded to all volume access modes.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">volume_manager_selinux_pod_context_mismatch_errors_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of errors when a Pod defines different SELinux contexts for its containers that use the same volume. Kubelet can't start such a Pod then and it will retry, therefore value of this metric may not represent the actual nr. of Pods.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">volume_manager_selinux_pod_context_mismatch_warnings_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of errors when a Pod defines different SELinux contexts for its containers that use the same volume. They are not errors yet, but they will become real errors when SELinuxMountReadWriteOncePod feature is expanded to all volume access modes.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">volume_manager_selinux_volume_context_mismatch_errors_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of errors when a Pod uses a volume that is already mounted with a different SELinux context than the Pod needs. Kubelet can't start such a Pod then and it will retry, therefore value of this metric may not represent the actual nr. of Pods.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">volume_manager_selinux_volume_context_mismatch_warnings_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of errors when a Pod uses a volume that is already mounted with a different SELinux context than the Pod needs. They are not errors yet, but they will become real errors when SELinuxMountReadWriteOncePod feature is expanded to all volume access modes.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">volume_manager_selinux_volumes_admitted_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Number of volumes whose SELinux context was fine and will be mounted with mount -o context option.</td>
<td class="metric_labels_varying">None</td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">volume_operation_total_errors</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Total volume operation errors</td>
<td class="metric_labels_varying"><div class="metric_label">operation_name</div><div class="metric_label">plugin_name</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">volume_operation_total_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">Storage operation end to end duration in seconds</td>
<td class="metric_labels_varying"><div class="metric_label">operation_name</div><div class="metric_label">plugin_name</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">watch_cache_capacity</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Total capacity of watch cache broken by resource type.</td>
<td class="metric_labels_varying"><div class="metric_label">resource</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">watch_cache_capacity_decrease_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Total number of watch cache capacity decrease events broken by resource type.</td>
<td class="metric_labels_varying"><div class="metric_label">resource</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">watch_cache_capacity_increase_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Total number of watch cache capacity increase events broken by resource type.</td>
<td class="metric_labels_varying"><div class="metric_label">resource</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">workqueue_adds_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Total number of adds handled by workqueue</td>
<td class="metric_labels_varying"><div class="metric_label">name</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">workqueue_depth</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">Current depth of workqueue</td>
<td class="metric_labels_varying"><div class="metric_label">name</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">workqueue_longest_running_processor_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">How many seconds has the longest running processor for workqueue been running.</td>
<td class="metric_labels_varying"><div class="metric_label">name</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">workqueue_queue_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">How long in seconds an item stays in workqueue before being requested.</td>
<td class="metric_labels_varying"><div class="metric_label">name</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">workqueue_retries_total</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="counter">Counter</td>
<td class="metric_description">Total number of retries handled by workqueue</td>
<td class="metric_labels_varying"><div class="metric_label">name</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">workqueue_unfinished_work_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="gauge">Gauge</td>
<td class="metric_description">How many seconds of work has done that is in progress and hasn't been observed by work_duration. Large values indicate stuck threads. One can deduce the number of stuck threads by observing the rate at which this increases.</td>
<td class="metric_labels_varying"><div class="metric_label">name</div></td>
<td class="metric_labels_constant">None</td></tr>
<tr class="metric"><td class="metric_name">workqueue_work_duration_seconds</td>
<td class="metric_stability_level" data-stability="alpha">ALPHA</td>
<td class="metric_type" data-type="histogram">Histogram</td>
<td class="metric_description">How long in seconds processing an item from workqueue takes.</td>
<td class="metric_labels_varying"><div class="metric_label">name</div></td>
<td class="metric_labels_constant">None</td></tr>
</tbody>
</table>
