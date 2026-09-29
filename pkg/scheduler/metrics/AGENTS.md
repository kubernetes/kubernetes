# Scheduler Metrics & Resource Collectors: Architectural Overview & Contributor Guide

This guide provides AI agents and human contributors with an in-depth architectural breakdown, metric inventory, sampling mechanics, concurrency model, and monitoring best practices for the Kubernetes Scheduler metrics subsystem (`pkg/scheduler/metrics`) and resource collectors (`pkg/scheduler/metrics/resources`).

---

## 1. High-Level Overview & Subsystem Role

The `pkg/scheduler/metrics` package is responsible for instrumentation, latency tracking, queue accounting, and performance observability across `kube-scheduler`. It exposes Prometheus-compatible metrics through `k8s.io/component-base/metrics` and provides dedicated collectors for workload resource allocations.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              kube-scheduler                                  │
│                                                                              │
│  ┌──────────────────────┐   ┌────────────────────────┐   ┌─────────────────┐ │
│  │ SchedulingQueue      │   │ Framework & Plugins    │   │ Cache & Backend │ │
│  │ (PriorityQueue)      │   │ (Filter, Score, Bind)  │   │ (NodeInfo/Pods) │ │
│  └──────────┬───────────┘   └───────────┬────────────┘   └────────┬────────┘ │
│             │                           │                         │          │
│             ▼                           ▼                         ▼          │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                       pkg/scheduler/metrics                            │  │
│  │                                                                        │  │
│  │  ┌─────────────────────────┐       ┌────────────────────────────────┐  │  │
│  │  │ Synchronous Direct      │       │ MetricAsyncRecorder            │  │  │
│  │  │ (Counters, Gauges, SLI) │       │ (Non-blocking Channel Buffers) │  │  │
│  │  └───────────┬─────────────┘       └───────────────┬────────────────┘  │  │
│  │              │                                     │                   │  │
│  │              ▼                                     ▼                   │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │                   k8s.io/component-base/metrics                  │  │  │
│  │  │                   (legacyregistry.DefaultGatherer)               │  │  │
│  │  └─────────────────────────────────┬────────────────────────────────┘  │  │
│  └────────────────────────────────────┼───────────────────────────────────┘  │
│                                       │                                      │
│  ┌────────────────────────────────────┴───────────────────────────────────┐  │
│  │ pkg/scheduler/metrics/resources (Independent Collector & Registry)     │  │
│  │  - Scrapes PodLister -> kube_pod_resource_request / limit              │  │
│  └────────────────────────────────────┬───────────────────────────────────┘  │
└───────────────────────────────────────┼──────────────────────────────────────┘
                                        │
                                        ▼ Scrape Endpoints
                   ┌─────────────────────────────────────────┐
                   │  /metrics           (Scheduler Metrics) │
                   │  /metrics/resources (Resource Collector)│
                   └─────────────────────────────────────────┘
```

### Core Responsibilities:
1. **End-to-End SLI Measurement**: Tracks single-attempt and multi-attempt pod scheduling latencies (`pod_scheduling_sli_duration_seconds`, `scheduling_attempt_duration_seconds`).
2. **Asynchronous Non-Blocking Recording**: Offloads high-frequency extension point and plugin latency observations via `MetricAsyncRecorder` to prevent lock contention in scheduling critical paths.
3. **Queue Accounting & Entity Depth**: Tracks pod counts and multi-entity group counts (`pending_pods`, `queued_entities`) across `activeQ`, `backoffQ`, `unschedulableEntities`, and `gated` states.
4. **Plugin Performance Diagnostics**: Measures per-plugin execution durations, queueing hint evaluation costs, and failure/rejection reasons.
5. **Cluster Resource Visibility**: Exposes exact container and pod resource requests and limits through a dedicated, isolated collector (`pkg/scheduler/metrics/resources`).

---

## 2. Directory & Package Structure

```
pkg/scheduler/metrics/
├── metrics.go               # Metric definitions, constants, buckets, and registration logic
├── metric_recorder.go      # QueuedEntitiesRecorder and MetricAsyncRecorder (buffered channels)
├── profile_metrics.go      # Scheduler profile-scoped helper functions (Scheduled, Unschedulable, Placements)
├── metric_recorder_test.go # Unit tests for async flushing, backpressure, and entity accounting
├── profile_metrics_test.go # Unit tests for profile-specific metric observations
└── resources/
    ├── resources.go        # Custom StableCollector for kube_pod_resource_request / limit
    └── resources_test.go   # Scenarios testing lifecycle phases, units, zero values, and lister queries
```

---

## 3. Metrics Inventory & Stability Reference

Scheduler metrics are classified into stability tiers: **STABLE** (guaranteed backward compatibility), **BETA** (enabled by default), and **ALPHA** (experimental or gated by feature gates).

### 3.1 Scheduling Latency & Lifecycle Metrics

| Metric Name | Type | Stability | Labels | Feature Gate | Description |
|---|---|---|---|---|---|
| `scheduler_schedule_attempts_total` | CounterVec | `STABLE` | `result`, `profile` | None | Total number of pod scheduling attempts. Results: `scheduled`, `unschedulable`, `error`. |
| `scheduler_scheduling_attempt_duration_seconds` | HistogramVec | `STABLE` | `result`, `profile` | None | End-to-end latency of a single scheduling attempt (scheduling algorithm + binding cycle). |
| `scheduler_scheduling_algorithm_duration_seconds` | Histogram | `BETA` | None | None | Latency of the pure scheduling algorithm (PreFilter -> Filter -> PreScore -> Score -> Reserve -> Permit), excluding async binding. |
| `scheduler_pod_scheduling_sli_duration_seconds` | HistogramVec | `BETA` | `attempts` | None | E2E latency from initial queue admission to successful scheduling across all retries. Buckets: 10ms to ~88min. |
| `scheduler_pod_scheduling_attempts` | Histogram | `STABLE` | None | None | Distribution of attempts required to schedule a pod successfully. Buckets: 1, 2, 4, 8, 16. |
| `scheduler_pod_scheduled_after_flush_total` | Counter | `ALPHA` | None | None | Number of pods successfully scheduled after being flushed from unschedulableEntities by timeout (detects QueueingHint bugs). |
| `scheduler_permit_wait_duration_seconds` | HistogramVec | `BETA` | `result` | None | Duration pods spent waiting on Permit plugin delays. |
| `scheduler_event_handling_duration_seconds` | HistogramVec | `ALPHA` | `event` | None | Latency of handling informer events in scheduler queue event handlers. |

### 3.2 Queue Depth & Entity Gauges

| Metric Name | Type | Stability | Labels | Feature Gate | Description |
|---|---|---|---|---|---|
| `scheduler_pending_pods` | GaugeVec | `STABLE` | `queue` | None | Number of pending pods by queue: `active`, `backoff`, `unschedulable`, `gated`, `incomplete`, `pending`. |
| `scheduler_queued_entities` | GaugeVec | `ALPHA` | `queue`, `type` | None | Number of queued scheduling entities (`pod`, `podgroup`, `compositepodgroup`) by queue. |
| `scheduler_queue_incoming_pods_total` | CounterVec | `STABLE` | `queue`, `event` | None | Number of pods added to scheduling queues by event and queue type. |
| `scheduler_queue_incoming_entities_total` | CounterVec | `ALPHA` | `queue`, `event`, `type` | None | Number of scheduling entities added to queues by event, queue, and entity type. |
| `scheduler_inflight_events` | GaugeVec | `ALPHA` | `event` | None | Number of cluster mutation events tracked while pods are actively in-flight. |
| `scheduler_unschedulable_pods` | GaugeVec | `BETA` | `plugin`, `profile` | None | Unschedulable pod count broken down by rejecting plugin and profile. |

### 3.3 Framework & Plugin Execution Metrics

| Metric Name | Type | Stability | Labels | Feature Gate | Description |
|---|---|---|---|---|---|
| `scheduler_framework_extension_point_duration_seconds` | HistogramVec | `STABLE` | `extension_point`, `status`, `profile` | None | Latency of running all plugins at a specific extension point. |
| `scheduler_plugin_execution_duration_seconds` | HistogramVec | `BETA` | `plugin`, `extension_point`, `status` | None | Latency of executing an individual plugin at an extension point (factor 1.5 fine buckets). |
| `scheduler_plugin_evaluation_total` | CounterVec | `BETA` | `plugin`, `extension_point`, `profile` | None | Count of plugin evaluations across `PreFilter`, `Filter`, `PreScore`, `Score`. |
| `scheduler_queueing_hint_execution_duration_seconds` | HistogramVec | `ALPHA` | `plugin`, `event`, `hint` | None | Latency of running a plugin's `QueueingHint` callback function. |
| `scheduler_pre_queueing_hint_evaluations_total` | CounterVec | `ALPHA` | `plugin`, `result` | None | Pre-queueing hint evaluations (`all_pods` vs `narrowed`). |

### 3.4 Preemption, Workloads & Feature-Gated Metrics

| Metric Name | Type | Stability | Labels | Feature Gate | Description |
|---|---|---|---|---|---|
| `scheduler_preemption_attempts_total` | Counter | `STABLE` | None | None | Total single-pod preemption attempts. |
| `scheduler_preemption_victims` | Histogram | `STABLE` | None | None | Distribution of victim pod count per preemption (1 to 64+). |
| `scheduler_workload_preemption_attempts_total` | CounterVec | `ALPHA` | `result` | `GenericWorkload` | Preemption attempts initiated by workloads/pod groups. |
| `scheduler_workload_preemption_victims` | Histogram | `ALPHA` | None | `GenericWorkload` | Number of victim pods in workload preemptions (1 to 1024+). |
| `scheduler_preemption_workload_disruptions` | HistogramVec | `ALPHA` | `preemptor` | `GenericWorkload` | Number of disrupted workload preemption units. |
| `scheduler_preemption_evaluation_duration_seconds` | HistogramVec | `ALPHA` | `preemptor`, `result` | `GenericWorkload` | Latency identifying preemption candidates (1ms to ~32.8s). |
| `scheduler_preemption_execution_duration_seconds` | HistogramVec | `ALPHA` | `preemptor`, `result` | `GenericWorkload` | Latency executing preemption victim deletions (1ms to ~32.8s). |
| `scheduler_preemption_pdb_violations_total` | CounterVec | `ALPHA` | `preemptor` | `GenericWorkload` | PDB violations caused by preemption. |
| `scheduler_podgroup_schedule_attempts_total` | CounterVec | `ALPHA` | `result`, `profile` | `GenericWorkload` | PodGroup scheduling attempts (`scheduled`, `unschedulable`, `error`). |
| `scheduler_podgroup_scheduling_attempt_duration_seconds` | HistogramVec | `ALPHA` | `result`, `profile` | `GenericWorkload` | E2E PodGroup scheduling attempt duration. |
| `scheduler_podgroup_scheduling_algorithm_duration_seconds` | Histogram | `ALPHA` | None | `GenericWorkload` | PodGroup scheduling algorithm duration. |
| `scheduler_generated_placements_total` | CounterVec | `ALPHA` | `profile` | `TopologyAwareWorkloadScheduling` | Number of candidate placements generated for pod groups. |
| `scheduler_placement_evaluations_total` | CounterVec | `ALPHA` | `result`, `profile` | `TopologyAwareWorkloadScheduling` | Candidate placements evaluated (`feasible`, `infeasible`). |
| `scheduler_placement_evaluation_duration_seconds` | HistogramVec | `ALPHA` | `result`, `profile` | `TopologyAwareWorkloadScheduling` | Placement evaluation latency. |
| `scheduler_dra_bindingconditions_allocations_total` | CounterVec | `ALPHA` | `profile`, `driver`, `status` | `DRADeviceBindingConditions` | DRA allocations using devices with BindingConditions. |
| `scheduler_dra_bindingconditions_wait_duration_seconds` | HistogramVec | `ALPHA` | `profile`, `driver`, `status` | `DRADeviceBindingConditions` | Time spent waiting for BindingConditions during PreBind. |
| `scheduler_async_api_call_execution_total` | CounterVec | `ALPHA` | `call_type`, `result` | `SchedulerAsyncAPICalls` | API calls executed via async dispatcher. |
| `scheduler_async_api_call_execution_duration_seconds` | HistogramVec | `ALPHA` | `call_type`, `result` | `SchedulerAsyncAPICalls` | Duration of async API calls. |
| `scheduler_pending_async_api_calls` | GaugeVec | `ALPHA` | `call_type` | `SchedulerAsyncAPICalls` | Number of calls pending in async dispatcher queue. |
| `scheduler_goroutines` | GaugeVec | `BETA` | `operation` | None | Running goroutines for operations (e.g., `binding`, `prioritizing_extender`). |
| `scheduler_cache_size` | GaugeVec | `ALPHA` | `type` | None | Items in scheduler cache (`nodes`, `pods`, `assumed_pods`). |

---

## 4. Histogram Bucket Distributions & Math

Histogram buckets in `pkg/scheduler/metrics` are tailored to specific latency profiles:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              Histogram Bucket Profiles                                 │
├────────────────────────┬────────────────┬──────────────┬───────────────┬───────────────┤
│ Metric                 │ Start Bucket   │ Growth Ratio │ Bucket Count  │ Upper Bound   │
├────────────────────────┼────────────────┼──────────────┼───────────────┼───────────────┤
│ plugin_execution...    │ 0.01ms (10μs)  │ 1.5x         │ 20 buckets    │ ~22.17ms      │
│ queueing_hint...       │ 0.01ms (10μs)  │ 1.5x         │ 20 buckets    │ ~22.17ms      │
│ framework_extension... │ 0.1ms (100μs)  │ 2.0x         │ 12 buckets    │ ~204.8ms      │
│ event_handling...      │ 0.1ms (100μs)  │ 2.0x         │ 12 buckets    │ ~204.8ms      │
│ batch_rescore...       │ 0.01ms (10μs)  │ 2.0x         │ 12 buckets    │ ~20.48ms      │
│ scheduling_attempt...  │ 1.0ms          │ 2.0x         │ 15 buckets    │ ~16.38s       │
│ scheduling_algorithm...│ 1.0ms          │ 2.0x         │ 15 buckets    │ ~16.38s       │
│ preemption_eval/exec   │ 1.0ms          │ 2.0x         │ 16 buckets    │ ~32.77s       │
│ pod_scheduling_sli...  │ 10.0ms         │ 2.0x         │ 20 buckets    │ ~87.38min     │
│ dra_bindingconditions..│ 100.0ms        │ 2.0x         │ 14 buckets    │ ~819.2s       │
│ preemption_victims     │ 1 victim       │ 2.0x         │ 7 buckets     │ [64, +Inf)    │
│ workload_preempt_vic.. │ 1 victim       │ 2.0x         │ 11 buckets    │ [1024, +Inf)  │
└────────────────────────┴────────────────┴──────────────┴───────────────┴───────────────┘
```

### Why Exponential Bucketing with 1.5x Multiplier?
For `plugin_execution_duration_seconds` and `queueing_hint_execution_duration_seconds`, a growth multiplier of **1.5** is chosen rather than the standard **2.0**. Because plugin filtering and queueing hints are evaluated millions of times across hundreds of nodes, minor microsecond regressions compound significantly. The 1.5x multiplier yields tighter statistical bounds between 10μs and 22ms without ballooning Prometheus metric cardinality.

---

## 5. Asynchronous Metric Recording Architecture

To maintain high scheduling throughput (hundreds of pods per second across thousands of nodes), recording metrics directly inside parallel Filter/Score loops or inside high-frequency queue event handlers would introduce severe cache line bouncing and lock contention.

`pkg/scheduler/metrics/metric_recorder.go` introduces the `MetricAsyncRecorder`:

```
           Scheduling Loop / Parallel Workers / Queue Handlers
                                 │
         ┌───────────────────────┼────────────────────────┐
         │ ObservePluginAsync()  │ ObserveInFlightAsync() │ ObserveCounterAsync()
         ▼                       ▼                        ▼
 ┌───────────────┐      ┌─────────────────┐      ┌────────────────┐
 │   bufferCh    │      │ aggregatedMap   │      │ counterBufferCh│
 │ (chan *hist,  │      │ (Lockless map   │      │ (chan *counter,│
 │  size: 1000)  │      │  under Q-lock)  │      │  size: 1000)   │
 └───────┬───────┘      └────────┬────────┘      └────────┬───────┘
         │                       │ Flush interval         │
         │                       ▼ (or forceFlush)        │
         │              ┌─────────────────┐               │
         │              │ inflightBufferCh│               │
         │              └────────┬────────┘               │
         │                       │                        │
         └───────────────────────┼────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ MetricAsyncRecorder.   │
                    │ run() Goroutine        │
                    │ (Flushes every 1s)     │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Prometheus Metric API  │
                    │ (Histogram.Observe,    │
                    │  Counter.Add, etc.)    │
                    └────────────────────────┘
```

### Key Concurrency Invariants:
1. **Non-Blocking Channel Ingestion**: All async observe methods (`ObservePluginDurationAsync`, `ObserveFrameworkExtensionPointDurationAsync`, `ObserveCounterAsync`) use non-blocking channel selects:
   ```go
   select {
   case r.bufferCh <- newMetric:
   default:
       // Discard if buffer reaches capacity (preserves scheduling latency under overload)
   }
   ```
2. **Lockless Aggregation for In-Flight Events**: `ObserveInFlightEventsAsync` aggregates deltas directly in `aggregatedInflightEventMetric map[gaugeVecMetricKey]int` without an internal mutex. It relies on the scheduling queue's existing lock (`activeQueue.lock`) held by the caller, and flushes to `aggregatedInflightEventMetricBufferCh` only after the interval expires or when `forceFlush=true` (e.g., when the last in-flight pod completes).
3. **Controlled Background Drain**: `run()` triggers `FlushMetrics()` every second (configurable), draining up to `bufferSize` items per loop to bound CPU consumption.

---

## 6. Queue Accounting & Multi-Entity Gauge Synchronization

The scheduling queue manages workloads of varying granularities: standalone `Pod`s, gang `PodGroup`s, and nested `CompositePodGroup`s.

`pkg/scheduler/metrics/metric_recorder.go` defines:
```go
type Entity interface {
    Size() int
    Type() fwk.EntityKeyType // "pod", "podgroup", "compositepodgroup"
}

type MetricRecorder interface {
    Add(entity Entity)
    Remove(entity Entity)
    Update(oldEntity, newEntity Entity)
    Clear()
}
```

### `QueuedEntitiesRecorder` Synchronization Mechanics:
- **Dual Tracking**: Tracks both individual pod capacity (`pending_pods` gauge) and scheduling unit entities (`queued_entities` gauge).
- **`Add(entity)`**:
  - `pending_pods.Add(float64(entity.Size()))`
  - `queued_entities.WithLabelValues(queue, entity.Type()).Inc()`
- **`Remove(entity)`**:
  - `pending_pods.Add(-float64(entity.Size()))`
  - `queued_entities.WithLabelValues(queue, entity.Type()).Dec()`
- **`Update(oldEntity, newEntity)`**:
  - Updates `pending_pods` by `diff = newEntity.Size() - oldEntity.Size()`.
  - `queued_entities` is unchanged because entity identity and type remain constant across updates.

```
Queue State Machine Metrics Mapping:
   ┌────────────────┐   Add()    ┌──────────────────────────────────────────────┐
   │ Incoming Event ├───────────►│ activeQ                                      │
   └────────────────┘            │ - pending_pods{queue="active"}               │
                                 │ - queued_entities{queue="active",type=...}   │
                                 └──────────────────────┬───────────────────────┘
                                                        │ Pop() (In-Flight)
                                                        ▼
                                 ┌──────────────────────────────────────────────┐
                                 │ Unschedulable / Gated / Backoff              │
                                 │ - pending_pods{queue="backoff|unsched|gated"}│
                                 │ - queued_entities{queue=...,type=...}        │
                                 │ - unschedulable_pods{plugin=...,profile=...} │
                                 └──────────────────────────────────────────────┘
```

---

## 7. Resource Consumption Collector (`pkg/scheduler/metrics/resources`)

The `pkg/scheduler/metrics/resources` subpackage provides detailed visibility into cluster resource allocation as interpreted by `kube-scheduler` and `kubelet`.

### 7.1 Architecture & Isolation

Unlike general scheduler metrics that reside on the default registry (`legacyregistry.DefaultGatherer`), the resource collector:
1. Implements `metrics.StableCollector` via `podResourceCollector`.
2. Registers on a dedicated `metrics.NewKubeRegistry()` served via `resources.Handler(podLister)`.
3. Protects the main `/metrics` endpoint from heavy `O(pods)` payload serializations by exposing a dedicated endpoint (typically `/metrics/resources`).

```
                    ┌────────────────────────────┐
                    │   client-go PodLister      │
                    └─────────────┬──────────────┘
                                  │ List(labels.Everything())
                                  ▼
                    ┌────────────────────────────┐
                    │    podResourceCollector    │
                    │   (CollectWithStability)   │
                    └─────────────┬──────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│   kube_pod_resource_request     │       │     kube_pod_resource_limit     │
│   (Gauge: requested resources)  │       │     (Gauge: resource limits)    │
└─────────────────────────────────┘       └─────────────────────────────────┘
```

### 7.2 Metric Schema & Dimensionality

Both `kube_pod_resource_request` and `kube_pod_resource_limit` share an identical label schema:

```
Labels:
  - namespace:       Pod namespace
  - pod:             Pod name
  - node:            Assigned node (.spec.nodeName)
  - scheduler:       Scheduler profile name (.spec.schedulerName)
  - priority:        Formatted pod priority string (.spec.priority)
  - resource:        Resource name (cpu, memory, ephemeral-storage, storage, hugepages-*, etc.)
  - unit:            Standardized unit ("cores", "bytes", "integer")
```

### 7.3 Performance Optimizations & Normalization:
- **Allocation Reuse**: Uses reusable `v1.ResourceList` buffers (`reuseReqs`, `reuseLimits`) in `podRequestsAndLimitsByLifecycle` to prevent heap churn when iterating thousands of pods.
- **Terminal Pod Exclusion**: Ignores pods in terminal phases (`PodSucceeded`, `PodFailed`) and unassigned pods (`len(pod.Spec.NodeName) == 0`).
- **Overhead Accounting**: Adds `Pod.Spec.Overhead` to container request and limit sums using `k8s.io/component-helpers/resource`.
- **Zero-Value Suppression**: Omits resources with zero quantity (`if val.IsZero() { return }`) to minimize metric cardinality.
- **Unit Normalization**:
  - `cpu` -> `"cores"`
  - `memory`, `storage`, `ephemeral-storage`, hugepages -> `"bytes"`
  - Attachable volume limits -> `"integer"`

---

## 8. Metric Registration & Feature Gate Lifecycle

Metric registration is governed by `metrics.Register()` and executed once via `sync.Once`:

```go
func Register() {
    registerMetrics.Do(func() {
        InitMetrics()
        RegisterMetrics(metricsList...)
        volumebindingmetrics.RegisterVolumeSchedulingMetrics()

        if utilfeature.DefaultFeatureGate.Enabled(features.SchedulerAsyncPreemption) {
            RegisterMetrics(PreemptionGoroutinesDuration, PreemptionGoroutinesExecutionTotal)
        }
        if utilfeature.DefaultFeatureGate.Enabled(features.SchedulerAsyncAPICalls) {
            RegisterMetrics(AsyncAPICallsTotal, AsyncAPICallDuration, AsyncAPIPendingCalls)
        }
        if utilfeature.DefaultFeatureGate.Enabled(features.DRAExtendedResource) {
            resourceclaimmetrics.RegisterMetrics()
        }
        if utilfeature.DefaultFeatureGate.Enabled(features.GenericWorkload) {
            RegisterMetrics(
                podGroupScheduleAttempts,
                podGroupSchedulingLatency,
                PodGroupSchedulingAlgorithmLatency,
                WorkloadPreemptionAttempts,
                WorkloadPreemptionVictims,
                PreemptionWorkloadDisruptions,
                PreemptionEvaluationDuration,
                PreemptionExecutionDuration,
                PreemptionPDBViolations,
            )
        }
        if utilfeature.DefaultFeatureGate.Enabled(features.TopologyAwareWorkloadScheduling) {
            RegisterMetrics(GeneratedPlacementsTotal, PlacementEvaluations, PlacementEvaluationDuration)
        }
        if utilfeature.DefaultFeatureGate.Enabled(features.DRADeviceBindingConditions) {
            RegisterMetrics(DRABindingConditionsAllocationsTotal, DRABindingConditionsPreBindDuration)
        }
    })
}
```

### Out-of-Tree Plugin Metric Registration:
Out-of-tree plugins should use the exported `RegisterMetrics(extraMetrics ...metrics.Registerable)` to register custom collectors into the scheduler's `legacyregistry`.

---

## 9. Monitoring Best Practices, SLOs & Alerting Playbook

### 9.1 Core Golden Signals & Recommended Alerts

```
                               Golden Signals Map
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ Signal                       │ Metric / Expression                                    │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ Latency (SLI)                │ histogram_quantile(0.99, sum(rate(                     │
│                              │   scheduler_pod_scheduling_sli_duration_seconds_bucket │
│                              │   [5m])) by (le))                                      │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ Scheduling Loop Latency      │ histogram_quantile(0.99, sum(rate(                     │
│                              │   scheduler_scheduling_attempt_duration_seconds_bucket │
│                              │   [5m])) by (le, result))                              │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ Throughput                   │ sum(rate(scheduler_schedule_attempts_total[5m]))       │
│                              │   by (result, profile)                                 │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ Rejection / Unschedulable    │ sum(scheduler_unschedulable_pods) by (plugin)          │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ Queue Starvation / Backlog   │ sum(scheduler_pending_pods) by (queue)                 │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ QueueingHint Flush Leak      │ rate(scheduler_pod_scheduled_after_flush_total[5m]) > 0│
└──────────────────────────────┴────────────────────────────────────────────────────────┘
```

### 9.2 Diagnosing Production Incidents

#### Scenario A: High End-to-End Scheduling Latency (`pod_scheduling_sli_duration_seconds`)
1. Compare `scheduler_pod_scheduling_sli_duration_seconds` against `scheduler_scheduling_attempt_duration_seconds`.
2. If single-attempt duration is low (<10ms) but SLI duration is high (>10s), pods are failing initial attempts and cycling through `backoffQ` and `unschedulableEntities`.
3. Check `scheduler_pod_scheduling_attempts` to verify how many attempts pods need.
4. Inspect `scheduler_unschedulable_pods` by `plugin` to identify the blocking constraint (e.g. `NodeResourcesFit`, `PodTopologySpread`).

#### Scenario B: Slow Scheduling Algorithm Latency (`scheduling_algorithm_duration_seconds`)
1. Inspect `scheduler_framework_extension_point_duration_seconds` for `PreFilter`, `Filter`, `PreScore`, `Score`.
2. Drill down into `scheduler_plugin_execution_duration_seconds{extension_point="..."}` to locate the slow plugin.
3. Check `scheduler_cache_size{type="nodes"}` and `scheduler_cache_size{type="pods"}` to determine if cluster scale is impacting iteration costs.

#### Scenario C: QueueingHint Misconfiguration / Flush Spikes
1. If `scheduler_pod_scheduled_after_flush_total` is climbing, pods are staying in `unschedulableEntities` until the periodic 5-minute queue flush rather than being awakened promptly by cluster events.
2. Check `scheduler_queueing_hint_execution_duration_seconds` and `scheduler_pre_queueing_hint_evaluations_total` for custom or in-tree plugins.

#### Scenario D: Async API Dispatcher Bottlenecks
1. When `SchedulerAsyncAPICalls` is enabled, monitor `scheduler_pending_async_api_calls`.
2. A persistent increase indicates API server rate limiting or network latency; correlate with `scheduler_async_api_call_execution_duration_seconds`.

---

## 10. Developer Guide & Testing Patterns

### 10.1 Adding a New Metric
1. Define the metric variable in `metrics.go` using `k8s.io/component-base/metrics`.
2. Specify explicit `StabilityLevel` (`STABLE`, `BETA`, or `ALPHA`).
3. If ALPHA or feature-gated:
   - Do NOT add to `metricsList`.
   - Add conditional registration inside `Register()` checking `utilfeature.DefaultFeatureGate.Enabled(...)`.
4. If high-frequency (invoked per node / per plugin):
   - Add async observer helper to `MetricAsyncRecorder` in `metric_recorder.go`.
   - Choose appropriate histogram buckets with 1.5x or 2.0x growth factor.
5. Add unit test assertions in `metric_recorder_test.go` or `profile_metrics_test.go`.

### 10.2 Testing Metric Observers
When testing metrics in unit tests:
```go
// Reset registry and initialize
metrics.InitMetrics()
testRegistry := metrics.NewKubeRegistry()
testRegistry.MustRegister(metrics.YourMetric)
defer metrics.YourMetric.Reset()

// Trigger scheduler logic ...

// Gather and assert
metricFamilies, err := testRegistry.Gather()
require.NoError(t, err)
// Validate values and labels
```
