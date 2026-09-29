# AGENTS.md: Developer & Agent Guide for `pkg/scheduler/framework/runtime`

This guide provides an architectural overview, execution flow, status coordination rules, plugin lifecycle management, error handling invariants, and testing strategies for the Kubernetes scheduler runtime package in `pkg/scheduler/framework/runtime`.

---

## 1. High-Level Purpose & Scope

The `pkg/scheduler/framework/runtime` package provides the concrete implementation of the Kubernetes Scheduling Framework (`fwk.Framework` and `fwk.Handle` interfaces). It serves as the orchestration engine of `kube-scheduler`, responsible for:

1. **Plugin Instantiation & Configuration**: Initializing plugins from a `Registry` based on `KubeSchedulerProfile` definitions, managing configuration decoding, MultiPoint plugin expansions, and score weight validation.
2. **Scheduling Cycle Execution**: Orchestrating extension points (PreEnqueue, QueueSort, PreFilter, Filter, PostFilter, PreScore, Score, Reserve, Permit, and PodGroup / Placement extensions) in a synchronous scheduling cycle per pod or pod group.
3. **Binding Cycle Execution**: Coordinating asynchronous binding phases (PreBind, Bind, PostBind) in dedicated goroutines, supporting parallel pre-bind execution and cancellation tracking.
4. **Status & Error Coordination**: Translating plugin return statuses (`fwk.Status`), handling short-circuit vs aggregation semantics, managing preemption triggers, and attributing plugin failures.
5. **Plugin Lifecycle & Concurrency**: Managing waiting pods during gang/permit delays (`waitingPodsMap`), tracking in-flight binding operations (`podsInPreBindMap`), caching node hints across cycles (`OpportunisticBatch`), and ensuring clean shutdown.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/runtime/
├── framework.go               # Core frameworkImpl struct, NewFramework, and extension point runners
├── registry.go                # Plugin Registry, factory types, FactoryAdapter, and DecodeInto
├── batch.go                   # OpportunisticBatching and PodSignature candidate rescoring
├── instrumented_plugins.go    # Metric-recording wrappers for Filter/PreFilter/Score/PreScore plugins
├── pods_in_prebind_map.go     # Thread-safe map and cancellation handles for pods in PreBind
├── waiting_pods_map.go        # Thread-safe map, timers, and signal channels for pods waiting on Permit
├── framework_test.go          # Unit tests covering all extension points, MultiPoint, weights, and errors
├── registry_test.go           # Tests for registry registration, unregistration, merging, and decoding
├── batch_test.go              # Tests for opportunistic batch caching, invalidation, and rescoring
├── waiting_pods_map_test.go   # Tests for waitingPod Permit lifecycle (Allow, Reject, Preempt, timeout)
├── pods_in_prebind_map_test.go# Tests for PreBind tracking, cancellation, and MarkPrebound
└── AGENTS.md                  # This agent documentation
```

---

## 3. `frameworkImpl` Struct & Core Handles

`frameworkImpl` implements both `framework.Framework` (the execution engine) and `fwk.Handle` (the interface passed to plugins allowing them to inspect cluster state and invoke other plugins).

```
                            ┌────────────────────────┐
                            │     frameworkImpl      │
                            └───────────┬────────────┘
                                        │
     ┌──────────────────────────────────┼──────────────────────────────────┐
     ▼                                  ▼                                  ▼
[ Plugin Pipelines ]          [ State & Lister Handles ]          [ Lifecycle & Async ]
• preFilterPlugins            • snapshotSharedLister              • waitingPods (*waitingPodsMap)
• filterPlugins               • mutableSnapshotLister             • podsInPreBind (*podsInPreBindMap)
• postFilterPlugins           • clientSet / kubeConfig            • batch (*OpportunisticBatch)
• preScorePlugins             • informerFactory                   • metricsRecorder
• scorePlugins                • sharedDRAManager                  • parallelizer (parallelize.Parallelizer)
• reservePlugins              • sharedCSIManager                  • apiDispatcher / apiCacher
• permitPlugins               • eventRecorder                     • preemptionManager
• preBindPlugins / bindPlugins• podGroupManager                   • scorePluginWeight map
```

### Key Struct Fields:
- **`pluginsMap map[string]fwk.Plugin`**: Unified map of all initialized plugin instances keyed by plugin name.
- **`scorePluginWeight map[string]int`**: Map of validated non-zero integer weights for Score plugins.
- **`waitingPods *waitingPodsMap`**: Coordinates Permit waiting states across goroutines.
- **`podsInPreBind *podsInPreBindMap`**: Tracks pods in the PreBind phase, enabling cancellation if preemption occurs or context expires.
- **`batch *OpportunisticBatch`**: Caches filtering/scoring results when `OpportunisticBatching` feature gate is enabled.
- **`parallelizer fwk.Parallelizer`**: Worker pool dispatcher used to parallelize node filtering, scoring, normalization, and pre-bind executions.

---

## 4. Plugin Registry & Framework Initialization (`NewFramework`)

Framework construction executes in strict sequential phases within `NewFramework(ctx, registry, profile, opts...)`:

```
1. Option Application (defaultFrameworkOptions -> Custom Options)
         │
2. Subsystem Setup (OpportunisticBatch, Extender defaultEnqueueExtension)
         │
3. Profile Inspection & Plugin Deduplication (f.pluginsNeeded)
         │
4. Plugin Instantiation (Registry Factory calls with Decoded Args)
         │
5. EnqueueExtensions Registration (fillEnqueueExtensions)
         │
6. Extension Point Population (updatePluginList per extension point)
         │
7. MultiPoint Expansion & Override Resolution (expandMultiPointPlugins)
         │
8. Structural Invariant Validation (QueueSort == 1, Bind >= 1, Score weights)
         │
9. Batching & Signature Computation (computeBatchablePlugins)
         │
10. Metric Instrumentation (setInstrumentedPlugins)
```

### 4.1. Registry & Factory Adapters (`registry.go`)
- **`PluginFactory`**: `func(ctx context.Context, configuration runtime.Object, f fwk.Handle) (fwk.Plugin, error)`.
- **`PluginFactoryWithFts` & `FactoryAdapter`**: Allows plugins to declare feature gate dependencies (`plfeature.Features`) cleanly while adapting to the standard `PluginFactory` signature.
- **`DecodeInto`**: Decodes `*runtime.Unknown` plugin args into strongly typed structs from JSON or YAML raw buffers.

### 4.2. MultiPoint Expansion (`expandMultiPointPlugins`)
Plugins configured under `profile.Plugins.MultiPoint.Enabled` are dynamically expanded into all individual extension point slices whose interface they implement:
- **Overrides**: If a plugin is enabled both in `MultiPoint` and explicitly in a regular extension point, the explicit configuration takes precedence (preserving its defined ordering).
- **Disabled Sets**: If a plugin is in `Disabled` for a specific extension point (or `Disabled: [{Name: "*"}]`), MultiPoint expansion skips that extension point.
- **Resulting Order**:
  1. Regular extension point override plugins (original order).
  2. MultiPoint-enabled plugins (order defined in MultiPoint).
  3. Non-overridden regular extension point plugins.

### 4.3. Score Weight Validation
- Every configured Score plugin must have a weight in `[1, MaxWeight]`.
- Total weight across all Score plugins cannot exceed `MaxTotalScore` (`math.MaxInt64 / MaxScore`).

---

## 5. Extension Point Execution & Status Coordination

The framework executes extension points according to specialized control flow, short-circuit, and aggregation semantics:

| Extension Point | Concurrency | Success Condition | Short-Circuit Behavior / Failure Semantics |
|---|---|---|---|
| **PreEnqueue** | Sequential | All return `Success` | Aborts enqueueing if any plugin returns non-success. |
| **QueueSort** | Single Plugin | `Less(p1, p2) bool` | Exactly one plugin active; determines heap priority. |
| **PreFilter** | Sequential | All return `Success`/`Skip` | • `UnschedulableAndUnresolvable`: immediate abort (no preemption).<br>• `Unschedulable`: accumulates reasons across **all** plugins for preemption.<br>• `Skip`: skips paired Filter & PreFilterExtensions.<br>• Node intersection: if `result.NodeNames` becomes empty, returns `UnschedulableAndUnresolvable`. |
| **Filter** | Parallel per Node | All return `Success` | • Short-circuits on first failing plugin for a given node.<br>• `Skip`: respected via `state.GetSkipFilterPlugins()`.<br>• Non-rejected statuses treated as `Error`. |
| **PostFilter** | Sequential | First `Success` | • Evaluated only when 0 nodes pass Filter.<br>• Runs until first plugin returns `Success` or `UnschedulableAndUnresolvable`.<br>• Aggregates `Unschedulable` reject reasons. |
| **PreScore** | Sequential | All return `Success`/`Skip` | • `Skip`: skips paired Score plugin in `CycleState.SetSkipScorePlugins()`.<br>• Aborts on any non-success/non-skip status. |
| **Score** | Parallel per Node | All return `Success` | • Evaluates `Score()` in parallel across candidate nodes.<br>• Cancels via `parallelize.ResultChannel` on first error.<br>• Runs `NormalizeScore` in parallel per plugin.<br>• Applies weights and verifies scores are in `[MinScore, MaxScore]`. |
| **Reserve** | Sequential | All return `Success` | • Runs `Reserve()` sequentially.<br>• On failure, stops and caller runs `RunReservePluginsUnreserve()`. |
| **Unreserve** | Sequential (Reverse) | Informational | • Runs in **strictly reverse order** of Reserve plugins to ensure symmetric teardown. |
| **Permit** | Sequential | `Success` or `Wait` | • `Reject`: immediately marks pod unschedulable.<br>• `Wait`: aggregates plugin durations (clamped to 15m), constructs `pluginsWaitTime`, and returns `Wait`.<br>• `Error`: returns error status. |
| **PreBind** | Sequential / Parallel Groups | All return `Success` | • `PreBindPreFlight` identifies `AllowParallel` plugins and `Skip` plugins.<br>• Chunks plugins into parallel vs single groups (`getPreBindPluginGroups`).<br>• Supports cancelation via `podsInPreBindMap`. |
| **Bind** | Sequential | First non-`Skip` | • Runs until first plugin returns non-`Skip`.<br>• Exactly one plugin binds the pod to the API server. |
| **PostBind** | Sequential | Informational | • Runs all post-bind plugins sequentially for metrics and cleanup. |

---

## 6. Deep Dive: Key Runtime Subsystems

### 6.1. Permit Lifecycle & `waitingPodsMap` (`waiting_pods_map.go`)

```
               RunPermitPlugins() returns Status(Wait)
                               │
                               ▼
              frameworkImpl.AddWaitingPod(pod, waitTimes)
                               │
                      [ newWaitingPod ]
        ┌──────────────────────┴──────────────────────┐
        ▼                                             ▼
  time.AfterFunc() timers                     w.s (chan *Status, buf=1)
  (1 per waiting plugin)                              │
        │                                             │
        ├───────── Timeout / Reject / Preempt ────────┤
        │                     │                       │
        │                     ▼                       ▼
        │          w.stopWithStatus(...) ──────> Send to w.s
        │                                             ▲
        └───────── All Plugins Allow() ───────────────┘
                              │
                    WaitOnPermit(ctx, pod)
                              │
                    Blocks on `<-w.s`
                              │
                   Unblocks & Removes Pod
```

- **Allow**: Each plugin calls `waitingPod.Allow(pluginName)`. The timer is stopped and removed. When `len(pendingPlugins) == 0`, a `Success` status is delivered to `w.s`.
- **Reject / Preempt / Timeout**: Triggers `stopWithStatus()`, stops all pending timers, and delivers `Unschedulable` (Reject/Timeout) or `Error` (Preempt) status to `w.s`.
- **Thread Safety**: Protected by `sync.RWMutex` on both `waitingPodsMap` and individual `waitingPod` instances.

### 6.2. PreBind Coordination & `podsInPreBindMap` (`pods_in_prebind_map.go`)
- Tracks pods currently executing `PreBind`.
- When a pod is preempted or canceled while waiting on slow volume/DRA attachments, `CancelPod(message)` cancels the context cause via `context.CancelCauseFunc`.
- `MarkPrebound()` transitions state before calling `RunBindPlugins`, ensuring binding cannot be canceled once pre-bind completes.

### 6.3. Opportunistic Batching (`batch.go`)
When `OpportunisticBatching` is enabled:
1. `SignPod`: Generates a deterministic JSON signature (`PodSignature`) combining scheduler name and fragments from plugins implementing `fwk.SignPlugin`.
2. `GetNodeHint`: Checks if consecutive pods share the same signature, verifies the previous cycle succeeded, verifies cache age (`DefaultMaxBatchAge = 500ms`), and re-filters/re-scores candidate nodes.
3. `StoreScheduleResults`: Caches remaining scored nodes in `framework.SortedScoredNodes` for subsequent pods.

### 6.4. Metric Instrumentation (`instrumented_plugins.go`)
`setInstrumentedPlugins()` wraps active `PreFilter`, `Filter`, `PreScore`, and `Score` plugins in decorator structs:
- Increments `metrics.PluginEvaluationTotal` with labels `(plugin, extension_point, profile)`.
- Skips metric increments when status is `Skip`.

---

## 7. Status Return Codes & Error Handling Conventions

`fwk.Status` encapsulates execution outcomes via `fwk.Code`:

| Status Code | Meaning | Framework Action |
|---|---|---|
| **`Success`** (0) | Operation completed successfully. | Continue to next plugin/phase. |
| **`Error`** (1) | Internal error during execution. | Abort cycle; trigger cleanup & requeue. |
| **`Unschedulable`** (2) | Pod cannot fit on node / cluster. | Accumulate reasons; trigger PostFilter/preemption if applicable. |
| **`UnschedulableAndUnresolvable`** (3) | Pod cannot fit and preemption cannot resolve it. | Abort scheduling cycle immediately; skip PostFilter. |
| **`Wait`** (4) | Permit plugin holds pod binding. | Add to `waitingPodsMap`; block in `WaitOnPermit`. |
| **`Skip`** (5) | Plugin declines execution for this pod. | Skip paired extension points; continue pipeline. |
| **`Pending`** (6) | Asynchronous binding / claim in progress. | Handled by dynamic resource / volume workflows. |

### Error Wrapping & Attribution:
- Always preserve plugin attribution using `status.WithPlugin(pl.Name())` or `s.SetPlugin(pl.Name())`.
- Wrap errors using `fmt.Errorf("running %q <ExtensionPoint>: %w", pl.Name(), err)`.
- Use `parallelize.NewResultChannel` and `context.WithCancelCause` for fail-fast goroutine termination.

---

## 8. Testing Guide & Coverage

The test suite in `pkg/scheduler/framework/runtime/` provides comprehensive coverage:

### 8.1. Test Suites:
- **`framework_test.go`**:
  - `TestNewFramework`: Tests MultiPoint expansion, duplicate plugins, missing factories, queue sort validation, and score weight overflow.
  - `TestRunPreFilterPlugins` & `TestRunFilterPlugins`: Tests short-circuiting, `Skip` propagation, node filtering, and `UnschedulableAndUnresolvable` semantics.
  - `TestRunScorePlugins` & `TestNormalizeScore`: Tests parallel score collection, normalization errors, out-of-bounds scores, and weight multiplication.
  - `TestRunReservePlugins`: Tests Reserve failure abort and Unreserve reverse-order execution.
  - `TestRunPermitPlugins` & `TestWaitOnPermit`: Tests permit timeout clamping, reject, allow, and channel synchronization.
  - `TestRunPreBindPlugins` & `TestRunBindPlugins`: Tests parallel pre-bind chunking, first-match bind execution, and pre-flight skipping.
- **`batch_test.go`**: Tests `OpportunisticBatch` hint generation, signature mismatch invalidation, timeout expiry, and candidate rescoring.
- **`waiting_pods_map_test.go`**: Tests concurrent permit resolutions, timer cancellations, and race-free status delivery.
- **`pods_in_prebind_map_test.go`**: Tests cancellation propagation and pre-bound state transitions.
- **`registry_test.go`**: Tests factory registrations, unregistrations, duplicates, and configuration decoding.

### 8.2. Running Tests:
```bash
# Run all tests in the runtime package
GOTOOLCHAIN=auto go test -v ./pkg/scheduler/framework/runtime/...

# Run with race detector
GOTOOLCHAIN=auto go test -v -race ./pkg/scheduler/framework/runtime/...
```

---

## 9. Critical Invariants for Developers & Agents

1. **CycleState Thread Safety & Lifetime**:
   - `CycleState` is created anew for each scheduling cycle. Data written in `PreFilter` or `PreScore` is accessed concurrently by multiple goroutines in `Filter` and `Score`. Ensure stored objects are immutable or thread-safe.
2. **Reserve / Unreserve Symmetry**:
   - Every stateful operation in `Reserve` MUST be undone in `Unreserve`. `Unreserve` is always executed in the **reverse order** of `Reserve` to prevent orphaned locks or allocations.
3. **Single Bind Execution**:
   - `RunBindPlugins` stops at the **first non-Skip plugin**. Never configure multiple binding plugins expecting chained execution; only one plugin executes binding.
4. **Permit Non-Blocking Signals**:
   - `waitingPod.Allow`, `Reject`, and `Preempt` use buffered channels (`make(chan *fwk.Status, 1)`) with non-blocking sends to prevent goroutine leaks if the waiting context is abandoned.
5. **Context Propagation & Cause Logging**:
   - Always propagate context with contextual loggers (`klog.LoggerWithName(logger, pl.Name())`). Log errors using `klog.FromContext(ctx)`.
6. **Skip Status Integrity**:
   - When returning `fwk.Status(Skip)` in `PreFilter` or `PreScore`, the runtime records the skip in `CycleState`. Do not assume downstream `Filter` or `Score` will execute for that plugin.
