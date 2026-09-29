# `pkg/scheduler/profile` - Agent Overview & Developer Guide

This guide provides AI agents and human developers with an architectural overview, dispatching mechanisms, configuration parsing and validation invariants, lifecycle management, and testing patterns for the multi-profile subsystem in `pkg/scheduler/profile`.

---

## 1. High-Level Overview & Role

The `pkg/scheduler/profile` package provides the multi-profile management layer for the Kubernetes Scheduler (`kube-scheduler`).

In Kubernetes, a single `kube-scheduler` process can manage multiple distinct scheduling profiles simultaneously. Each profile corresponds to a distinct `KubeSchedulerProfile` configuration and operates as an independent instance of `framework.Framework`. Pods select their target profile by specifying `spec.schedulerName`.

### Core Responsibilities:
1. **Multi-Profile Framework Instantiation (`NewMap`)**: Constructs and initializes individual `framework.Framework` instances for each profile configuration using the provided plugin registry, recorder factory, and framework options.
2. **Configuration Validation & Invariant Enforcement (`cfgValidator`)**: Enforces multi-profile constraints, such as profile name uniqueness and strict identity across `QueueSort` plugins and plugin arguments.
3. **Pod Routing & Dispatching (`FrameworkForPod`, `HandlesSchedulerName`)**: Routes incoming pods to their designated framework instance based on `pod.Spec.SchedulerName`, including defaulting empty scheduler names to `v1.DefaultSchedulerName` (`"default-scheduler"`).
4. **Isolated Event Recording (`RecorderFactory`)**: Provides isolated `events.EventRecorderLogger` instances per profile so emitted Kubernetes events (such as `Scheduled`, `FailedScheduling`) reflect the appropriate profile/scheduler name.
5. **Lifecycle Management (`Close`)**: Cleanly tears down all registered framework instances, ensuring background goroutines, timers, and plugin handles are properly closed.

---

## 2. Architecture & File Map

```
pkg/scheduler/profile/
├── profile.go          # Profile constructors, Map type, routing helpers, and cfgValidator
├── profile_test.go     # Unit tests for Map creation, validation rules, routing, and Close
└── AGENTS.md           # This agent documentation
```

### Architectural Position within `pkg/scheduler`:

```
                           ┌───────────────────────────────┐
                           │   KubeSchedulerConfiguration  │
                           │  - Profiles: []ProfileConfig  │
                           └───────────────┬───────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ pkg/scheduler/profile.Map                                                              │
│                                                                                        │
│   "default-scheduler"           "custom-batch-profile"         "gpu-cosched-profile"   │
│   ┌───────────────────────┐     ┌───────────────────────┐      ┌────────────────────┐  │
│   │ framework.Framework 1 │     │ framework.Framework 2 │      │framework.Framework3│  │
│   │  - Plugin Pipeline 1  │     │  - Plugin Pipeline 2  │      │ - Plugin Pipeline 3│  │
│   │  - EventRecorder 1    │     │  - EventRecorder 2    │      │ - EventRecorder 3  │  │
│   │  - Score Weights 1    │     │  - Score Weights 2    │      │ - Score Weights 3  │  │
│   └───────────────────────┘     └───────────────────────┘      └────────────────────┘  │
└──────────────────────────────────────────▲─────────────────────────────────────────────┘
                                           │ FrameworkForPod(pod) / HandlesSchedulerName
                                           │
                   ┌───────────────────────┴───────────────────────┐
                   │                                               │
           [ Event Handlers ]                            [ Scheduling Cycle ]
       responsibleForPod(pod, Map)                    scheduleOne / FrameworkForPod
```

---

## 3. Core Types & Key Functions

### 3.1 `Map`
```go
type Map map[string]framework.Framework
```
- A map from `schedulerName` (`string`) to initialized `framework.Framework` instances.
- Populated once during scheduler startup (`NewMap`) and treated as read-only throughout the scheduler lifecycle.

### 3.2 `RecorderFactory`
```go
type RecorderFactory func(string) events.EventRecorderLogger
```
- Factory function creating a profile-specific `events.EventRecorderLogger` given a `schedulerName`.
- `NewRecorderFactory(b events.EventBroadcaster) RecorderFactory`: Helper that constructs recorder instances using `b.NewRecorder(scheme.Scheme, name)`.

### 3.3 `NewMap`
```go
func NewMap(
    ctx context.Context,
    cfgs []config.KubeSchedulerProfile,
    r frameworkruntime.Registry,
    recorderFact RecorderFactory,
    opts ...frameworkruntime.Option,
) (Map, error)
```
1. Iterates over every `config.KubeSchedulerProfile` configuration.
2. Invokes internal `newProfile` to create the `framework.Framework` instance via `frameworkruntime.NewFramework`.
3. Validates the profile against internal multi-profile rules (`cfgValidator.validate`).
4. Registers the framework in `Map[cfg.SchedulerName]`.

### 3.4 `HandlesSchedulerName` & `FrameworkForPod`
- **`HandlesSchedulerName(name string) bool`**:
  Returns `true` if `name` is present in `Map`. Used by informer event handlers (`responsibleForPod`) to ignore pods belonging to other scheduler instances.
- **`FrameworkForPod(pod *v1.Pod) (framework.Framework, error)`**:
  Resolves the framework corresponding to `pod.Spec.SchedulerName`.
  - If `pod.Spec.SchedulerName == ""`, defaults to `v1.DefaultSchedulerName` (`"default-scheduler"`).
  - Returns `fmt.Errorf("profile not found for scheduler name %q", name)` if no profile is found.

### 3.5 `Close`
```go
func (m Map) Close() error
```
- Iterates through all registered frameworks and calls `f.Close()`.
- Collects errors using `errors.Join(errs...)` so failure to close one framework does not prevent attempting to close the others.

---

## 4. Multi-Profile Shared vs. Isolated Subsystems

When multiple profiles are configured in `kube-scheduler`, certain subsystems are shared globally across the entire process, while others are isolated per profile:

| Subsystem | Scope | Rationale / Behavior |
|---|---|---|
| **`SchedulingQueue` (PriorityQueue)** | **Shared** | Single queue manages all pods (`activeQ`, `backoffQ`, `unschedulablePods`) across all profiles. |
| **`QueueSortPlugin`** | **Shared (Strict)** | Because there is only one `SchedulingQueue`, all profiles must configure the identical `QueueSort` plugin and identical `PluginConfig` arguments. |
| **Scheduler Cache & Node Snapshot** | **Shared** | Node states, assumed pods, and node snapshots are shared globally. |
| **Informer Event Handlers** | **Shared** | Single set of informer handlers; filters pods via `responsibleForPod(pod, profiles)`. |
| **Shared Managers (DRA, CSI, Preemption)** | **Shared** | Injected across all frameworks via `frameworkruntime.Option` slices during initialization. |
| **Plugin Pipelines** | **Per-Profile** | Each profile configures its own `PreFilter`, `Filter`, `PostFilter`, `PreScore`, `Score`, `Reserve`, `Permit`, `PreBind`, `Bind`, `PostBind`, and PodGroup plugins. |
| **Score Weights** | **Per-Profile** | Scoring plugins and their respective weights are independently calculated per profile. |
| **Event Recorders** | **Per-Profile** | Events report the profile's `SchedulerName` as the source component. |
| **`PercentageOfNodesToScore`** | **Per-Profile** | Overrides node sampling stopping threshold on a per-profile basis. |
| **Pod Signing & Queueing Hints** | **Per-Profile** | Queueing hints and pod signing functions are extracted per profile name in `scheduler.go`. |

---

## 5. Configuration Validation Invariants (`cfgValidator`)

During `NewMap` execution, `cfgValidator` performs critical consistency checks:

```go
type cfgValidator struct {
    m             Map
    queueSort     string
    queueSortArgs runtime.Object
}
```

1. **Non-Empty Scheduler Name**: `len(f.ProfileName()) > 0`. Every profile must declare a non-empty `schedulerName`.
2. **Non-Nil Plugins**: `cfg.Plugins != nil`. Every profile must declare a plugin configuration.
3. **No Duplicate Profile Names**: `v.m[f.ProfileName()] == nil`. Each profile name must be unique.
4. **QueueSort Plugin Identity**:
   - `queueSort := f.ListPlugins().QueueSort.Enabled[0].Name`
   - The first profile sets `v.queueSort` and `v.queueSortArgs`.
   - Every subsequent profile must match `v.queueSort == queueSort`. If different:
     `"different queue sort plugins for profile %q: %q, first: %q"`
5. **QueueSort Plugin Arguments Identity**:
   - `diff.Diff(v.queueSortArgs, queueSortArgs) == ""`
   - If args differ between profiles:
     `"different queue sort plugin args for profile %q"`
6. **Framework Invariants (delegated to `frameworkruntime.NewFramework`)**:
   - Exactly one `QueueSort` plugin enabled per profile.
   - At least one `Bind` plugin enabled per profile.
   - At most one `PlacementGenerate` plugin enabled per profile.
   - Valid, non-zero weights for all enabled `Score` and `PlacementScore` plugins.

---

## 6. Multi-Profile Dispatching & Routing Lifecycle

```
1. Pod Event Ingress (Informer Handler)
   └── eventhandlers.go: responsibleForPod(pod, sched.Profiles)
       └── profile.Map.HandlesSchedulerName(pod.Spec.SchedulerName)
           ├── [Match] -> Enqueue into sched.SchedulingQueue
           └── [No Match] -> Ignore pod (handled by another scheduler)

2. Scheduling Queue (PriorityQueue)
   └── Pods sorted across profiles in activeQ using the shared QueueSort plugin
   └── Dequeue next pod

3. Scheduling Cycle (scheduleOne / scheduleOnePod)
   └── sched.Profiles.FrameworkForPod(pod)
       ├── Check pod.Spec.SchedulerName (default to "default-scheduler" if empty)
       ├── [Found] -> Return framework.Framework instance
       └── [Not Found] -> Return error, mark unschedulable

4. Plugin Pipeline Execution
   └── Execute Framework's PreFilter -> Filter -> PostFilter -> PreScore -> Score -> Reserve -> Permit

5. Binding Cycle Execution (Goroutine)
   └── Execute Framework's PreBind -> Bind -> PostBind
   └── Emit events via profile's isolated EventRecorder
```

---

## 7. Testing Patterns & Guidelines

Unit and integration tests for `pkg/scheduler/profile` and multi-profile workflows follow established patterns:

### 7.1 Unit Testing `NewMap` and `FrameworkForPod` (`profile_test.go`)
- **Table-Driven Test Cases**:
  Use `struct { name string; cfgs []config.KubeSchedulerProfile; wantErr string }` to test valid and invalid configurations.
- **Fake Plugin Registry**:
  Construct minimal fake plugins implementing necessary extension points (`fwk.QueueSortPlugin`, `fwk.BindPlugin`):
  ```go
  var fakeRegistry = frameworkruntime.Registry{
      "QueueSort": newFakePlugin("QueueSort"),
      "Bind1":     newFakePlugin("Bind1"),
      "Bind2":     newFakePlugin("Bind2"),
      "Another":   newFakePlugin("Another"),
  }
  ```
- **Context & Teardown**:
  Use `ktesting.NewTestContext(t)` for contextual logging and ensure `defer m.Close()` is called whenever `m != nil` to test clean teardown.

### 7.2 Multi-Profile Scheduling Tests (`pkg/scheduler`)
When writing tests across `pkg/scheduler` (such as `schedule_one_test.go` or `eventhandlers_test.go`):
- Construct `profile.Map` directly or via `profile.NewMap`:
  ```go
  profiles := profile.Map{
      "default-scheduler": defaultFwk,
      "custom-scheduler":  customFwk,
  }
  ```
- Ensure both frameworks share the same `QueueSortPlugin` name to match production invariants.
- Test fallback behavior when `pod.Spec.SchedulerName` is unset vs explicitly set.

---

## 8. Invariants & Common Pitfalls

> [!IMPORTANT]
> **Strict QueueSort Uniformity**: All profiles in a single scheduler binary MUST share the identical `QueueSort` plugin and arguments. Attempting to configure profile A with `PrioritySort` and profile B with a custom queue sort plugin will fail at startup.

> [!WARNING]
> **Unregistered `schedulerName`**: If a pod arrives with a `schedulerName` that is not registered in `profile.Map`, `FrameworkForPod` will fail. Ensure informer event handlers guard entry with `responsibleForPod` before passing pods to scheduling cycles.

> [!NOTE]
> **Default Scheduler Name**: An empty string `pod.Spec.SchedulerName == ""` is automatically mapped to `v1.DefaultSchedulerName` (`"default-scheduler"`). While the Kubernetes API defaulting webhook normally sets this field, synthetic pods created in tests or scheduling simulations rely on this fallback in `FrameworkForPod`.
