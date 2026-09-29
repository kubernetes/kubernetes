# Plugin Names Registry (`pkg/scheduler/framework/plugins/names`)

This guide provides an architectural overview, constant catalog, subsystem integration, and configuration binding contracts for the `names` package in `pkg/scheduler/framework/plugins/names`.

---

## 1. High-Level Purpose & Scope

The `names` package serves as the single source of truth for canonical plugin name identifiers in the Kubernetes scheduler framework. It eliminates string duplication and prevents typos across scheduler component registration, configuration parsing, cycle state storage, logging, and metrics instrumentation.

### Core Responsibilities:
1. **Canonical Plugin Identifiers**: Defines typed constants for all in-tree scheduler framework plugins.
2. **Configuration Profile Mapping**: Provides the exact string keys matched against `KubeSchedulerProfile` plugin configurations in `KubeSchedulerConfiguration`.
3. **Subsystem Synchronization**: Synchronizes plugin names across plugin registration (`pkg/scheduler/framework/plugins/registry.go`), metrics collectors, and logging contexts.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/names/
├── names.go       # Plugin name string constants
└── AGENTS.md      # This agent documentation
```

---

## 3. Registered Plugin Name Catalog

| Constant Name | String Value | Primary Extension Points | Description |
| :--- | :--- | :--- | :--- |
| `PrioritySort` | `"PrioritySort"` | `QueueSort` | Active queue priority & arrival heap ordering. |
| `DefaultBinder` | `"DefaultBinder"` | `Bind` | Commits Pod-to-Node `v1.Binding` API requests. |
| `DefaultPreemption` | `"DefaultPreemption"` | `PostFilter`, `PreEnqueue` | Evaluates candidate nodes and selects victim pods for eviction. |
| `DynamicResources` | `"DynamicResources"` | `PreFilter`, `Filter`, `PreScore`, `Score`, `Reserve` | Dynamic Resource Allocation (DRA) claim lifecycle and device matching. |
| `GangScheduling` | `"GangScheduling"` | `PreFilter`, `Filter`, `Permit`, `Reserve` | All-or-nothing co-scheduling for pod groups. |
| `ImageLocality` | `"ImageLocality"` | `Score`, `SignPlugin` | Node prioritization based on container image caching. |
| `InterPodAffinity` | `"InterPodAffinity"` | `PreFilter`, `Filter`, `PreScore`, `Score` | Inter-pod affinity and anti-affinity placement rules. |
| `NodeAffinity` | `"NodeAffinity"` | `PreFilter`, `Filter`, `PreScore`, `Score` | Node selector terms and preferred node affinities. |
| `NodeDeclaredFeatures` | `"NodeDeclaredFeatures"`| `PreFilter`, `Filter` | Node feature capability matching and DRA integration. |
| `NodeName` | `"NodeName"` | `PreFilter`, `Filter`, `SignPlugin` | Deterministic node targeting via `pod.Spec.NodeName`. |
| `NodePorts` | `"NodePorts"` | `PreFilter`, `Filter`, `SignPlugin` | Host network container port conflict checking. |
| `NodeResourcesBalancedAllocation` | `"NodeResourcesBalancedAllocation"` | `Score` | Resource usage variance and balance optimization. |
| `NodeResourcesFit` | `"NodeResourcesFit"` | `PreFilter`, `Filter`, `PreScore`, `Score` | CPU, memory, storage, hugepages, and scalar capacity filtering/scoring. |
| `NodeUnschedulable` | `"NodeUnschedulable"` | `PreFilter`, `Filter`, `SignPlugin` | Cordoned node (`node.Spec.Unschedulable`) filtering and taint bypass. |
| `NodeVolumeLimits` | `"NodeVolumeLimits"` | `PreFilter`, `Filter` | CSI and cloud-provider attached volume limit enforcement. |
| `PodTopologySpread` | `"PodTopologySpread"` | `PreFilter`, `Filter`, `PreScore`, `Score` | Even pod distribution across failure domains / topology zones. |
| `DeferredPodScheduling`| `"DeferredPodScheduling"` | `PreFilter`, `Filter`, `Reserve` | In-place pod resource resizing and deferred scheduling flows. |
| `SchedulingGates` | `"SchedulingGates"` | `PreEnqueue`, `EnqueueExtensions` | Blocks pods carrying active `spec.schedulingGates`. |
| `TaintToleration` | `"TaintToleration"` | `Filter`, `PreScore`, `Score` | Node taint filtering and preference scoring. |
| `VolumeBinding` | `"VolumeBinding"` | `PreFilter`, `Filter`, `Reserve`, `PreBind` | PV/PVC dynamic provisioning and node volume topology binding. |
| `VolumeRestrictions` | `"VolumeRestrictions"` | `Filter` | Conflicting volume mount checks (e.g. read-write once conflicts). |
| `VolumeZone` | `"VolumeZone"` | `Filter` | Storage zone topology constraint validation. |
| `TopologyPlacementGenerator` | `"TopologyPlacementGenerator"` | `PlacementGenerator` | Topo-aware workload placement topologies. |
| `PodGroupPodsCount` | `"PodGroupPodsCount"` | `PreFilter`, `Filter` | Workload group minimum/maximum member count checks. |

---

## 4. Architectural Invariants & Usage Contracts

1. **Registry Association**: In `pkg/scheduler/framework/plugins/registry.go`, `NewInTreeRegistry()` maps each `names.<PluginConstant>` directly to its constructor factory `frameworkruntime.PluginFactory`.
2. **Immutable Constants**: All entries are untyped string constants, preventing unintended package mutations.
3. **No Cross-Package Cycles**: The `names` package has zero internal Kubernetes dependencies, allowing any scheduler package or test suite to import it without introducing cyclic dependencies.
