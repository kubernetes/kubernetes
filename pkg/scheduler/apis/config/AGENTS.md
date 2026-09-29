# Agent Guide: Scheduler Configuration API (`pkg/scheduler/apis/config`)

This guide provides AI agents and human contributors with an architectural overview, type system walkthrough, lifecycle mechanics, validation invariants, and testing guide for the Kubernetes Scheduler component configuration package located under `pkg/scheduler/apis/config`.

---

## 1. Overview & Purpose

The `pkg/scheduler/apis/config` package defines the internal (unversioned) and external (`v1`) API specifications for configuring the Kubernetes scheduler (`kube-scheduler`).

### Core Responsibilities:
1. **Component Configuration Definition**: Defines `KubeSchedulerConfiguration`, which encapsulates global runtime settings (e.g., parallelism, leader election, client connection, backoff parameters) and multi-profile scheduling configurations.
2. **Scheduling Profiles & Extension Points**: Models multi-profile configuration (`KubeSchedulerProfile`) and plugin enablement/disabling across all scheduling framework extension points via `Plugins` and `PluginSet`.
3. **Plugin Argument Specifications**: Defines configuration schemas and structs (`*Args`) for in-tree framework plugins (e.g., `NodeResourcesFitArgs`, `PodTopologySpreadArgs`, `DynamicResourcesArgs`, `DefaultPreemptionArgs`).
4. **Defaulting, Conversion, & Serialization**: Provides registration with API machinery runtime schemes (`Scheme`, `Codecs`), versioned defaulting functions (`v1`), and conversion logic between versioned `v1` and internal structures.
5. **Semantic Validation**: Implements comprehensive validation rules (`pkg/scheduler/apis/config/validation`) for scheduler configuration fields, profile integrity, queue sort consistency, plugin arguments, and feature gate constraints.

---

## 2. Directory & Subpackage Structure

```
pkg/scheduler/apis/config/
├── doc.go                           # Package doc and deepcopy-gen annotations (+groupName=kubescheduler.config.k8s.io)
├── register.go                      # Internal API GroupVersion (runtime.APIVersionInternal) & SchemeBuilder
├── types.go                         # Top-level internal config types (KubeSchedulerConfiguration, KubeSchedulerProfile, Plugins, Extender)
├── types_pluginargs.go              # Internal plugin argument structs (NodeResourcesFitArgs, DynamicResourcesArgs, etc.)
├── types_test.go                    # Unit tests for internal type methods (e.g., Plugins.Names())
├── zz_generated.deepcopy.go         # Auto-generated deepcopy functions for internal types
├── latest/
│   └── latest.go                    # Helper latest.Default() to construct a defaulted internal configuration
├── scheme/
│   ├── scheme.go                    # Global runtime.Scheme and Codecs registering internal and versioned types
│   └── scheme_test.go               # Roundtrip serialization/deserialization tests
├── v1/                              # Versioned API (v1) implementation
│   ├── conversion.go                # Custom conversion functions (v1 <-> internal) & plugin args scheme
│   ├── defaults.go                  # Defaulting functions for KubeSchedulerConfiguration and PluginArgs
│   ├── default_plugins.go           # Default plugin lists and MultiPoint merge algorithm
│   ├── default_plugins_test.go      # Tests for default plugin lists and feature gate combinations
│   ├── defaults_test.go             # Tests for v1 config and plugin argument defaulting
│   ├── doc.go                       # Package doc for v1
│   ├── register.go                  # Registration of v1 types with external k8s.io/kube-scheduler/config/v1
│   ├── zz_generated.conversion.go   # Generated conversion code
│   └── zz_generated.defaults.go     # Generated defaulting code
├── validation/
│   ├── validation.go                # Validation for KubeSchedulerConfiguration, profiles, extenders, queue sort
│   ├── validation_pluginargs.go     # Semantic validation for plugin argument structures
│   ├── validation_test.go           # Comprehensive test suite for KubeSchedulerConfiguration validation
│   └── validation_pluginargs_test.go# Tests for individual plugin argument validation logic
└── testing/
    ├── config.go                    # Test helper V1ToInternalWithDefaults()
    └── defaults/
        └── defaults.go              # Golden reference plugin sets (PluginsV1, ExpandedPluginsV1, PluginConfigsV1)
```

---

## 3. Core Configuration Types

### 3.1 `KubeSchedulerConfiguration` (`types.go`)
The top-level configuration object:
- **`TypeMeta`**: Contains `Kind` and `APIVersion`. When converted from a versioned configuration (e.g., `v1`), `APIVersion` retains the source version string (`kubescheduler.config.k8s.io/v1`). This is used during validation to check for version-specific deprecated or removed plugins.
- **`Parallelism`**: Number of parallel goroutines for scheduling algorithms (filtering/scoring). Defaults to `16` (must be `> 0`).
- **`LeaderElection`**: Component leader election configuration (`componentbaseconfig.LeaderElectionConfiguration`). In Kubernetes 1.20+, `ResourceLock` defaults to `"leases"`.
- **`ClientConnection`**: API server client connection configuration (QPS, Burst, kubeconfig path). Defaults: QPS `50`, Burst `100`, ContentType `protobuf`.
- **`PercentageOfNodesToScore`**: Global percentage (`0`-`100`) of nodes to check before stopping search early. `0` means adaptive percentage (5%-50% based on cluster size).
- **`PodInitialBackoffSeconds` & `PodMaxBackoffSeconds`**: Exponential backoff duration for unschedulable pods. Defaults: Initial `1s`, Max `10s`.
- **`Profiles`**: Slice of `KubeSchedulerProfile` instances. Pods match profiles by `pod.spec.schedulerName`.
- **`Extenders`**: Slice of `Extender` configurations for out-of-tree HTTP scheduling extenders.
- **`DelayCacheUntilActive`**: If `true`, defers filling informer caches until after winning leader election to reduce idle memory usage.

### 3.2 `KubeSchedulerProfile` (`types.go`)
Configures a single scheduler profile:
- **`SchedulerName`**: Name of the profile. A pod whose `spec.schedulerName` matches this string is scheduled with this profile. If only one profile is defined and name is omitted, it defaults to `"default-scheduler"`.
- **`PercentageOfNodesToScore`**: Profile-level override for node scoring percentage.
- **`Plugins`**: Set of enabled/disabled plugins for each extension point.
- **`PluginConfig`**: Slice of custom `PluginConfig` arguments for specific plugins.

### 3.3 Extension Points & `Plugins` (`types.go`)
The `Plugins` struct defines plugin registration across all framework extension points:

| Extension Point | Description |
|---|---|
| **`MultiPoint`** | High-level configuration field allowing a plugin to be enabled across all extension points it implements. |
| **`PreEnqueue`** | Invoked prior to enqueueing pods into the active scheduling queue. |
| **`QueueSort`** | Orders pods in the active scheduling queue. Exactly one QueueSort plugin can be active across all profiles. |
| **`PreFilter`** | Pre-computes pod/cluster state and verifies prerequisite invariants before filtering nodes. |
| **`Filter`** | Evaluates node feasibility in parallel. |
| **`PostFilter`** | Invoked when no feasible nodes are found (e.g. `DefaultPreemption`, `DynamicResources`). |
| **`PreScore`** | Pre-computes scoring state before scoring nodes. |
| **`Score`** | Assigns weighted scores (0–100) to candidate nodes. |
| **`Reserve`** | In-memory reservation of resources before binding; pairs with `Unreserve`. |
| **`Permit`** | Holds or delays pod binding (e.g. gang/coscheduling). |
| **`PreBind`** | Prepares dependencies (e.g., dynamic resource allocations, volume attachments) before binding. |
| **`Bind`** | Issues the node assignment binding to the API server. |
| **`PostBind`** | Executed after successful binding for metrics and post-commit actions. |
| **`PlacementGenerate`** | Pod group scheduling cycle: generates candidate placements for a group of pods. |
| **`PlacementScore`** | Pod group scheduling cycle: ranks candidate pod group placements. |
| **`PlacementFeasible`** | Pod group scheduling cycle: verifies feasibility of a group placement. |
| **`PodGroupPostFilter`** | Invoked when a PodGroup cannot be scheduled (equivalent to PostFilter for workloads). |

#### `PluginSet` & `Plugin` Mechanics:
- `PluginSet.Enabled`: List of `Plugin{Name, Weight}` called in order. `Weight` applies to `Score` and `PlacementScore`.
- `PluginSet.Disabled`: List of plugins to disable from defaults. Disabling with `Name: "*"` disables all default plugins for that extension point.

---

## 4. In-Tree Plugin Arguments (`types_pluginargs.go`)

Each plugin with configurable parameters defines a corresponding typed struct registered in the scheme:

| Plugin Name | Configuration Struct | Key Fields & Semantics |
|---|---|---|
| **DefaultPreemption** | `DefaultPreemptionArgs` | `MinCandidateNodesPercentage` (default `10%`), `MinCandidateNodesAbsolute` (default `100`). Both cannot be zero simultaneously. |
| **InterPodAffinity** | `InterPodAffinityArgs` | `HardPodAffinityWeight` (range `[0, 100]`, default `1`), `IgnorePreferredTermsOfExistingPods`. |
| **NodeResourcesFit** | `NodeResourcesFitArgs` | `IgnoredResources`, `IgnoredResourceGroups`, `ScoringStrategy` (`LeastAllocated`, `MostAllocated`, or `RequestedToCapacityRatio` with custom `Shape`). |
| **PodTopologySpread** | `PodTopologySpreadArgs` | `DefaultingType` (`"System"` or `"List"`), `DefaultConstraints []v1.TopologySpreadConstraint` (must not set `LabelSelector`). |
| **NodeResourcesBalancedAllocation** | `NodeResourcesBalancedAllocationArgs` | `Resources []ResourceSpec` (names e.g. `cpu`, `memory`; weights must equal `1`). |
| **VolumeBinding** | `VolumeBindingArgs` | `BindTimeoutSeconds` (default `600s`), `Shape []UtilizationShapePoint` (custom scoring curve when `StorageCapacityScoring` feature gate is enabled). |
| **NodeAffinity** | `NodeAffinityArgs` | `AddedAffinity *v1.NodeAffinity` (cluster-wide additional required or preferred node affinity terms). |
| **DynamicResources** | `DynamicResourcesArgs` | `FilterTimeout` (default `10s`), `BindingTimeout` (default `10m`). Controlled by `DRASchedulerFilterTimeout` and `DRADeviceBindingConditions` feature gates. |

---

## 5. Configuration Lifecycle: Loading, Defaulting, Conversion, & Validation

```
[ Raw YAML / JSON / CLI Flags ]
              │
              ▼
[ v1.KubeSchedulerConfiguration (External Versioned) ]
              │
              ▼  (1) Scheme Defaulting (`v1/defaults.go` + `v1/default_plugins.go`)
              │      - Populate default settings (Parallelism, Leases, Backoff)
              │      - Default plugin sets (MultiPoint enabled list + feature gates)
              │      - Default plugin configs via `GetPluginArgConversionScheme()`
              │
              ▼  (2) Conversion (`v1/conversion.go`)
              │      - Auto-convert top-level structures
              │      - Decode `runtime.RawExtension` into typed internal `runtime.Object` args
              │      - Preserve `TypeMeta.APIVersion = "kubescheduler.config.k8s.io/v1"`
              │
              ▼
[ config.KubeSchedulerConfiguration (Internal Unversioned) ]
              │
              ▼  (3) Validation (`validation/validation.go` + `validation_pluginargs.go`)
              │      - ClientConnection & LeaderElection validation
              │      - QueueSort uniformity check across all profiles
              │      - Disallow removed/deprecated plugins for the source APIVersion
              │      - Type-safe validation of each PluginConfig's Args
              │      - Extender uniqueness (at most 1 binder) and resource naming
              │
              ▼
[ Scheduler Framework Instantiation (`pkg/scheduler/framework/runtime`) ]
```

### Dedicated Plugin Arguments Scheme (`v1/conversion.go`):
Because `PluginConfig.Args` is untyped `runtime.RawExtension` in external `v1` and typed `runtime.Object` in internal `config`, `v1.GetPluginArgConversionScheme()` initializes an isolated `runtime.Scheme` to convert and default plugin argument objects without polluting the global Kubernetes API scheme.

---

## 6. Critical Invariants & Rules for Agents

1. **QueueSort Uniformity**:
   - Every profile in `Profiles` **MUST** use the exact same `QueueSort` plugin and identical `PluginConfig.Args`. The scheduling queue is shared across all profiles; varying sort algorithms would break heap ordering invariants.
2. **Extender Binder Exclusivity**:
   - At most **one** extender in `Extenders` can define `BindVerb`. Multiple binders cause conflicting binding ownership and validation failure.
3. **APIVersion Preservation in `TypeMeta`**:
   - Internal `KubeSchedulerConfiguration.TypeMeta.APIVersion` must record the version string (e.g. `kubescheduler.config.k8s.io/v1`) from which it was converted. Validation checks `invalidPluginsByVersion` against this version to reject removed plugins.
4. **MultiPoint Expansion & Merge Rules**:
   - In `v1`, default plugins are specified under `MultiPoint`. When merging custom profile configurations, plugins disabled in an extension point or disabled globally (`*`) are respected, and explicit custom plugin order is preserved after defaults.
5. **Feature-Gated Defaults & Args**:
   - Several plugin arguments and default plugins depend on active feature gates (`features.StorageCapacityScoring`, `features.DynamicResourceAllocation`, `features.NodeDeclaredFeatures`, `features.GenericWorkload`, `features.TopologyAwareWorkloadScheduling`).
   - If a feature gate is disabled, specifying its corresponding plugin argument fields during validation triggers an error rather than silent omission.
6. **No Cyclic Imports**:
   - `pkg/scheduler/apis/config` and its subpackages **must not** import `pkg/scheduler/framework/runtime` or `pkg/scheduler/backend`. To refer to plugin names without cycles, import `pkg/scheduler/framework/plugins/names`.

---

## 7. Testing Patterns

### 7.1 Running Tests
```bash
# Run all unit tests for the config package and subpackages
make test WHAT=./pkg/scheduler/apis/config/... GOFLAGS="-v -race"

# Run validation test suites specifically
make test WHAT=./pkg/scheduler/apis/config/validation GOFLAGS="-v"

# Run defaulting and plugin merge tests
make test WHAT=./pkg/scheduler/apis/config/v1 GOFLAGS="-v"
```

### 7.2 Testing Utilities & Fixtures
- **`testing.V1ToInternalWithDefaults(t, versionedCfg)`**: Convenience helper in `pkg/scheduler/apis/config/testing` to construct a fully defaulted and converted internal configuration for test cases.
- **`defaults.PluginsV1` & `defaults.ExpandedPluginsV1`**: Reference plugin sets in `pkg/scheduler/apis/config/testing/defaults` representing the default unexpanded and expanded plugin sets.
- **Table-Driven Tests**: Follow established patterns in `validation_test.go` and `validation_pluginargs_test.go` using `field.ErrorList` and `cmp.Diff` for comparing configuration errors.

### 7.3 Code Generation & Verification
When adding new fields or types to `pkg/scheduler/apis/config`:
```bash
# Verify deepcopy, conversions, and openapi generators
make verify

# Regenerate zz_generated files if API types or conversions change
make update
```
