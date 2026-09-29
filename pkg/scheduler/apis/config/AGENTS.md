# Agent Guide: Scheduler Configuration API (`pkg/scheduler/apis/config`)

This guide provides AI agents and human contributors with an in-depth architectural overview, type system walkthrough, lifecycle mechanics, scheme registration, version negotiation, latest defaulting constructor, testing fixtures, and developer invariants for the Kubernetes Scheduler component configuration package located under `pkg/scheduler/apis/config`.

---

## 1. Overview & Purpose

The `pkg/scheduler/apis/config` package defines the internal (unversioned) and external (`v1`) API specifications for configuring the Kubernetes scheduler (`kube-scheduler`).

### Core Responsibilities:
1. **Component Configuration Definition**: Defines `KubeSchedulerConfiguration`, which encapsulates global runtime settings (e.g., parallelism, leader election, client connection, backoff parameters) and multi-profile scheduling configurations.
2. **Scheduling Profiles & Extension Points**: Models multi-profile configuration (`KubeSchedulerProfile`) and plugin enablement/disabling across all scheduling framework extension points via `Plugins` and `PluginSet`.
3. **Plugin Argument Specifications**: Defines configuration schemas and structs (`*Args`) for in-tree framework plugins (e.g., `NodeResourcesFitArgs`, `PodTopologySpreadArgs`, `DynamicResourcesArgs`, `DefaultPreemptionArgs`).
4. **Scheme Registration & Codecs (`scheme/`)**: Provides registration with API machinery runtime schemes (`Scheme`, `Codecs` with `serializer.EnableStrict`), establishing version priority and codecs for strict encoding/decoding.
5. **Latest Defaulting Helper (`latest/`)**: Provides `latest.Default()` to programmatically generate fully defaulted internal configurations from the latest versioned schema.
6. **Testing Helpers & Golden Fixtures (`testing/`, `testing/defaults/`)**: Exposes test converters (`V1ToInternalWithDefaults`) and golden reference fixtures (`PluginsV1`, `ExpandedPluginsV1`, `PluginConfigsV1`) for consistent test setups.
7. **Defaulting, Conversion, & Serialization (`v1/`)**: Versioned defaulting functions, MultiPoint plugin merging, and bidirectional conversion between versioned `v1` and internal structs.
8. **Semantic Validation (`validation/`)**: Implements comprehensive validation rules for scheduler configuration fields, profile integrity, queue sort uniformity, plugin arguments, and feature gate constraints.

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
├── latest/                          # Latest API version helper
│   └── latest.go                    # Helper latest.Default() to construct a defaulted internal configuration
├── scheme/                          # Central runtime.Scheme and serializer codec initialization
│   ├── scheme.go                    # Global runtime.Scheme and Codecs (EnableStrict)
│   └── scheme_test.go               # Roundtrip serialization, strict decoding & defaulting tests
├── v1/                              # Versioned API (v1) implementation
│   ├── conversion.go                # Custom conversion functions (v1 <-> internal) & GetPluginArgConversionScheme()
│   ├── defaults.go                  # Defaulting functions for KubeSchedulerConfiguration and PluginArgs
│   ├── default_plugins.go           # Default plugin lists and MultiPoint merge algorithm
│   ├── default_plugins_test.go      # Tests for default plugin lists and feature gate combinations
│   ├── defaults_test.go             # Tests for v1 config and plugin argument defaulting
│   ├── doc.go                       # Package doc for v1
│   ├── register.go                  # Registration of v1 types with external k8s.io/kube-scheduler/config/v1
│   ├── zz_generated.conversion.go   # Generated conversion code
│   └── zz_generated.defaults.go     # Generated defaulting code
├── validation/                      # Configuration validation rules
│   ├── validation.go                # Validation for KubeSchedulerConfiguration, profiles, extenders, queue sort
│   ├── validation_pluginargs.go     # Semantic validation for plugin argument structures
│   ├── validation_test.go           # Comprehensive test suite for KubeSchedulerConfiguration validation
│   └── validation_pluginargs_test.go# Tests for individual plugin argument validation logic
└── testing/                         # Testing utilities and golden fixtures
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

## 4. Scheme Registration Architecture (`pkg/scheduler/apis/config/scheme`)

The `pkg/scheduler/apis/config/scheme` package constructs the unified `runtime.Scheme` and serializer codec factory used across the scheduler binary, CLI tools, and test suites.

### 4.1 Central Scheme and Codec Factory
```go
var (
    // Scheme is the runtime.Scheme to which all kubescheduler api types are registered.
    Scheme = runtime.NewScheme()

    // Codecs provides access to encoding and decoding for the scheme.
    Codecs = serializer.NewCodecFactory(Scheme, serializer.EnableStrict)
)
```

- **`Scheme`**: An instance of `runtime.Scheme` containing internal types (`pkg/scheduler/apis/config`) and versioned external types (`pkg/scheduler/apis/config/v1`).
- **`Codecs` with `serializer.EnableStrict`**: Strict unmarshaling mode is enabled by default. Decoding operations reject unknown fields or duplicate keys in configuration YAML/JSON to prevent silent configuration errors.

### 4.2 Version Registration & Priority
During package initialization (`init()`), `AddToScheme(Scheme)` registers all known versions and configures version priority:
1. `config.AddToScheme(scheme)`: Registers internal types (`KubeSchedulerConfiguration` and all internal plugin arg types under `__internal`).
2. `configv1.AddToScheme(scheme)`: Registers `v1` types (`kubescheduler.config.k8s.io/v1`).
3. `scheme.SetVersionPriority(configv1.SchemeGroupVersion)`: Establishes `v1` as the highest priority version during version negotiation.

### 4.3 Dual-Scheme Architecture: Top-Level vs PluginArg Schemes
The scheduler config system uses two distinct scheme instances:
1. **Root `scheme.Scheme`**:
   - Handles top-level `KubeSchedulerConfiguration` objects and global fields.
   - Encodes and decodes full configuration payloads.
2. **`pluginArgConversionScheme` (in `v1.GetPluginArgConversionScheme()`)**:
   - A dedicated `runtime.Scheme` initialized lazily via `sync.Once`.
   - Used specifically to decode, default, and convert polymorphic `PluginConfig.Args` embedded inside `runtime.RawExtension` (in external types) or `runtime.Object` (in internal types).
   - Isolates plugin argument type handling from root document parsing.

```
                  ┌────────────────────────────────────────┐
                  │          Input YAML / JSON             │
                  └──────────────────┬─────────────────────┘
                                     │ Codecs.UniversalDecoder()
                                     ▼
                  ┌────────────────────────────────────────┐
                  │    v1.KubeSchedulerConfiguration       │
                  │  ┌──────────────────────────────────┐  │
                  │  │ PluginConfig[i].Args             │  │
                  │  │ (runtime.RawExtension)           │  │
                  │  └──────────────────────────────────┘  │
                  └──────────────────┬─────────────────────┘
                                     │ scheme.Scheme.Default()
                                     │ └── v1.setDefaults_KubeSchedulerProfile()
                                     │     └── pluginArgConversionScheme.Default()
                                     ▼
                  ┌────────────────────────────────────────┐
                  │       Defaulted v1 Configuration       │
                  └──────────────────┬─────────────────────┘
                                     │ scheme.Scheme.Convert()
                                     │ └── convertToInternalPluginConfigArgs()
                                     │     └── pluginArgConversionScheme.ConvertToVersion()
                                     ▼
                  ┌────────────────────────────────────────┐
                  │      config.KubeSchedulerConfiguration │
                  │  ┌──────────────────────────────────┐  │
                  │  │ PluginConfig[i].Args             │  │
                  │  │ (runtime.Object / Typed Struct)  │  │
                  │  └──────────────────────────────────┘  │
                  └────────────────────────────────────────┘
```

---

## 5. Version Negotiation & Conversion Pipeline

The lifecycle of scheduler configuration ingestion proceeds through clear decoding, defaulting, and conversion stages.

### 5.1 In-Tree vs Out-of-Tree Plugin Args Handling
- **In-Tree Plugins**:
  - Plugin argument types (e.g., `NodeResourcesFitArgs`, `InterPodAffinityArgs`, `VolumeBindingArgs`) are registered in both `config` and `config/v1`.
  - When parsed, their typed structs are instantiated, defaulted via `pluginArgConversionScheme.Default(args)`, and converted to internal types.
- **Out-of-Tree Plugins**:
  - If a plugin argument type is not recognized in the scheme, it is preserved as `*runtime.Unknown` or raw JSON/YAML within `runtime.RawExtension`.
  - During conversion (`convertToInternalPluginConfigArgs`), arguments of type `*runtime.Unknown` are skipped, allowing custom out-of-tree plugins to pass through without errors.

### 5.2 Preserving TypeMeta during Conversion
When converting from versioned `v1.KubeSchedulerConfiguration` to internal `config.KubeSchedulerConfiguration`, the Kubernetes API machinery clears `TypeMeta.APIVersion`. The scheduler configuration requires `TypeMeta.APIVersion` to be retained for downstream logging and profile validation. Converters explicitly restore this field:
```go
cfg.TypeMeta.APIVersion = v1.SchemeGroupVersion.String()
```

---

## 6. Default Configuration Generation (`pkg/scheduler/apis/config/latest`)

The `pkg/scheduler/apis/config/latest` package provides the programmatic entry point for generating fully defaulted internal scheduler configurations.

### 6.1 `latest.Default()` Workflow
```go
func Default() (*config.KubeSchedulerConfiguration, error) {
    versionedCfg := v1.KubeSchedulerConfiguration{}
    versionedCfg.DebuggingConfiguration = *v1alpha1.NewRecommendedDebuggingConfiguration()

    scheme.Scheme.Default(&versionedCfg)
    cfg := config.KubeSchedulerConfiguration{}
    if err := scheme.Scheme.Convert(&versionedCfg, &cfg, nil); err != nil {
        return nil, err
    }
    cfg.TypeMeta.APIVersion = v1.SchemeGroupVersion.String()
    return &cfg, nil
}
```

1. **Allocates Versioned Struct**: Instantiates an empty `v1.KubeSchedulerConfiguration`.
2. **Applies Recommended Debugging Config**: Sets default profiling and contention monitoring flags from `component-base`.
3. **Executes Full Defaulting Chain**: Invokes `scheme.Scheme.Default(&versionedCfg)`:
   - Default Leader Election settings (lease duration, renew deadline, retry period, resource lock).
   - Default Client Connection parameters (QPS, burst, content type).
   - Default Profile setup via `setDefaults_KubeSchedulerProfile`:
     - Populates default `MultiPoint` plugins via `getDefaultPlugins()`.
     - Dynamically evaluates feature gates (`NodeDeclaredFeatures`, `GenericWorkload`, `TopologyAwareWorkloadScheduling`, `InPlacePodVerticalScalingSchedulerPreemption`).
     - Merges custom plugin overrides with defaults (`mergePlugins`).
     - Automatically instantiates and defaults missing `PluginConfig` entries for all enabled plugins.
4. **Converts to Internal Representation**: Converts the populated `v1` struct into `config.KubeSchedulerConfiguration`.
5. **Sets APIVersion**: Restores `cfg.TypeMeta.APIVersion = "kubescheduler.config.k8s.io/v1"`.

---

## 7. Testing Helpers & Fixtures (`pkg/scheduler/apis/config/testing`)

To prevent test duplication and ensure test configurations accurately reflect production defaulting and conversion rules, the `testing` and `testing/defaults` packages expose standard fixtures and helpers.

### 7.1 `testing.V1ToInternalWithDefaults`
Located in `pkg/scheduler/apis/config/testing/config.go`:
```go
func V1ToInternalWithDefaults(t *testing.T, versionedCfg v1.KubeSchedulerConfiguration) *config.KubeSchedulerConfiguration
```
- Accepts a partially specified `v1.KubeSchedulerConfiguration` in unit tests.
- Applies recommended debugging defaults, runs the official `scheme.Scheme.Default()` pipeline, converts to internal `config.KubeSchedulerConfiguration`, and fails the test immediately if conversion fails (`t.Fatal(err)`).
- Used extensively in framework unit tests and plugin tests to build valid profile objects.

### 7.2 `testing/defaults` Fixture Constants
Located in `pkg/scheduler/apis/config/testing/defaults/defaults.go`:

| Fixture Variable | Type | Description |
|---|---|---|
| `PluginsV1` | `*config.Plugins` | Default set of plugins configured under the `MultiPoint` extension point prior to MultiPoint expansion. Contains weights for scoring plugins (e.g. `TaintToleration: 3`, `NodeAffinity: 2`, `PodTopologySpread: 2`, `InterPodAffinity: 2`, `DynamicResources: 2`, `NodeResourcesFit: 1`, etc.). |
| `ExpandedPluginsV1` | `*config.Plugins` | The fully expanded plugin mapping across all individual extension points (`PreEnqueue`, `QueueSort`, `PreFilter`, `Filter`, `PostFilter`, `PodGroupPostFilter`, `PreScore`, `Score`, `Reserve`, `PreBind`, `Bind`, `PlacementScore`). |
| `PluginConfigsV1` | `[]config.PluginConfig` | Ground-truth default arguments for all built-in plugins (`DefaultPreemptionArgs`, `DynamicResourcesArgs`, `InterPodAffinityArgs`, `NodeAffinityArgs`, `NodeResourcesBalancedAllocationArgs`, `NodeResourcesFitArgs`, `PodTopologySpreadArgs`, `VolumeBindingArgs`). |

### 7.3 Purpose & Usage in Test Suites
- **Golden Comparison**: Used in `pkg/scheduler/apis/config/scheme/scheme_test.go` and `pkg/scheduler/profile/profile_test.go` to assert that decoded and defaulted configuration profiles match expected plugin lists and argument defaults.
- **Mocking & Isolation**: Enables plugin tests to import standard plugin configs without needing to manually construct deeply nested argument structs.

---

## 8. In-Tree Plugin Arguments (`types_pluginargs.go`)

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

## 9. MultiPoint Plugin Defaulting & Feature Gates

The Kubernetes Scheduler uses a **MultiPoint** configuration paradigm: a plugin enabled under `MultiPoint` is automatically registered to all extension points it implements.

### 9.1 MultiPoint Default Plugin Composition
The baseline set of default plugins in `getDefaultPlugins()` includes:
- `SchedulingGates` (PreEnqueue)
- `PrioritySort` (QueueSort)
- `NodeName`, `NodeUnschedulable`, `TaintToleration`, `NodeAffinity`, `NodePorts`, `NodeResourcesFit`, `VolumeRestrictions`, `NodeVolumeLimits`, `VolumeBinding`, `VolumeZone`, `PodTopologySpread`, `InterPodAffinity`, `DynamicResources` (PreFilter / Filter)
- `DynamicResources`, `DefaultPreemption` (PostFilter / PodGroupPostFilter)
- Scoring plugins with calibrated weights (`TaintToleration: 3`, `NodeAffinity: 2`, `PodTopologySpread: 2`, `InterPodAffinity: 2`, `DynamicResources: 2`, `NodeResourcesFit: 1`, `NodeResourcesBalancedAllocation: 1`, `ImageLocality: 1`)
- `VolumeBinding`, `DynamicResources` (Reserve / PreBind)
- `DefaultBinder` (Bind)

### 9.2 Dynamic Feature Gate Hooks
Default plugins are conditionally augmented during defaulting (`applyFeatureGates` in `pkg/scheduler/apis/config/v1/default_plugins.go`):
- `features.NodeDeclaredFeatures`: Appends `NodeDeclaredFeatures` plugin to MultiPoint.
- `features.GenericWorkload`: Appends `GangScheduling` plugin to MultiPoint.
- `features.TopologyAwareWorkloadScheduling`: Appends `TopologyPlacementGenerator` and `PodGroupPodsCount` (weight 1).
- `features.InPlacePodVerticalScalingSchedulerPreemption`: Adjusts preemption plugins for in-place resizing.

---

## 10. Critical Invariants & Rules for Agents

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
6. **Strict Deserialization**:
   - `scheme.Codecs` enforces `serializer.EnableStrict`. All configuration tests and production decoders reject unknown or misspelled JSON/YAML fields.
7. **No Cyclic Imports**:
   - `pkg/scheduler/apis/config` and its subpackages **must not** import `pkg/scheduler/framework/runtime` or `pkg/scheduler/backend`. To refer to plugin names without cycles, import `pkg/scheduler/framework/plugins/names`.

---

## 11. Maintenance & Developer Workflows

### 11.1 Adding or Modifying In-Tree Plugins
1. **Define Types**:
   - Add internal argument types in `pkg/scheduler/apis/config/types_pluginargs.go`.
   - Add versioned argument types in `staging/src/k8s.io/kube-scheduler/config/v1/types_pluginargs.go`.
2. **Register Types**:
   - Register internal type in `pkg/scheduler/apis/config/register.go` (`addKnownTypes`).
   - Register versioned type in `staging/src/k8s.io/kube-scheduler/config/v1/register.go`.
3. **Implement Defaulting**:
   - Define defaulting rules in `pkg/scheduler/apis/config/v1/defaults.go` (e.g., `SetDefaults_<PluginName>Args`).
4. **Update Testing Fixtures**:
   - Update `PluginsV1`, `ExpandedPluginsV1`, and `PluginConfigsV1` in `pkg/scheduler/apis/config/testing/defaults/defaults.go`.
5. **Run Generators**:
   ```bash
   make update
   ```
   This regenerates `zz_generated.deepcopy.go`, `zz_generated.defaults.go`, and `zz_generated.conversion.go`.
6. **Add Unit & Validation Tests**:
   - Add validation rules in `pkg/scheduler/apis/config/validation/validation_pluginargs.go`.
   - Add roundtrip and decoding tests in `pkg/scheduler/apis/config/scheme/scheme_test.go`.

### 11.2 Bumping Scheduler API Versions
1. Create new version package (e.g. `pkg/scheduler/apis/config/v2`).
2. Update `pkg/scheduler/apis/config/scheme/scheme.go` to add the new version to `AddToScheme` and update `SetVersionPriority`.
3. Update `pkg/scheduler/apis/config/latest/latest.go` to construct defaults using the new version.
4. Update `pkg/scheduler/apis/config/testing/` to provide conversion helpers for the new version.

### 11.3 Running Tests & Verification
```bash
# Run all unit tests across config packages
make test WHAT=./pkg/scheduler/apis/config/... GOFLAGS="-v -race"

# Run scheme and serialization tests specifically
make test WHAT=./pkg/scheduler/apis/config/scheme GOFLAGS="-v"

# Run validation tests
make test WHAT=./pkg/scheduler/apis/config/validation GOFLAGS="-v"

# Verify repository generators
make verify
```
