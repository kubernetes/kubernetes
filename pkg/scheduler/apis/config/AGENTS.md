# Agent Guide: Scheduler Config Helpers & Scheme Registration (`pkg/scheduler/apis/config`)

This guide provides AI agents and human contributors with an in-depth architectural overview of the configuration subsystem in the Kubernetes Scheduler (`kube-scheduler`), focusing on scheme registration, latest version defaulting, test helpers, and default test fixtures.

---

## 1. Overview & Package Map

The `pkg/scheduler/apis/config` directory defines the internal configuration data structures, external versioned API schemas (such as `v1`), scheme registration, defaulting logic, conversion routines, validation rules, and testing utilities for `kube-scheduler`.

```
pkg/scheduler/apis/config/
├── doc.go                           # Package doc & deepcopy generator tags
├── types.go                         # Internal KubeSchedulerConfiguration & profile structs
├── types_pluginargs.go              # Internal plugin argument structs (NodeResourcesFitArgs, etc.)
├── register.go                      # Internal SchemeBuilder & known type registration
├── scheme/                          # Central runtime.Scheme & serializer codec initialization
│   ├── scheme.go                    # Global Scheme & Codecs (EnableStrict)
│   └── scheme_test.go               # Roundtrip serialization, strict decoding & defaulting tests
├── latest/                          # Latest API version helper
│   └── latest.go                    # Default() constructor producing internal config from latest version
├── testing/                         # Testing helper utilities
│   ├── config.go                    # V1ToInternalWithDefaults() conversion helper
│   └── defaults/                    # Golden test fixtures & baseline plugin expectations
│       └── defaults.go              # PluginsV1, ExpandedPluginsV1, PluginConfigsV1
├── v1/                              # Versioned v1 external API configuration
│   ├── register.go                  # v1 GroupVersion registration & defaulting hook
│   ├── defaults.go                  # Defaulting rules for KubeSchedulerConfiguration & profiles
│   ├── default_plugins.go           # getDefaultPlugins() & MultiPoint feature gate integration
│   ├── conversion.go                # Custom conversion functions & GetPluginArgConversionScheme()
│   ├── zz_generated.conversion.go   # Generated conversion code
│   ├── zz_generated.defaults.go     # Generated defaulting code
│   └── doc.go                       # Package doc for v1
└── validation/                      # Configuration validation rules
    ├── validation.go                # KubeSchedulerConfiguration validation
    └── validation_pluginargs.go     # Individual plugin args validation
```

---

## 2. Scheme Registration Architecture (`pkg/scheduler/apis/config/scheme`)

The `pkg/scheduler/apis/config/scheme` package constructs the unified `runtime.Scheme` and serializer codec factory used across the scheduler binary, CLI tools, and test suites.

### 2.1 Central Scheme and Codec Factory
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

### 2.2 Version Registration & Priority
During package initialization (`init()`), `AddToScheme(Scheme)` registers all known versions and configures version priority:
1. `config.AddToScheme(scheme)`: Registers internal types (`KubeSchedulerConfiguration` and all internal plugin arg types under `__internal`).
2. `configv1.AddToScheme(scheme)`: Registers `v1` types (`kubescheduler.config.k8s.io/v1`).
3. `scheme.SetVersionPriority(configv1.SchemeGroupVersion)`: Establishes `v1` as the highest priority version during version negotiation.

### 2.3 Dual-Scheme Architecture: Top-Level vs PluginArg Schemes
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

## 3. Version Negotiation & Conversion Pipeline

The lifecycle of scheduler configuration ingestion proceeds through clear decoding, defaulting, and conversion stages.

### 3.1 In-Tree vs Out-of-Tree Plugin Args Handling
- **In-Tree Plugins**:
  - Plugin argument types (e.g., `NodeResourcesFitArgs`, `InterPodAffinityArgs`, `VolumeBindingArgs`) are registered in both `config` and `config/v1`.
  - When parsed, their typed structs are instantiated, defaulted via `pluginArgConversionScheme.Default(args)`, and converted to internal types.
- **Out-of-Tree Plugins**:
  - If a plugin argument type is not recognized in the scheme, it is preserved as `*runtime.Unknown` or raw JSON/YAML within `runtime.RawExtension`.
  - During conversion (`convertToInternalPluginConfigArgs`), arguments of type `*runtime.Unknown` are skipped, allowing custom out-of-tree plugins to pass through without errors.

### 3.2 Preserving TypeMeta during Conversion
When converting from versioned `v1.KubeSchedulerConfiguration` to internal `config.KubeSchedulerConfiguration`, the Kubernetes API machinery clears `TypeMeta.APIVersion`. The scheduler configuration requires `TypeMeta.APIVersion` to be retained for downstream logging and profile validation. Converters explicitly restore this field:
```go
cfg.TypeMeta.APIVersion = v1.SchemeGroupVersion.String()
```

---

## 4. Default Configuration Generation (`pkg/scheduler/apis/config/latest`)

The `pkg/scheduler/apis/config/latest` package provides the programmatic entry point for generating fully defaulted internal scheduler configurations.

### 4.1 `latest.Default()` Workflow
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

## 5. Testing Helpers & Fixtures (`pkg/scheduler/apis/config/testing`)

To prevent test duplication and ensure test configurations accurately reflect production defaulting and conversion rules, the `testing` and `testing/defaults` packages expose standard fixtures and helpers.

### 5.1 `testing.V1ToInternalWithDefaults`
Located in `pkg/scheduler/apis/config/testing/config.go`:
```go
func V1ToInternalWithDefaults(t *testing.T, versionedCfg v1.KubeSchedulerConfiguration) *config.KubeSchedulerConfiguration
```
- Accepts a partially specified `v1.KubeSchedulerConfiguration` in unit tests.
- Applies recommended debugging defaults, runs the official `scheme.Scheme.Default()` pipeline, converts to internal `config.KubeSchedulerConfiguration`, and fails the test immediately if conversion fails (`t.Fatal(err)`).
- Used extensively in framework unit tests and plugin tests to build valid profile objects.

### 5.2 `testing/defaults` Fixture Constants
Located in `pkg/scheduler/apis/config/testing/defaults/defaults.go`:

| Fixture Variable | Type | Description |
|---|---|---|
| `PluginsV1` | `*config.Plugins` | Default set of plugins configured under the `MultiPoint` extension point prior to MultiPoint expansion. Contains weights for scoring plugins (e.g. `TaintToleration: 3`, `NodeAffinity: 2`, `PodTopologySpread: 2`, `InterPodAffinity: 2`, `DynamicResources: 2`, `NodeResourcesFit: 1`, etc.). |
| `ExpandedPluginsV1` | `*config.Plugins` | The fully expanded plugin mapping across all individual extension points (`PreEnqueue`, `QueueSort`, `PreFilter`, `Filter`, `PostFilter`, `PodGroupPostFilter`, `PreScore`, `Score`, `Reserve`, `PreBind`, `Bind`, `PlacementScore`). |
| `PluginConfigsV1` | `[]config.PluginConfig` | Ground-truth default arguments for all built-in plugins (`DefaultPreemptionArgs`, `DynamicResourcesArgs`, `InterPodAffinityArgs`, `NodeAffinityArgs`, `NodeResourcesBalancedAllocationArgs`, `NodeResourcesFitArgs`, `PodTopologySpreadArgs`, `VolumeBindingArgs`). |

### 5.3 Purpose & Usage in Test Suites
- **Golden Comparison**: Used in `pkg/scheduler/apis/config/scheme/scheme_test.go` and `pkg/scheduler/profile/profile_test.go` to assert that decoded and defaulted configuration profiles match expected plugin lists and argument defaults.
- **Mocking & Isolation**: Enables plugin tests to import standard plugin configs without needing to manually construct deeply nested argument structs.

---

## 6. MultiPoint Plugin Defaulting & Feature Gates

The Kubernetes Scheduler uses a **MultiPoint** configuration paradigm: a plugin enabled under `MultiPoint` is automatically registered to all extension points it implements.

### 6.1 MultiPoint Default Plugin Composition
The baseline set of default plugins in `getDefaultPlugins()` includes:
- `SchedulingGates` (PreEnqueue)
- `PrioritySort` (QueueSort)
- `NodeName`, `NodeUnschedulable`, `TaintToleration`, `NodeAffinity`, `NodePorts`, `NodeResourcesFit`, `VolumeRestrictions`, `NodeVolumeLimits`, `VolumeBinding`, `VolumeZone`, `PodTopologySpread`, `InterPodAffinity`, `DynamicResources` (PreFilter / Filter)
- `DynamicResources`, `DefaultPreemption` (PostFilter / PodGroupPostFilter)
- Scoring plugins with calibrated weights (`TaintToleration: 3`, `NodeAffinity: 2`, `PodTopologySpread: 2`, `InterPodAffinity: 2`, `DynamicResources: 2`, `NodeResourcesFit: 1`, `NodeResourcesBalancedAllocation: 1`, `ImageLocality: 1`)
- `VolumeBinding`, `DynamicResources` (Reserve / PreBind)
- `DefaultBinder` (Bind)

### 6.2 Dynamic Feature Gate Hooks
Default plugins are conditionally augmented during defaulting (`applyFeatureGates` in `pkg/scheduler/apis/config/v1/default_plugins.go`):
- `features.NodeDeclaredFeatures`: Appends `NodeDeclaredFeatures` plugin to MultiPoint.
- `features.GenericWorkload`: Appends `GangScheduling` plugin to MultiPoint.
- `features.TopologyAwareWorkloadScheduling`: Appends `TopologyPlacementGenerator` and `PodGroupPodsCount` (weight 1).
- `features.InPlacePodVerticalScalingSchedulerPreemption`: Adjusts preemption plugins for in-place resizing.

---

## 7. Maintenance & Invariants for Agents

When making changes to scheduler configuration, adding new plugins, or bumping API versions, follow these invariants:

### 7.1 Adding or Modifying In-Tree Plugins
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

### 7.2 Bumping Scheduler API Versions
1. Create new version package (e.g. `pkg/scheduler/apis/config/v2`).
2. Update `pkg/scheduler/apis/config/scheme/scheme.go` to add the new version to `AddToScheme` and update `SetVersionPriority`.
3. Update `pkg/scheduler/apis/config/latest/latest.go` to construct defaults using the new version.
4. Update `pkg/scheduler/apis/config/testing/` to provide conversion helpers for the new version.

---

## 8. Development & Verification Workflows

### Run Unit Tests
```bash
# Run unit tests across all config packages
make test WHAT=./pkg/scheduler/apis/config/... GOFLAGS="-v -race"

# Run scheme and serialization tests specifically
make test WHAT=./pkg/scheduler/apis/config/scheme GOFLAGS="-v"

# Run validation tests
make test WHAT=./pkg/scheduler/apis/config/validation GOFLAGS="-v"
```

### Run Code Generators and Linters
```bash
# Update generated deepcopy, conversion, and defaulting code
make update

# Run repository verifications
make verify
```
