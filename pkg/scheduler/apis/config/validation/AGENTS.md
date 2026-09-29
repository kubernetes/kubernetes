# AGENTS.md: Developer & Agent Guide for `pkg/scheduler/apis/config/validation`

This guide provides an architectural overview, validation rules, structural invariants, error construction patterns, and test strategies for the Kubernetes scheduler configuration validation package under `pkg/scheduler/apis/config/validation`.

---

## 1. High-Level Purpose & Scope

The `pkg/scheduler/apis/config/validation` package performs static schema and semantic validation of the internal scheduler configuration structs (`config.KubeSchedulerConfiguration`, `config.KubeSchedulerProfile`, plugin configurations, plugin args, and extenders).

### Core Responsibilities:
1. **Structural & Bounds Checking**: Validates configuration ranges, mandatory fields, mutually exclusive flags, and valid enumeration values.
2. **Multi-Profile Uniformity Invariants**: Enforces invariants that span multiple scheduler profiles (e.g., uniqueness of profile names, common `QueueSort` plugin configuration across all profiles).
3. **Plugin Args Reflection Dispatch & Semantic Validation**: Dispatches strongly-typed plugin args to dedicated sub-validators using reflection, validating plugin-specific constraint rules.
4. **Feature Gate & Version Gating**: Enforces feature-gate-dependent constraints (e.g., DRA timeouts, storage capacity scoring) and checks for removed/deprecated plugins across component config API versions.
5. **Standardized Error Path Construction**: Constructs structured `field.ErrorList` hierarchies with precise field path pointers, flattening them into `utilerrors.Aggregate` for consumption by the scheduler startup pipeline.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/apis/config/validation/
├── validation.go                 # Top-level config, profile, plugin set, queue sort, and extender validation
├── validation_pluginargs.go      # Plugin-specific argument validators (Preemption, Affinity, Topology, Resources, etc.)
├── validation_test.go            # Comprehensive table-driven tests for KubeSchedulerConfiguration & profiles
├── validation_pluginargs_test.go # Fine-grained table-driven tests for individual plugin arguments & feature gates
└── AGENTS.md                     # This agent documentation
```

---

## 3. Validation Rules & Invariants

### 3.1. Top-Level Scheduler Configuration (`ValidateKubeSchedulerConfiguration`)

| Field | Invariant / Validation Rule | Error Constructor |
|---|---|---|
| `clientConnection` | Validated via `componentbasevalidation.ValidateClientConnectionConfiguration`. | `field.Error` via component-base |
| `leaderElection` | Validated via `componentbasevalidation.ValidateLeaderElectionConfiguration`. | `field.Error` via component-base |
| `leaderElection.resourceLock` | When `leaderElection.leaderElect == true`, `resourceLock` must be `"leases"`. | `field.Invalid` |
| `parallelism` | Must be an integer greater than 0 (`cc.Parallelism > 0`). | `field.Invalid` |
| `profiles` | Must contain at least one profile (`len(cc.Profiles) > 0`). | `field.Required` |
| `profiles[i].schedulerName` | Must be non-empty and unique across all profiles in the configuration. | `field.Required`, `field.Duplicate` |
| `percentageOfNodesToScore` | If non-nil, must be in the closed interval `[0, 100]`. | `field.Invalid` |
| `podInitialBackoffSeconds` | Must be strictly greater than 0 (`> 0`). | `field.Invalid` |
| `podMaxBackoffSeconds` | Must be greater than or equal to `podInitialBackoffSeconds`. | `field.Invalid` |
| `extenders` | Validated for positive prioritization weights, single bind extender, and valid extended resource names. | `field.Invalid` |

### 3.2. Multi-Profile Invariants (`validateCommonQueueSort`)

The scheduler uses a single scheduling queue across all profiles. Consequently, **all profiles must configure the exact same `QueueSort` plugin and arguments**:
- **Single QueueSort Plugin**: Exactly one `QueueSort` plugin can be enabled in a profile (`len(QueueSort.Enabled) == 1`).
- **Profile Parity**: `QueueSort.Enabled` plugins must match identically across all profiles (checked with `apiequality.Semantic.DeepEqual`).
- **Args Parity**: The `pluginConfig` args for the configured `QueueSort` plugin must be semantically identical across all profiles.

### 3.3. Deprecated & Removed Plugin Gating (`isPluginInvalid`)

The package maintains a versioned deprecation list `invalidPluginsByVersion`:
- Deprecated/removed plugins (such as `AzureDiskLimits`, `CinderLimits`, `EBSLimits`, `GCEPDLimits`) are disallowed if configured in API versions where they have been removed.
- Validated across all plugin extension points (PreEnqueue, QueueSort, PreFilter, Filter, PostFilter, PreScore, Score, Reserve, Permit, PreBind, Bind, PostBind, PlacementGenerate, PlacementScore, PlacementFeasible, PodGroupPostFilter) and `PluginConfig` lists.

### 3.4. Plugin Configuration Dispatch (`validatePluginConfig`)

For each `PluginConfig` entry:
1. **Duplicate Detection**: Plugin names within a profile's `PluginConfig` must be unique (`field.Duplicate`).
2. **Dynamic Dispatch**: Plugin arguments are matched against a typed dispatch map (`m[name]`).
3. **Type Matching**: The concrete Go type of `args` must match the expected argument struct type via reflection (`reflect.TypeOf(args) == reflect.ValueOf(validateFunc).Type().In(1)`). If mismatched, a `field.Invalid` error is returned.

---

## 4. Plugin Arguments Validation Rules (`validation_pluginargs.go`)

### 4.1. `DefaultPreemptionArgs`
- `minCandidateNodesPercentage`: Must be in `[0, 100]`.
- `minCandidateNodesAbsolute`: Must be `>= 0`.
- **Non-Zero Invariant**: `minCandidateNodesPercentage` and `minCandidateNodesAbsolute` **cannot both be zero** simultaneously.

### 4.2. `InterPodAffinityArgs`
- `hardPodAffinityWeight`: Must be in range `[0, 100]`.

### 4.3. `PodTopologySpreadArgs`
- `defaultingType`: Must be `SystemDefaulting` or `ListDefaulting`.
- If `defaultingType == SystemDefaulting`, `defaultConstraints` must be empty.
- For each `defaultConstraints[i]`:
  - `maxSkew`: Must be strictly `> 0`.
  - `topologyKey`: Must be a valid non-empty Kubernetes label name (`metav1validation.ValidateLabelName`).
  - `whenUnsatisfiable`: Must be `DoNotSchedule` or `ScheduleAnyway`.
  - `labelSelector`: Must be `nil` (forbidden, as selectors are deduced dynamically per pod).
  - Constraints must not repeat the same `(topologyKey, whenUnsatisfiable)` pair within the list.

### 4.4. `NodeResourcesBalancedAllocationArgs`
- `resources`: Resource names must not be duplicated.
- Resource `weight`: Must be strictly `1`.

### 4.5. `NodeAffinityArgs`
- `addedAffinity`: Validates syntax of `RequiredDuringSchedulingIgnoredDuringExecution` via `nodeaffinity.NewNodeSelector` and `PreferredDuringSchedulingIgnoredDuringExecution` via `nodeaffinity.NewPreferredSchedulingTerms`.

### 4.6. `VolumeBindingArgs`
- `bindTimeoutSeconds`: Must be `>= 0`.
- `shape` (Storage Capacity Scoring):
  - Forbidden if feature gate `StorageCapacityScoring` is disabled.
  - When enabled, `shape` must contain at least one point, points must be sorted in strictly increasing utilization order, utilization must be in `[0, 100]`, and score must be in `[0, 10]` (`MaxCustomPriorityScore`).

### 4.7. `NodeResourcesFitArgs`
- `ignoredResources`: Must be valid label names.
- `ignoredResourceGroups`: Cannot contain `/` and must be valid label names.
- `scoringStrategy`: Required. Type must be one of `LeastAllocated`, `MostAllocated`, `RequestedToCapacityRatio`.
- Resource weights: Must be in range `(0, 100]`.
- If `RequestedToCapacityRatio`, `requestedToCapacityRatio.shape` is required and must satisfy `validateFunctionShape`.

### 4.8. `DynamicResourcesArgs`
- Feature-gated timeouts:
  - `filterTimeout`: Requires `DRASchedulingFilterTimeout` feature gate. When enabled, must be `>= 0`. When disabled, field must be nil.
  - `bindingTimeout`: Requires both `DRADeviceBindingConditions` and `DRAResourceClaimDeviceStatus` feature gates. When enabled, duration must be `>= 1s`. When disabled, field must be nil.

---

## 5. Error List Construction & Aggregation

Validation in this package follows the Kubernetes API convention of collecting all errors without fail-fast behavior:

```
                  ┌──────────────────────┐
                  │   *field.Path root   │
                  └──────────┬───────────┘
                             │
     ┌───────────────────────┼────────────────────────┐
     ▼                       ▼                        ▼
field.Child("profiles")  field.Child("parallelism")  field.Child("extenders")
     │
field.Index(i)
     │
field.Child("pluginConfig").Index(j).Child("args")
```

### Key Error Patterns:
- **`field.ErrorList`**: Accumulated slice of `*field.Error` pointers.
- **Conversion to Aggregate**: `allErrs.ToAggregate()` converts `field.ErrorList` into a standard Go `error` / `utilerrors.Aggregate`.
- **`utilerrors.Flatten()`**: Recursively unwraps nested aggregates into a single flat aggregate list at the top level in `ValidateKubeSchedulerConfiguration`.

---

## 6. Testing Guide & Coverage

Tests are organized into two primary suites:

### 6.1. `validation_test.go`
- **`TestValidateKubeSchedulerConfigurationV1`**: Table-driven scenario matrix testing positive base cases and individual failure cases across:
  - Parallelism values
  - Leader election configurations (leases requirement)
  - Missing or duplicate scheduler names
  - Profile percentage bounds
  - QueueSort mismatches across profiles
  - Extender weight, binding limit, and duplicate managed resources
  - Invalid and deprecated plugin sets

### 6.2. `validation_pluginargs_test.go`
- Table-driven unit tests for each plugin argument validator:
  - `TestValidateDefaultPreemptionArgs`
  - `TestValidateInterPodAffinityArgs`
  - `TestValidatePodTopologySpreadArgs`
  - `TestValidateNodeResourcesBalancedAllocationArgs`
  - `TestValidateNodeAffinityArgs`
  - `TestValidateVolumeBindingArgs`
  - `TestValidateNodeResourcesFitArgs` (LeastAllocated, MostAllocated, RequestedToCapacityRatio)
  - `TestValidateDynamicResourcesArgs`
- Uses `featuregatetesting.SetFeatureGateDuringTest` for testing gated configuration fields.
- Uses `cmp.Diff` with `ignoreBadValueDetail` for exact field path and error type assertion.

### Running Tests:
```bash
# Run validation package unit tests
GOTOOLCHAIN=auto go test -v ./pkg/scheduler/apis/config/validation/...

# Run all config-related tests
GOTOOLCHAIN=auto go test -v ./pkg/scheduler/apis/config/...
```

---

## 7. Critical Invariants for Developers & Agents

1. **No Early Exit in Validators**:
   - Always accumulate errors into `field.ErrorList` or `[]error` slices and return aggregated errors so users receive full configuration diagnostics in a single run.
2. **Precise `field.Path` Tracking**:
   - Never construct bare string errors. Always pass and extend `*field.Path` (using `.Child()`, `.Index()`) so error messages indicate the exact YAML/JSON location.
3. **QueueSort Uniformity**:
   - Any new queue sorting features or parameters must maintain multi-profile parity checks in `validateCommonQueueSort`.
4. **Feature Gate Defensive Validation**:
   - Whenever introducing a feature-gated field in plugin args, ensure setting the field while the feature gate is disabled returns a `field.Forbidden` error rather than being silently ignored.
5. **Plugin Deprecation Tracking**:
   - When deprecating or removing plugins in future component config versions, add the plugin name and API version to `invalidPluginsByVersion`.
