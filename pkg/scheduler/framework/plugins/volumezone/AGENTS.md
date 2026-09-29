# AGENTS.md: Developer & Agent Guide for `pkg/scheduler/framework/plugins/volumezone`

This guide provides an architectural overview, topology label evaluation rules, beta-to-GA label translation mechanics, event handling logic, and testing strategies for the Kubernetes scheduler `VolumeZone` plugin in `pkg/scheduler/framework/plugins/volumezone`.

---

## 1. High-Level Purpose & Scope

The `VolumeZone` plugin (`names.VolumeZone = "VolumeZone"`) evaluates zone and region scheduling constraints for pods requesting pre-bound PersistentVolumes. It ensures that pods are scheduled onto nodes residing in the same availability zones or regions where their pre-existing storage volumes physically exist.

### Core Objectives:
1. **Topology Constraint Enforcement**: Validates that candidate nodes have matching zone and region labels corresponding to the topology labels specified on the pod's bound PersistentVolumes.
2. **Dual-Label Compatibility (Beta & GA)**: Seamlessly supports and cross-evaluates both legacy beta topology labels (`failure-domain.beta.kubernetes.io/*`) and GA topology labels (`topology.kubernetes.io/*`).
3. **Delayed Binding Coexistence**: Distinguishes between immediate-binding PVs (which must have zone constraints enforced during filtering) and `VolumeBindingWaitForFirstConsumer` claims (which defer zone placement to the `VolumeBinding` plugin).
4. **Single-Zone / Non-Topological Tolerance**: Automatically permits scheduling on nodes that define no topology labels, supporting clusters without zone awareness.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/volumezone/
├── volume_zone.go             # VolumeZone plugin struct, PreFilter, Filter, EnqueueExtensions, QueueingHints, and label evaluation logic
├── volume_zone_test.go        # Unit tests covering zone/region matching, beta/GA translation, unbound PVC handling, and queueing hints
└── AGENTS.md                  # This agent documentation
```

---

## 3. Core Data Structures & Interfaces

### 3.1. `VolumeZone` Struct

```go
type VolumeZone struct {
    pvLister                                           corelisters.PersistentVolumeLister
    pvcLister                                          corelisters.PersistentVolumeClaimLister
    scLister                                           storagelisters.StorageClassLister
    enableInPlacePodVerticalScalingSchedulerPreemption bool
}
```

### 3.2. Topology Data Structures

```go
// pvTopology holds the parsed zone/region constraints of a single PersistentVolume
type pvTopology struct {
    pvName string
    key    string          // e.g. "topology.kubernetes.io/zone"
    values sets.Set[string]// Set of allowed zone strings (e.g. {"us-central1-a", "us-central1-b"})
}

// stateData is stored in CycleState under "PreFilterVolumeZone"
type stateData struct {
    podPVTopologies []pvTopology
}
```

### 3.3. Supported Topology Labels

```go
var topologyLabels = []string{
    v1.LabelFailureDomainBetaZone,   // "failure-domain.beta.kubernetes.io/zone"
    v1.LabelFailureDomainBetaRegion, // "failure-domain.beta.kubernetes.io/region"
    v1.LabelTopologyZone,            // "topology.kubernetes.io/zone"
    v1.LabelTopologyRegion,          // "topology.kubernetes.io/region"
}
```

---

## 4. Extension Point Implementation & Topology Evaluation

```
[ PreFilter Phase ]
  │
  ├── 1. Extract PVCs from pod.Spec.Volumes
  │
  ├── 2. For each PVC (getPVbyPod):
  │      • Retrieve PVC object
  │      • If pvc.Spec.VolumeName == "":
  │        - Inspect StorageClass.VolumeBindingMode
  │        - If WaitForFirstConsumer: Skip claim (handled by VolumeBinding plugin)
  │        - If Immediate: return UnschedulableAndUnresolvable ("PersistentVolume had no name")
  │      • Retrieve bound PV object
  │      • Extract topology constraints (getPVTopologies):
  │        - For each topology key in topologyLabels present on PV.ObjectMeta.Labels:
  │        - Parse comma-separated zones via volumehelpers.LabelZonesToSet(value)
  │        - Append pvTopology{pvName, key, values} to podPVTopologies
  │
  ├── 3. Early Skip:
  │      • If len(podPVTopologies) == 0: return Skip
  │
  └── 4. Write stateData{podPVTopologies} to CycleState

[ Filter Phase (Per Candidate Node) ]
  │
  ├── 1. Node Topology Check:
  │      • Check if node has ANY topology label from topologyLabels
  │      • If node has no topology labels: return Success (non-zoned cluster tolerance)
  │
  ├── 2. Evaluate PV Topology Constraints:
  │      • For each pvTopology in podPVTopologies:
  │        - Check exact key (node.Labels[pvTopology.key])
  │        - Check translated GA key (node.Labels[translateToGALabel(pvTopology.key)])
  │        - If node has either label:
  │            verify pvTopology.values.Has(nodeLabelValue)
  │            if NOT in set:
  │                return UnschedulableAndUnresolvable ("node(s) had no available volume zone")
  │
  └── 3. Return Success if all PV topology constraints match node labels
```

---

## 5. Detailed Component Mechanics

### 5.1. `PreFilter` & Claim Inspection
- Iterates over `pod.Spec.Volumes` filtering for `vol.PersistentVolumeClaim != nil`.
- Resolves each claim against the `PersistentVolumeClaimLister`:
  - If the PVC does not exist: returns `UnschedulableAndUnresolvable`.
  - If `pvc.Spec.VolumeName == ""`:
    - Checks `StorageClass.VolumeBindingMode`. If `VolumeBindingWaitForFirstConsumer`, skips this PVC because volume zone selection will occur during dynamic binding in `VolumeBinding`.
    - If immediate binding mode and no PV name is present, fails immediately with `UnschedulableAndUnresolvable`.
- Retrieves the bound `PersistentVolume` from `PersistentVolumeLister`.
- Extracts topology labels from `pv.ObjectMeta.Labels` (`getPVTopologies`):
  - Uses `volumehelpers.LabelZonesToSet(value)` to support multi-zone specifications (e.g. `"us-east-1a,us-east-1b"`).
  - Note: `spec.nodeAffinity` on the PV is not evaluated here; it is evaluated in the `NodeAffinity` and `VolumeBinding` plugins.
- Stores the slice of `pvTopology` in `CycleState` under key `"PreFilterVolumeZone"`.

### 5.2. `Filter` & Label Matching
- **Non-Zoned Cluster Bypass**:
  - If candidate node does not contain any of the four recognized topology labels, `Filter` immediately returns `nil`. This prevents scheduling failures in single-zone or on-premise clusters lacking zone annotations.
- **Cross-Version Label Translation (`translateToGALabel`)**:
  - If the PV specifies `failure-domain.beta.kubernetes.io/zone = us-central1-a` and the node specifies `topology.kubernetes.io/zone = us-central1-a`, `translateToGALabel` ensures the match succeeds.
  - Checks:
    ```go
    nodeLabelVal, ok := node.Labels[pvTopology.key]
    if !ok {
        nodeLabelVal, ok = node.Labels[translateToGALabel(pvTopology.key)]
    }
    ```
- **Constraint Satisfaction**:
  - If the node has a matching key, its value must exist in `pvTopology.values`.
  - If the value does not exist in `pvTopology.values`, the node cannot satisfy the storage locality; returns `UnschedulableAndUnresolvable` with reason `ErrReasonConflict = "node(s) had no available volume zone"`.

---

## 6. Event Handling, QueueingHints & EnqueueExtensions

`VolumeZone` registers the following cluster events:

| Resource | Action | QueueingHint Function | Trigger Condition |
|---|---|---|---|
| **`StorageClass`** | `Add` | `isSchedulableAfterStorageClassAdded` | • New StorageClass created with `volumeBindingMode == WaitForFirstConsumer`. |
| **`Node`** | `Add \| UpdateNodeLabel` | `nil` (always Queue) | • Node created or topology labels updated. |
| **`PersistentVolumeClaim`** | `Add \| Update` | `isSchedulableAfterPersistentVolumeClaimChange` | • PVC created or updated in matching namespace.<br>• PVC is referenced in pod's `Spec.Volumes`. |
| **`PersistentVolume`** | `Add \| Update` | `isSchedulableAfterPersistentVolumeChange` | • New PV created.<br>• Existing PV's topology labels modified (`!reflect.DeepEqual(oldTopologies, newTopologies)`). |

---

## 7. Testing Guide & Verification

### 7.1. Key Test Cases in `volume_zone_test.go`:
- **Single & Multi-Zone Matching**: Validates matching nodes, mismatched zone nodes, and multi-zone PV sets.
- **Beta vs GA Labels**: Verifies cross-compatibility between beta labels on PV and GA labels on nodes (and vice versa).
- **Region & Zone Combinations**: Tests simultaneous region and zone constraints across nodes.
- **Non-Topological Nodes**: Verifies nodes without topology labels are accepted.
- **WaitForFirstConsumer Claims**: Verifies unbound delayed claims are skipped in PreFilter while immediate unbound claims fail.
- **QueueingHints**: Tests precise requeueing when PV topology labels change or matching PVCs are bound.

### 7.2. Running Tests:
```bash
# Run unit tests for volumezone plugin
GOTOOLCHAIN=auto go test -v ./pkg/scheduler/framework/plugins/volumezone/...

# Run with race detector
GOTOOLCHAIN=auto go test -v -race ./pkg/scheduler/framework/plugins/volumezone/...
```

---

## 8. Critical Invariants for Developers & AI Agents

1. **CycleState Caching**:
   - `PreFilter` pre-computes all `pvTopology` slices. `Filter` must retrieve them from `CycleState` (`getStateData(cs)`) to avoid repeated API/lister lookups during parallel node filtering.
2. **Immediate vs Delayed Binding Distinction**:
   - Unbound PVCs with `WaitForFirstConsumer` MUST be skipped in `PreFilter`. They must never trigger `UnschedulableAndUnresolvable` in `VolumeZone`.
3. **Dual-Label Normalization**:
   - Always evaluate both the declared PV label key and its translated GA equivalent via `translateToGALabel()`.
4. **Node Topology Optionality**:
   - Do not require nodes to have topology labels. A node lacking topology labels is treated as having no zone constraints.
5. **PV NodeAffinity Boundary**:
   - `VolumeZone` inspects `pv.ObjectMeta.Labels`. `pv.Spec.NodeAffinity` is handled by `VolumeBinding` and `NodeAffinity` plugins; do not duplicate node affinity parsing in `VolumeZone`.
