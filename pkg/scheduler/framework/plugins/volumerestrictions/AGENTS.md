# AGENTS.md: Developer & Agent Guide for `pkg/scheduler/framework/plugins/volumerestrictions`

This guide provides an architectural overview, volume conflict detection rules, `ReadWriteOncePod` access mode enforcement, preemption state management, event handling logic, and testing strategies for the Kubernetes scheduler `VolumeRestrictions` plugin in `pkg/scheduler/framework/plugins/volumerestrictions`.

---

## 1. High-Level Purpose & Scope

The `VolumeRestrictions` plugin (`names.VolumeRestrictions = "VolumeRestrictions"`) enforces two primary storage access constraints:

1. **In-Tree Raw Volume Disk Conflicts**: Detects node-level disk attachment conflicts for in-tree volume types (GCE Persistent Disk, AWS Elastic Block Store, ISCSI, and Ceph RBD), preventing multiple pods from mounting the same disk in conflicting read/write modes on the same host node.
2. **`ReadWriteOncePod` (RWOP) Access Mode Enforcement**: Enforces cluster-wide single-pod access for PVCs declaring the `ReadWriteOncePod` access mode, ensuring that no two pods across the entire cluster can reference the same RWOP claim simultaneously.
3. **Preemption Coordination**: Implements `PreFilterExtensions` (`AddPod` and `RemovePod`) to dynamically update RWOP claim conflict counts during preemption evaluations (`PostFilter`), allowing higher-priority pods to preempt existing pods holding RWOP claims.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/volumerestrictions/
├── volume_restrictions.go     # Plugin struct, PreFilter, Filter, PreFilterExtensions (AddPod/RemovePod), EnqueueExtensions, and conflict logic
├── volume_restrictions_test.go# Unit tests for raw disk conflicts, RWOP access mode enforcement, preemption, and queueing hints
└── AGENTS.md                  # This agent documentation
```

---

## 3. Core Data Structures & Interfaces

### 3.1. `VolumeRestrictions` Struct

```go
type VolumeRestrictions struct {
    pvcLister                                          corelisters.PersistentVolumeClaimLister
    sharedLister                                       fwk.SharedLister
    enableInPlacePodVerticalScalingSchedulerPreemption bool
}
```

- **`sharedLister`**: Provides access to `StorageInfoLister` (`sharedLister.StorageInfos().IsPVCUsedByPods(key)`), an index maintained by the scheduler cache tracking all PVCs currently in use by scheduled/assumed pods.
- **`pvcLister`**: Used to inspect `pvc.Spec.AccessModes` for `v1.ReadWriteOncePod`.

### 3.2. `preFilterState` (Stored in `CycleState`)

```go
type preFilterState struct {
    readWriteOncePodPVCs   sets.Set[string] // Names of the pod's PVCs using ReadWriteOncePod
    conflictingPVCRefCount int              // Number of references to these RWOP PVCs by existing scheduled pods
}
```
- Stored under key `"PreFilterVolumeRestrictions"`.
- Cloned via `Clone()` when the scheduling framework branches cycle states during parallel node preemption simulations.

---

## 4. Extension Point Implementation & Conflict Detection

```
[ PreFilter Phase ]
  │
  ├── 1. Identify Raw Disk Volumes (needsRestrictionsCheck):
  │      • GCEPersistentDisk, AWSElasticBlockStore, RBD, ISCSI
  │
  ├── 2. Identify ReadWriteOncePod PVCs (readWriteOncePodPVCsForPod):
  │      • Inspect pvc.Spec.AccessModes for v1.ReadWriteOncePod
  │
  ├── 3. Calculate RWOP In-Use References (calPreFilterState):
  │      • For each RWOP PVC: check sharedLister.StorageInfos().IsPVCUsedByPods(pvcKey)
  │      • Set conflictingPVCRefCount
  │
  ├── 4. Early Skip:
  │      • If no raw disk volumes AND conflictingPVCRefCount == 0: return Skip
  │
  └── 5. Write preFilterState to CycleState

[ Filter Phase (Per Candidate Node) ]
  │
  ├── 1. Raw Disk Conflict Check (satisfyVolumeConflicts):
  │      • Compare pod.Spec.Volumes against all existing pods on node (nodeInfo.GetPods())
  │      • GCE PD: Same PDName allowed only if BOTH pods mount ReadOnly
  │      • AWS EBS: Same VolumeID disallowed under all circumstances
  │      • ISCSI: Same IQN allowed only if BOTH pods mount ReadOnly
  │      • Ceph RBD: Same (Monitors ∩ Pool ∩ Image) allowed only if BOTH pods mount ReadOnly
  │      • If conflict detected: return Unschedulable ("node(s) had no available disk")
  │
  └── 2. ReadWriteOncePod Conflict Check (satisfyReadWriteOncePod):
         • If state.conflictingPVCRefCount > 0:
           return Unschedulable ("node(s) unavailable due to PersistentVolumeClaim with ReadWriteOncePod access mode already in-use by another pod")

[ Preemption Evaluation (PostFilter) ]
  │
  ├── RemovePod:
  │   • When a victim pod is speculatively removed from candidate node:
  │   • Decrement state.conflictingPVCRefCount if victim pod mounted the RWOP claim
  │
  └── AddPod:
      • If victim removal is rolled back:
      • Increment state.conflictingPVCRefCount
```

---

## 5. Detailed Conflict Rules

### 5.1. Raw Volume Conflict Rules (`isVolumeConflict`)

| Storage Plugin | Conflict Criteria | Allowed Concurrency |
|---|---|---|
| **GCE Persistent Disk** (`GCEPersistentDisk`) | `v1.PDName == v2.PDName` | Allowed **only if both** mounts specify `ReadOnly: true`. |
| **AWS Elastic Block Store** (`AWSElasticBlockStore`) | `v1.VolumeID == v2.VolumeID` | **Never allowed** (AWS EBS attaches to only one instance at a time). |
| **ISCSI** (`ISCSI`) | `v1.IQN == v2.IQN` | Allowed **only if both** mounts specify `ReadOnly: true`. |
| **Ceph RBD** (`RBD`) | Common Monitors (`haveOverlap(mon1, mon2)`) **AND** `v1.RBDPool == v2.RBDPool` **AND** `v1.RBDImage == v2.RBDImage` | Allowed **only if both** mounts specify `ReadOnly: true`. |

### 5.2. `ReadWriteOncePod` (RWOP) Access Mode Semantics
- Introduced in Kubernetes to restrict volume access to a single pod across the entire cluster (unlike `ReadWriteOnce` which restricts access to a single node).
- In `PreFilter`, the plugin queries `sharedLister.StorageInfos().IsPVCUsedByPods(key)`.
- If the claim is in use by another running, scheduled, or assumed pod, `conflictingPVCRefCount` is set to `1` (or more).
- During `Filter`, all nodes are marked `Unschedulable`. This enables the scheduler to trigger `PostFilter` (preemption): if the unscheduled pod has higher priority than the pod currently using the RWOP claim, the preemption algorithm can evict the victim pod.
- When the victim pod is simulated for removal, `RemovePod` decrements `conflictingPVCRefCount` to `0`, making candidate nodes feasible.

---

## 6. Event Handling, QueueingHints & EnqueueExtensions

`VolumeRestrictions` registers cluster events to requeue unschedulable pods:

| Resource | Action | QueueingHint Function | Trigger Condition |
|---|---|---|---|
| **`AssignedPod`** | `Delete` | `isSchedulableAfterAssignedPodDeleted` | • Deleted pod was assigned/nominated in same namespace.<br>• Deleted pod had conflicting raw disk volume with target pod.<br>• Deleted pod shared any PVC with target pod (potential RWOP release). |
| **`Node`** | `Add` | `nil` (always Queue) | • New node added to cluster. |
| **`PersistentVolumeClaim`** | `Add` | `isSchedulableAfterPersistentVolumeClaimAdded` | • PVC created in pod namespace matching one of the pod's requested volume claims. |

---

## 7. Testing Guide & Verification

### 7.1. Key Test Cases in `volume_restrictions_test.go`:
- **GCE PD Conflicts**: Single-node multi-pod read-write vs read-only mount combinations.
- **AWS EBS Conflicts**: Multiple pods requesting same volume ID on same node.
- **ISCSI & Ceph RBD Conflicts**: Tests IQN matching and Ceph monitor/pool/image overlap.
- **ReadWriteOncePod Filtering**: Rejection of pods when RWOP PVC is already in use by another scheduled pod.
- **PreFilterExtensions Preemption**: Verifies `AddPod` / `RemovePod` properly mutate `conflictingPVCRefCount` during simulated preemption.
- **QueueingHints**: Tests precise wake-up conditions when pods holding conflicting disks or RWOP claims are deleted.

### 7.2. Running Tests:
```bash
# Run unit tests for volumerestrictions plugin
GOTOOLCHAIN=auto go test -v ./pkg/scheduler/framework/plugins/volumerestrictions/...

# Run with race detector
GOTOOLCHAIN=auto go test -v -race ./pkg/scheduler/framework/plugins/volumerestrictions/...
```

---

## 8. Critical Invariants for Developers & AI Agents

1. **Read-Only Volume Concurrency**:
   - For GCE PD, ISCSI, and Ceph RBD, conflict checks must strictly require `disk.ReadOnly && existingDisk.ReadOnly`. If either is read-write, conflict is declared.
2. **Cluster-Wide vs Node-Local Isolation**:
   - Raw disk conflicts are evaluated **locally per node** during `Filter` against `nodeInfo.GetPods()`.
   - `ReadWriteOncePod` conflicts are evaluated **cluster-wide** in `PreFilter` via `StorageInfoLister.IsPVCUsedByPods()`.
3. **PreFilter State Cloning**:
   - `preFilterState.Clone()` must create a deep copy of `readWriteOncePodPVCs` and copy `conflictingPVCRefCount` so concurrent preemption evaluations on different candidate nodes do not race.
4. **PreFilter Skip Optimization**:
   - If a pod has no in-tree raw volumes (`needsRestrictionsCheck` is false) and `conflictingPVCRefCount == 0`, `PreFilter` returns `Skip` to eliminate per-node filter overhead.
5. **No PVC Fetch in AssignedPod QueueingHint**:
   - `isSchedulableAfterAssignedPodDeleted` compares PVC claim names directly without fetching PVC objects from listers to keep QueueingHint execution latency minimal.
