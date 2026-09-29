# AGENTS.md: Developer & Agent Guide for `pkg/scheduler/framework/plugins/nodevolumelimits`

This guide provides an architectural overview, volume counting algorithms, CSI attach limit enforcement rules, in-tree CSI migration translation, event handling logic, and testing strategies for the Kubernetes scheduler `NodeVolumeLimits` (CSI) plugin in `pkg/scheduler/framework/plugins/nodevolumelimits`.

---

## 1. High-Level Purpose & Scope

The `NodeVolumeLimits` plugin (`names.NodeVolumeLimits = "NodeVolumeLimits"`, struct `CSILimits`) enforces maximum attachable volume limits per CSI storage driver on each cluster node. Cloud providers and storage backends impose hardware or driver-level constraints on how many block or file volumes a single virtual machine/node can simultaneously attach (e.g., AWS EBS volume limits, GCP Persistent Disk limits, Azure Managed Disks).

### Core Objectives:
1. **CSI Driver Limit Enforcement**: Enforces `.spec.drivers[].allocatable.count` limits defined on `CSINode` objects.
2. **Multi-Source Volume Counting**: Aggregates attachable volumes across PVC references, generic ephemeral inline volumes, migrated in-tree volumes, and active API `VolumeAttachment` objects.
3. **Volume Deduplication**: Correctly counts shared volumes mounted by multiple pods on the same node exactly once against the driver limit.
4. **CSI Driver Availability Check**: When `VolumeLimitScaling` is enabled, validates driver installation on the node for drivers requiring `spec.preventPodSchedulingIfMissing`.
5. **In-Tree CSI Migration Transparency**: Automatically translates legacy in-tree volume specifications (AWS EBS, GCE PD, Azure Disk, Portworx, Cinder) into their equivalent CSI driver representations during limit checks.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/nodevolumelimits/
├── csi.go                     # CSILimits plugin struct, PreFilter, Filter, EnqueueExtensions, QueueingHints, and volume counting logic
├── csi_manager.go             # DefaultCSIManager and CSINode lister wrappers (implements fwk.CSIManager)
├── utils.go                   # isCSIMigrationOn helper for supported storage plugins
├── csi_test.go                # Comprehensive unit test suite covering attach limits, in-tree translation, VA deduplication, and hints
└── AGENTS.md                  # This agent documentation
```

---

## 3. Core Data Structures & Interfaces

### 3.1. `CSILimits` Struct

```go
type CSILimits struct {
    csiManager                                         fwk.CSIManager
    pvLister                                           corelisters.PersistentVolumeLister
    pvcLister                                          corelisters.PersistentVolumeClaimLister
    scLister                                           storagelisters.StorageClassLister
    vaLister                                           storagelisters.VolumeAttachmentLister
    csiDriverLister                                    storagelisters.CSIDriverLister
    vaIndexer                                          cache.Indexer

    randomVolumeIDPrefix                               string
    enableVolumeLimitScaling                           bool
    enableInPlacePodVerticalScalingSchedulerPreemption bool
    translator                                         InTreeToCSITranslator
}
```

- **`CSIManager`**: Abstracts access to `CSINode` resources (`csiManager.CSINodes().Get(nodeName)`).
- **`vaIndexer`**: An in-memory cache index on `VolumeAttachment` resources keyed by `va.spec.nodename` (`vaIndexKey = "va.spec.nodename"`).
- **`randomVolumeIDPrefix`**: A 32-character random string generated at plugin initialization used to form synthetic volume identifiers for unbound claims.
- **`InTreeToCSITranslator`**: Translates in-tree volume specifications to CSI driver names and volume handles (`k8s.io/csi-translation-lib`).

### 3.2. Volume Unique Identifier Format
Volumes are identified across pods and attachments using a deterministic format:
```
<driverName>/<volumeHandle>
```
For example: `ebs.csi.aws.com/vol-0123456789abcdef0`.

---

## 4. Extension Point Implementation & Volume Counting Algorithm

```
[ PreFilter Phase ]
  │
  ├── Check pod.Spec.Volumes:
  │   • If has PVC, Ephemeral inline volume, or migratable in-tree volume: proceed
  │   • Otherwise: return Skip
  │
[ Filter Phase (Per Candidate Node) ]
  │
  ├── 1. Fetch CSINode object for candidate node
  │
  ├── 2. Extract attachable volumes for candidate Pod (filterAttachableVolumes):
  │      • PVCs: resolve driver & volumeHandle from PV or StorageClass
  │      • Ephemeral inline volumes: resolve computed claim name
  │      • Inline volumes: translate via in-tree translator if migration enabled
  │      • Generate map[volumeUniqueName]driverName (newVolumes)
  │      • If newVolumes is empty: return Success (no CSI volumes to attach)
  │
  ├── 3. Driver Presence Check (if EnableVolumeLimitScaling):
  │      • For drivers with preventPodSchedulingIfMissing=true: verify driver in CSINode.Spec.Drivers
  │      • If missing: return Unschedulable ("<driver> CSI driver is not installed on the node")
  │
  ├── 4. Read Node Volume Limits (getVolumeLimits):
  │      • Extract map[driverName]int64 from CSINode.Spec.Drivers[].Allocatable.Count
  │      • If node has no limits: return Success
  │
  ├── 5. Count Existing Attached Volumes on Node:
  │      • Step A: Iterate existing scheduled pods on node (nodeInfo.GetPods())
  │        - Collect attached volumes into attachedVolumes map
  │      • Step B: Deduplicate candidate pod volumes
  │        - If volume is already in attachedVolumes: delete from newVolumes (shared mount!)
  │      • Step C: Query VolumeAttachment indexer by node.Name
  │        - For each VA matching node: if volume not yet in attachedVolumes, count it
  │
  ├── 6. Aggregate Counts & Check Limits:
  │      • For each driver:
  │        if (attachedVolumeCount[driver] + newVolumeCount[driver]) > maxVolumeLimit[driver]:
  │            return Unschedulable ("node(s) exceed max volume count")
  │
  └── 7. Return Success if all driver limits are satisfied
```

---

## 5. Detailed Component Mechanics

### 5.1. `PreFilter`
- Checks if the pod has any volume that could contribute to CSI attach limits:
  - `vol.PersistentVolumeClaim != nil`
  - `vol.Ephemeral != nil`
  - `translator.IsInlineMigratable(vol)`
- If none of these are present, returns `fwk.NewStatus(fwk.Skip)` to bypass the `Filter` phase entirely.

### 5.2. `filterAttachableVolumes` & Volume Resolution
- **Bound PVCs**: Reads `pv.Spec.CSI.Driver` and `pv.Spec.CSI.VolumeHandle`. If the PV is an in-tree migratable volume (e.g. `pv.Spec.AWSElasticBlockStore`), translates to CSI PV via `translator.TranslateInTreePVToCSI`.
- **Unbound PVCs**: When the PVC is not yet bound to a PV (`pvc.Spec.VolumeName == ""`), falls back to `getCSIDriverInfoFromSC`:
  - Retrieves `StorageClass` from `pvc.Spec.StorageClassName`.
  - Determines driver name from `storageClass.Provisioner` (translating in-tree provisioners if migrated).
  - Assigns a unique synthetic handle: `<randomPrefix>-<namespace>/<pvcName>`.
  - *Invariant*: The random prefix ensures unbound claims do not collide with existing PV handles while still allowing deduplication within the same scheduling cycle.
- **Generic Ephemeral Volumes**: Computes claim name via `ephemeral.VolumeClaimName(pod, vol)` and verifies ownership (`ephemeral.VolumeIsForPod(pod, pvc)`).
- **Inline Migratable Volumes**: Translated to temporary CSI PV representation via `translator.TranslateInTreeInlineVolumeToCSI`.

### 5.3. Volume Deduplication Logic
1. **Intra-Pod Sharing**: If a pod defines multiple volume entries pointing to the same PVC, `filterAttachableVolumes` maps them to the same `volumeUniqueName`, deduplicating them in `newVolumes`.
2. **Inter-Pod Sharing on Same Node**: If an existing scheduled pod on the node already mounts `ebs.csi.aws.com/vol-abc`, and the new pod also references `ebs.csi.aws.com/vol-abc`:
   - `delete(newVolumes, "ebs.csi.aws.com/vol-abc")`
   - The volume is counted once in `attachedVolumeCount`, and `newVolumeCount` for that volume becomes `0`.
3. **VolumeAttachment Deduplication**: Volumes discovered via `VolumeAttachment` resources that are already accounted for via existing scheduled pods are not counted twice.

### 5.4. `VolumeAttachment` Indexer
- The plugin registers an indexer `va.spec.nodename` on the `VolumeAttachment` informer during initialization (`NewCSI`).
- In `Filter`, `pl.getNodeVolumeAttachmentInfo(logger, node.Name)` queries `vaIndexer.ByIndex(vaIndexKey, nodeName)`.
- This ensures volumes currently attached to the node by the attach/detach controller (even if the associated pod is terminating or starting) are accounted for, preventing attach limit overcommit.

---

## 6. Event Handling, QueueingHints & EnqueueExtensions

`NodeVolumeLimits` registers the following cluster events to re-evaluate unschedulable pods:

| Resource | Action | QueueingHint Function | Trigger Condition |
|---|---|---|---|
| **`CSINode`** | `Add` | `nil` (always Queue) | • Any new `CSINode` might have available volume capacity. |
| **`CSINode`** | `Update` | `isSchedulableAfterCSINodeUpdated` | • `CSINode.Spec.Drivers[].Allocatable.Count` increased for any driver. |
| **`AssignedPod`** | `Delete` | `isSchedulableAfterAssignedPodDeleted` | • Deleted pod had assigned/nominated node and referenced PVCs, generic ephemeral, or migratable volumes. |
| **`PersistentVolumeClaim`** | `Add` | `isSchedulableAfterPVCAdded` | • PVC created in pod namespace matching pod's volume claim name. |
| **`VolumeAttachment`** | `Delete` | `isSchedulableAfterVolumeAttachmentDeleted` | • Deleted VA matches pod's PVC claim or CSI driver of an inline migratable volume. |

---

## 7. Testing Guide & Verification

### 7.1. Key Test Cases in `csi_test.go`:
- **Limit Enforcement**: Tests nodes with single and multiple CSI drivers reaching max capacity (`TestCSILimitsFilter`).
- **Volume Deduplication**: Verifies that shared ReadOnly or ReadWrite volumes mounted by multiple pods on the same node count as a single attachment (`TestCSILimitsFilterWithSharedVolumes`).
- **Unbound Claims**: Tests volume count allocation with unbound PVCs and storage class provisioners.
- **In-Tree Migration**: Tests limit tracking for migrated in-tree volumes (AWS EBS, GCE PD, Azure Disk, Cinder) translated into CSI equivalents.
- **VolumeAttachments**: Verifies attachments in progress are counted and deduplicated against running pods.
- **QueueingHints**: Tests precise triggering of `isSchedulableAfterCSINodeUpdated`, `isSchedulableAfterAssignedPodDeleted`, `isSchedulableAfterPVCAdded`, and `isSchedulableAfterVolumeAttachmentDeleted`.

### 7.2. Running Tests:
```bash
# Run unit tests for nodevolumelimits plugin
GOTOOLCHAIN=auto go test -v ./pkg/scheduler/framework/plugins/nodevolumelimits/...

# Run with race detector
GOTOOLCHAIN=auto go test -v -race ./pkg/scheduler/framework/plugins/nodevolumelimits/...
```

---

## 8. Critical Invariants for Developers & AI Agents

1. **Volume Unique Name Format**:
   - Always use `getVolumeUniqueName(driverName, volumeHandle)` (`"<driverName>/<volumeHandle>"`) to ensure consistent map keys between pod volume scans and `VolumeAttachment` scans.
2. **Deduplication Priority**:
   - Always subtract existing node volume references from the candidate pod's `newVolumes` map prior to aggregating driver counts. Shared volume mounts must never consume additional attach slots.
3. **Unbound PVC Synthetic Identifiers**:
   - When a PVC is unbound, use `randomVolumeIDPrefix` when constructing the synthetic volume handle. Never use an empty string or fixed string that could collide across claims.
4. **Missing Driver Opt-In (`PreventPodSchedulingIfMissing`)**:
   - When `enableVolumeLimitScaling` is active, only block scheduling if `csiDriver.Spec.PreventPodSchedulingIfMissing` is explicitly `true`. If `false` or unset, allow scheduling even if `CSINode` does not list the driver.
5. **No Mutation of Informer Objects**:
   - Objects returned from `pvLister`, `pvcLister`, `scLister`, `vaLister`, and `csiManager` are shared informer pointers. Never mutate them directly.
