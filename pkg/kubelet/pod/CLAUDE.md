# Package: pod

Manages pod storage and static-to-mirror pod mappings for the kubelet.

## Key Types

- **Manager**: Interface for accessing pods, maintaining mappings between static pods and their mirror pod counterparts.
- **basicManager**: Implementation storing pods indexed by UID and full name (namespace/name).
- **MirrorClient**: Interface for creating/deleting mirror pods in the API server.
- **basicMirrorClient**: Implementation of MirrorClient using Kubernetes API.

## Key Functions

### Manager
- `NewBasicPodManager()`: Creates a new pod manager.
- `GetPodByFullName() / GetPodByName() / GetPodByUID()`: Retrieve pods by various identifiers.
- `GetPodByMirrorPod() / GetMirrorPodByPod()`: Navigate between static and mirror pod pairs.
- `GetPods()`: Returns all non-mirror pods bound to the kubelet.
- `SetPods() / AddPod() / UpdatePod() / RemovePod()`: Manage pod storage.
- `TranslatePodUID()`: Converts mirror pod UID to static pod UID.

### MirrorClient
- `CreateMirrorPod()`: Creates mirror pod with config hash annotation and node owner reference.
- `DeleteMirrorPod()`: Deletes mirror pod by full name, with optional UID precondition.

## Design Notes

- Static pods come from file/http sources; mirror pods are API server representations
- Mirror pods have same full name (namespace/name) as static pods but different UIDs
- Mirror pods use ConfigMirrorAnnotationKey annotation containing static pod hash
- MirrorPodNodeRestriction requires mirror pods to have node owner reference
- All maps protected by RWMutex for thread safety
