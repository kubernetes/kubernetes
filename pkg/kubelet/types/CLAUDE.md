# Package: types

## Purpose
The `types` package defines common types, constants, and utility functions used throughout the Kubelet. It includes pod update operations, pod sources, sync types, and helper utilities.

## Key Types/Structs

- **PodOperation**: Enum for pod configuration changes (ADD, DELETE, REMOVE, UPDATE, RECONCILE).
- **PodUpdate**: Describes an operation on pods from a configuration source. Contains Pods slice, Op, and Source.
- **SyncPodType**: Enum classifying pod sync operations (SyncPodSync, SyncPodUpdate, SyncPodCreate, SyncPodKill).
- **Timestamp**: Wrapper around time.Time with RFC3339Nano formatting utilities.
- **SortedContainerStatuses**: Type for sorting container statuses by name.
- **Reservation**: Represents reserved resources for system and Kubernetes components.
- **ResolvedPodUID/MirrorPodUID**: Type aliases for pod UIDs to distinguish between source pods and mirror pods.
- **HTTPDoer**: Interface wrapping http.Do functionality.

## Key Functions

- **GetValidatedSources**: Validates and returns allowed pod sources (file, http, api).
- **GetPodSource**: Returns the source of a pod from its annotations.
- **IsMirrorPod**: Checks if a pod is a mirror pod (has mirror annotation).
- **IsStaticPod**: Checks if a pod is a static pod (source is not apiserver).
- **IsCriticalPod**: Returns true if pod is static, mirror, or has SystemCriticalPriority.
- **IsCriticalPodBasedOnPriority**: Checks if priority meets SystemCriticalPriority threshold.
- **IsNodeCriticalPod**: Checks if pod has system-node-critical priority class.
- **Preemptable**: Determines if one pod can preempt another based on criticality and priority.
- **SortInitContainerStatuses**: Sorts init container statuses to match spec order.
- **HasRestartableInitContainer**: Returns true if pod has any sidecar (restartable init) containers.

## Constants

- Pod sources: FileSource, HTTPSource, ApiserverSource, AllSource
- Annotation keys: ConfigSourceAnnotationKey, ConfigMirrorAnnotationKey, ConfigFirstSeenAnnotationKey, ConfigHashAnnotationKey
- NamespaceDefault: Default namespace string

## Design Notes

- Static pods come from file or HTTP sources; regular pods come from the API server.
- Mirror pods are created in the API server to represent static pods.
- Critical pods include static pods, mirror pods, and pods with system-critical priority.
