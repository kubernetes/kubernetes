# Package: apps

## Purpose
Defines the internal (unversioned) API types for the apps API group, which includes workload controllers like Deployments, StatefulSets, DaemonSets, ReplicaSets, and ControllerRevisions.

## Key Types

### StatefulSet
Manages pods with consistent identities (network and storage). Guarantees stable DNS hostnames and persistent volume claims per replica.
- `StatefulSetSpec`: Defines replicas, selector, template, volume claim templates, service name, update strategy
- `StatefulSetStatus`: Tracks replica counts, current/update revisions, conditions
- `StatefulSetUpdateStrategy`: Supports RollingUpdate and OnDelete strategies
- `StatefulSetPersistentVolumeClaimRetentionPolicy`: Controls PVC lifecycle on delete/scale

### Deployment
Provides declarative updates for pods via ReplicaSets.
- `DeploymentSpec`: Replicas, selector, template, strategy, revision history, progress deadline
- `DeploymentStatus`: Replica counts, conditions, collision count
- `DeploymentStrategy`: Supports Recreate and RollingUpdate with MaxUnavailable/MaxSurge

### DaemonSet
Ensures a pod runs on all (or selected) nodes.
- `DaemonSetSpec`: Selector, template, update strategy, min ready seconds
- `DaemonSetStatus`: Scheduled/ready/available counts per node
- `DaemonSetUpdateStrategy`: Supports RollingUpdate and OnDelete

### ReplicaSet
Maintains a stable set of replica pods.
- `ReplicaSetSpec`: Replicas, selector, template
- `ReplicaSetStatus`: Replica counts, conditions

### ControllerRevision
Immutable snapshot of controller state data for rollback support.
- `Data`: Raw extension containing serialized state
- `Revision`: Monotonically increasing revision number

## Key Functions
- `AddToScheme`: Registers all apps types with a scheme
- `Kind(kind string)`: Returns Group-qualified GroupKind
- `Resource(resource string)`: Returns Group-qualified GroupResource

## Design Notes
- Internal types use `api.PodTemplateSpec` and `api.PersistentVolumeClaim` from core
- All workload controllers support label selectors and pod templates
- Update strategies control how pods are replaced during updates
