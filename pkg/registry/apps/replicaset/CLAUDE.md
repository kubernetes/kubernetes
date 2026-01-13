# Package: replicaset

## Purpose
Implements the registry strategy for ReplicaSet resources, which maintain a stable set of replica Pods running at any given time.

## Key Types

- **rsStrategy**: Implements verification logic and REST strategies for ReplicaSets
- **rsStatusStrategy**: Strategy for status-only updates

## Key Functions

- **Strategy**: Default logic for creating/updating ReplicaSet objects
- **StatusStrategy**: Default logic for status updates
- **PrepareForCreate()**: Clears status, sets Generation=1
- **PrepareForUpdate()**: Preserves status, increments Generation on spec changes
- **Validate()**: Validates new ReplicaSets with pod template validation
- **ValidateUpdate()**: Validates ReplicaSet updates
- **WarningsOnCreate()**: Warns if name is not a valid DNS label
- **ToSelectableFields()**: Returns field set for filtering (includes status.replicas)
- **GetAttrs()**: Returns labels and fields for watch filtering
- **MatchReplicaSet()**: Creates SelectionPredicate for etcd watch routing
- **dropDisabledStatusFields()**: Drops TerminatingReplicas if feature gate is disabled

## Design Notes

- Namespace-scoped resource
- Note in code: changes should also be made to ReplicationController
- Supports field selectors on status.replicas
- Implements GarbageCollectionDeleteStrategy for cascading deletes
- Supports DeploymentReplicaSetTerminatingReplicas feature gate
