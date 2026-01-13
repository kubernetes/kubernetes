# Package: deployment

## Purpose
Implements the registry strategy for Deployment resources, which provide declarative updates for Pods and ReplicaSets.

## Key Types

- **deploymentStrategy**: Implements verification logic and REST strategies for Deployments
- **deploymentStatusStrategy**: Strategy for status-only updates

## Key Functions

- **Strategy**: Default logic for creating/updating Deployment objects
- **StatusStrategy**: Default logic for status updates
- **PrepareForCreate()**: Clears status, sets Generation=1, drops disabled template fields
- **PrepareForUpdate()**: Preserves status, increments Generation on spec/annotation changes
- **Validate()**: Validates new Deployments with pod template validation
- **ValidateUpdate()**: Validates Deployment updates
- **WarningsOnCreate()**: Warns if name is not a valid DNS label (affects Pod names)
- **DefaultGarbageCollectionPolicy()**: Returns DeleteDependents (cascading delete)
- **GetResetFields()**: Returns "status" field for spec updates, "spec"+"labels" for status updates
- **dropDisabledStatusFields()**: Drops TerminatingReplicas if feature gate is disabled

## Design Notes

- Namespace-scoped resource
- Generation increments on spec OR annotation changes (annotations propagate to ReplicaSets)
- Status strategy also resets labels to prevent modification during status updates
- Supports DeploymentReplicaSetTerminatingReplicas feature gate
