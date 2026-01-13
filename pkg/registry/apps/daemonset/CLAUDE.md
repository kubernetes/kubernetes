# Package: daemonset

## Purpose
Implements the registry strategy for DaemonSet resources, which ensure that a copy of a Pod runs on all (or selected) nodes in the cluster.

## Key Types

- **daemonSetStrategy**: Implements verification logic and REST strategies for DaemonSets
- **daemonSetStatusStrategy**: Strategy for status-only updates

## Key Functions

- **Strategy**: Default logic for creating/updating DaemonSet objects
- **StatusStrategy**: Default logic for status updates
- **PrepareForCreate()**: Clears status, sets Generation=1, initializes TemplateGeneration
- **PrepareForUpdate()**: Preserves status, increments Generation/TemplateGeneration on spec changes
- **Validate()**: Validates new DaemonSets with pod template validation
- **ValidateUpdate()**: Validates DaemonSet updates
- **DefaultGarbageCollectionPolicy()**: Returns DeleteDependents (cascading delete)
- **GetResetFields()**: Returns "status" field to prevent user modification via spec updates

## Design Notes

- Namespace-scoped resource
- TemplateGeneration tracks pod template changes separately from other spec changes
- Status updates are isolated - spec changes during status update are ignored
- Implements GarbageCollectionDeleteStrategy for cascading deletes
- Drops disabled pod template fields based on feature gates
