# Package: statefulset

## Purpose
Implements the registry strategy for StatefulSet resources, which manage stateful applications with stable network identities and persistent storage.

## Key Types

- **statefulSetStrategy**: Implements verification logic and REST strategies for StatefulSets
- **statefulSetStatusStrategy**: Strategy for status-only updates

## Key Functions

- **Strategy**: Default logic for creating/updating StatefulSet objects
- **StatusStrategy**: Default logic for status updates
- **PrepareForCreate()**: Clears status, sets Generation=1, drops disabled fields
- **PrepareForUpdate()**: Preserves status, increments Generation on spec changes
- **Validate()**: Validates new StatefulSets with pod template validation
- **ValidateUpdate()**: Validates StatefulSet updates
- **WarningsOnCreate/Update()**: Warns about pod template issues, PVC specs, negative revisionHistoryLimit
- **dropStatefulSetDisabledFields()**: Drops MaxUnavailable if feature gate is disabled
- **maxUnavailableInUse()**: Checks if MaxUnavailable is set in rolling update strategy

## Design Notes

- Namespace-scoped resource
- Implements GarbageCollectionDeleteStrategy for cascading deletes
- Supports MaxUnavailableStatefulSet feature gate for rolling updates
- Includes validation for VolumeClaimTemplates (PVC specs)
- Warns about negative revisionHistoryLimit (retains all revisions)
