# Package: controllerrevision

## Purpose
Implements the registry strategy for ControllerRevision resources, which store immutable snapshots of controller state used for rollback operations (e.g., by StatefulSets and DaemonSets).

## Key Types

- **strategy**: Implements `rest.RESTCreateStrategy` and `rest.RESTUpdateStrategy` for ControllerRevision objects

## Key Functions

- **Strategy**: Package-level variable providing the default strategy instance
- **NamespaceScoped()**: Returns true - ControllerRevisions are namespace-scoped
- **PrepareForCreate()**: Prepares a ControllerRevision before creation
- **PrepareForUpdate()**: Prepares a ControllerRevision before update
- **Validate()**: Validates a new ControllerRevision using validation.ValidateControllerRevisionCreate
- **ValidateUpdate()**: Validates updates using validation.ValidateControllerRevisionUpdate
- **AllowCreateOnUpdate()**: Returns false - POST is required to create
- **AllowUnconditionalUpdate()**: Returns true - unconditional updates are allowed

## Design Notes

- ControllerRevisions are designed to be immutable snapshots
- Used by StatefulSet and DaemonSet controllers to track revision history
- Follows standard Kubernetes strategy pattern for REST operations
