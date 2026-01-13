# Package: leasecandidate

## Purpose
Implements the registry strategy for LeaseCandidate resources in the coordination.k8s.io API group. LeaseCandidates are used for leader election to track candidate controllers competing for a lease.

## Key Types

- **LeaseCandidateStrategy**: Implements the RESTCreateStrategy, RESTUpdateStrategy interfaces for LeaseCandidate validation and lifecycle management.

## Key Functions

- **NamespaceScoped()**: Returns true - LeaseCandidate resources are namespaced.
- **PrepareForCreate/PrepareForUpdate**: Lifecycle hooks (currently no-ops).
- **Validate()**: Validates a new LeaseCandidate using coordination validation.
- **ValidateUpdate()**: Validates LeaseCandidate updates.
- **AllowCreateOnUpdate()**: Returns true - allows creating via PUT requests.
- **AllowUnconditionalUpdate()**: Returns false - requires resource version for updates.
- **GetAttrs()**: Returns labels and fields for filtering/selection.
- **ToSelectableFields()**: Converts LeaseCandidate to selectable fields, including `spec.leaseName`.

## Design Notes

- Uses the standard Kubernetes registry strategy pattern.
- Delegates validation to `pkg/apis/coordination/validation`.
- Supports field selection on `spec.leaseName` for efficient queries.
- Strategy is exported as a package-level `Strategy` variable using the legacy scheme.
