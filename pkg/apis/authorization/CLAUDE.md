# Package: authorization

## Purpose
Defines the internal (unversioned) API types for the authorization.k8s.io API group, handling access control decisions through SubjectAccessReview APIs.

## Key Types

### SubjectAccessReview
Checks whether a user or group can perform an action cluster-wide.
- `SubjectAccessReviewSpec`: Specifies the user/groups, and resource or non-resource attributes to check
- `SubjectAccessReviewStatus`: Returns allowed/denied decision with reason

### SelfSubjectAccessReview
Checks whether the current user can perform an action (no impersonation needed).

### LocalSubjectAccessReview
Namespace-scoped version of SubjectAccessReview for easier RBAC policy grants.

### SelfSubjectRulesReview
Enumerates all actions the current user can perform within a namespace.
- `SubjectRulesReviewStatus`: Returns lists of ResourceRules and NonResourceRules

### ResourceAttributes
Describes a resource access request with namespace, verb, group, version, resource, subresource, name.
- `FieldSelectorAttributes`: Limits access based on field selectors
- `LabelSelectorAttributes`: Limits access based on label selectors

### NonResourceAttributes
Describes a non-resource access request with path and HTTP verb.

## Key Functions
- `AddToScheme`: Registers all authorization review types
- `Kind(kind string)`: Returns Group-qualified GroupKind
- `Resource(resource string)`: Returns Group-qualified GroupResource

## Design Notes
- SubjectAccessReview is the authoritative way to check permissions
- SelfSubjectRulesReview is for UI display only, not authorization decisions
- Supports both resource (CRUD) and non-resource (URL path) access checks
