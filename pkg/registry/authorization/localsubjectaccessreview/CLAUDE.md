# Package: localsubjectaccessreview

## Purpose
Implements the REST endpoint for LocalSubjectAccessReview, which checks if a user can perform an action in a specific namespace.

## Key Types

- **REST**: Implements the LocalSubjectAccessReview REST endpoint
  - authorizer: The authorizer.Authorizer to check permissions

## Key Functions

- **NewREST(authorizer)**: Creates REST handler with the authorizer
- **NamespaceScoped()**: Returns true - namespace-scoped resource (unlike SubjectAccessReview)
- **Create()**: Checks authorization and returns allowed/denied status
- **GetSingularName()**: Returns "localsubjectaccessreview"

## Design Notes

- Create-only resource - no persistent storage
- Namespace-scoped: must be created in a namespace, and the namespace in the URL must match spec.resourceAttributes.namespace
- Supports AuthorizeWithSelectors feature gate for field/label selector authorization
- Uses authorizationutil.AuthorizationAttributesFrom to convert spec to authorizer attributes
- Returns allowed, denied, reason, and evaluationError in status
- Useful for checking namespace-specific permissions
