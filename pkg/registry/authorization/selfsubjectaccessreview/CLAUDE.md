# Package: selfsubjectaccessreview

## Purpose
Implements the REST endpoint for SelfSubjectAccessReview, allowing users to check their own permissions for a specific action.

## Key Types

- **REST**: Implements the SelfSubjectAccessReview REST endpoint
  - authorizer: The authorizer.Authorizer to check permissions

## Key Functions

- **NewREST(authorizer)**: Creates REST handler with the authorizer
- **NamespaceScoped()**: Returns false - cluster-scoped resource
- **Create()**: Checks if the requesting user can perform the specified action
- **GetSingularName()**: Returns "selfsubjectaccessreview"

## Design Notes

- Create-only resource - no persistent storage
- Uses the user from the request context (not from spec) as the subject
- Can check either ResourceAttributes (API resources) or NonResourceAttributes (URLs)
- Supports AuthorizeWithSelectors feature gate for field/label selector authorization
- Returns allowed, denied, reason, and evaluationError in status
- Commonly used by kubectl auth can-i
