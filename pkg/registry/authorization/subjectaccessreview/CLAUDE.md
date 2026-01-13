# Package: subjectaccessreview

## Purpose
Implements the REST endpoint for SubjectAccessReview, allowing privileged users to check if any user/group can perform a specific action.

## Key Types

- **REST**: Implements the SubjectAccessReview REST endpoint
  - authorizer: The authorizer.Authorizer to check permissions

## Key Functions

- **NewREST(authorizer)**: Creates REST handler with the authorizer
- **NamespaceScoped()**: Returns false - cluster-scoped resource
- **Create()**: Checks authorization for the subject specified in spec
- **GetSingularName()**: Returns "subjectaccessreview"

## Design Notes

- Create-only resource - no persistent storage
- Unlike SelfSubjectAccessReview, checks permissions for a specified user (not the requester)
- Subject is specified in spec: user, groups, uid, extra
- Supports AuthorizeWithSelectors feature gate for field/label selector authorization
- Uses authorizationutil.AuthorizationAttributesFrom to build authorizer attributes
- Returns allowed, denied, reason, and evaluationError in status
- Requires authorization to create (typically admin-only)
