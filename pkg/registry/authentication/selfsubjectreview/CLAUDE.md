# Package: selfsubjectreview

## Purpose
Implements the REST endpoint for SelfSubjectReview, allowing users to query their own authentication information (username, UID, groups, extra attributes).

## Key Types

- **REST**: Implements rest.Creater, rest.Scoper, rest.Storage for SelfSubjectReview
  - No backing store - creates responses dynamically from request context

## Key Functions

- **NewREST()**: Creates a new REST handler (no dependencies required)
- **NamespaceScoped()**: Returns false - cluster-scoped resource
- **Create()**: Extracts user info from request context and returns it as SelfSubjectReview status
- **GetSingularName()**: Returns "selfsubjectreview"

## Design Notes

- Create-only resource (no backing storage)
- Extracts user information directly from the request context (genericapirequest.UserFrom)
- Returns username, UID, groups, and extra attributes
- Useful for debugging authentication - lets users see how the API server authenticated them
- No validation of the input object beyond type checking
