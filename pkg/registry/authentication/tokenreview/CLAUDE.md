# Package: tokenreview

## Purpose
Implements the REST endpoint for TokenReview, allowing validation of bearer tokens and retrieval of user information associated with them.

## Key Types

- **REST**: Implements the TokenReview REST endpoint
  - tokenAuthenticator: The authenticator.Request to validate tokens
  - apiAudiences: Default audiences to check if none specified in request

## Key Functions

- **NewREST(tokenAuthenticator, apiAudiences)**: Creates REST handler with authenticator
- **NamespaceScoped()**: Returns false - cluster-scoped resource
- **Create()**: Validates the provided token and returns authentication result
- **GetSingularName()**: Returns "tokenreview"

## Design Notes

- Create-only resource - no persistent storage
- Creates a fake HTTP request with the token as Bearer authorization header
- Supports audience validation - checks that token audiences intersect with expected audiences
- Returns authenticated=true/false, user info (username, UID, groups, extra), and any error
- If tokenAuthenticator is nil, returns the review unchanged (useful for testing)
- Token must be non-empty in spec
