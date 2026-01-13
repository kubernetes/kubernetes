# Package: rest

## Purpose
Provides the REST storage provider for the "authentication" API group, wiring up authentication-related resources (TokenReview, SelfSubjectReview) to the API server.

## Key Types

- **RESTStorageProvider**: Implements the storage provider interface for authentication API group
  - Contains Authenticator (for token validation) and APIAudiences

## Key Functions

- **NewRESTStorage(apiResourceConfigSource, restOptionsGetter)**: Creates APIGroupInfo with storage handlers for enabled authentication resources
- **v1Storage()**: Creates storage map for authentication/v1 with tokenreviews and selfsubjectreviews
- **v1beta1Storage()**: Creates storage map for authentication/v1beta1 with selfsubjectreviews
- **GroupName()**: Returns "authentication.k8s.io"

## Design Notes

- TokenReview requires an authenticator to validate bearer tokens
- SelfSubjectReview does not require an authenticator (returns info about current user)
- Both v1 and v1beta1 API versions are supported for selfsubjectreviews
- Note: When adding versions, also update aggregator.go priorities
