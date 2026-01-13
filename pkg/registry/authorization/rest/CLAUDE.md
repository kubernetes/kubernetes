# Package: rest

## Purpose
Provides the REST storage provider for the "authorization" API group, wiring up all authorization review resources to the API server.

## Key Types

- **RESTStorageProvider**: Implements the storage provider interface
  - Authorizer: The authorizer.Authorizer for access checks
  - RuleResolver: The authorizer.RuleResolver for listing permissions

## Key Functions

- **NewRESTStorage(apiResourceConfigSource, restOptionsGetter)**: Creates APIGroupInfo with storage handlers
- **v1Storage()**: Creates storage map for authorization/v1 with:
  - subjectaccessreviews (cluster-scoped access checks)
  - selfsubjectaccessreviews (check own permissions)
  - localsubjectaccessreviews (namespace-scoped access checks)
  - selfsubjectrulesreviews (list own permissions in a namespace)
- **GroupName()**: Returns "authorization.k8s.io"

## Design Notes

- Returns empty APIGroupInfo if Authorizer is nil (disables authorization API)
- All resources are create-only (no persistent storage)
- Note: When adding versions, also update aggregator.go priorities
