# Package: v1

## Purpose
Provides v1 versioned API support for the authorization.k8s.io API group, including type registration, defaulting, and conversion between v1 and internal types.

## Key Constants/Variables
- `GroupName`: "authorization.k8s.io"
- `SchemeGroupVersion`: authorization.k8s.io/v1

## Key Functions
- `Resource(resource string)`: Returns a Group-qualified GroupResource
- `AddToScheme`: Adds v1 types to a scheme
- `addDefaultingFuncs`: Registers defaulting functions

## Design Notes
- v1 is the stable API version for authorization
- Uses code generation for conversion and defaulting
- Relies on external types from k8s.io/api/authorization/v1
- SubjectAccessReview and SelfSubjectAccessReview are the primary resources
