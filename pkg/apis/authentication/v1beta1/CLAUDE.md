# Package: v1beta1

## Purpose
Provides v1beta1 versioned API support for the authentication.k8s.io API group, including type registration, defaulting, and conversion.

## Key Constants/Variables
- `GroupName`: "authentication.k8s.io"
- `SchemeGroupVersion`: authentication.k8s.io/v1beta1

## Key Functions
- `Resource(resource string)`: Returns a Group-qualified GroupResource
- `AddToScheme`: Adds v1beta1 types to a scheme
- `addDefaultingFuncs`: Registers defaulting functions

## Design Notes
- Beta version providing stable preview of authentication features
- Uses code generation for conversion and defaulting
- Relies on external types from k8s.io/api/authentication/v1beta1
- Contains SelfSubjectReview for users to check their own identity
