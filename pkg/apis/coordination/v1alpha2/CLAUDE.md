# Package: v1alpha2

## Purpose
Provides conversion and defaulting logic for coordination.k8s.io/v1alpha2 API types (LeaseCandidate). This package bridges the external v1alpha2 API types from k8s.io/api to the internal types.

## Key Functions

- **Resource(resource string)**: Returns a Group-qualified GroupResource for v1alpha2.
- **AddToScheme**: Registers v1alpha2 types and conversion functions with a scheme.
- **RegisterDefaults**: Registers defaulting functions for v1alpha2 types.

## Key Constants

- **GroupName**: "coordination.k8s.io"
- **SchemeGroupVersion**: coordination.k8s.io/v1alpha2

## Design Notes

- Alpha version introducing LeaseCandidate for coordinated leader election.
- Uses generated conversion functions to convert between v1alpha2 external and internal types.
- External types are defined in k8s.io/api/coordination/v1alpha2, not in this package.
