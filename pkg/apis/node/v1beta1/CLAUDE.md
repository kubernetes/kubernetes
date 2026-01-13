# Package: v1beta1

## Purpose
Provides the v1beta1 versioned API registration for the node.k8s.io API group.

## Key Constants
- `GroupName`: "node.k8s.io"
- `SchemeGroupVersion`: node.k8s.io/v1beta1

## Key Functions
- `Resource(resource string)`: Returns qualified GroupResource
- `AddToScheme`: Registers types with a scheme

## Code Generation Markers
- `+k8s:conversion-gen`: Generates conversion to/from internal node types
- `+k8s:validation-gen`: Generates validation functions

## Notes
- External types sourced from `k8s.io/api/node/v1beta1`
- Beta version; structure matches v1 (flat handler field)
- No custom conversion needed (auto-generated conversions work)
