# Package: v1alpha1

## Purpose
Provides the v1alpha1 versioned API registration and custom conversions for the node.k8s.io API group.

## Key Constants
- `GroupName`: "node.k8s.io"
- `SchemeGroupVersion`: node.k8s.io/v1alpha1

## Key Functions

### Conversion Functions
Handles structural differences between v1alpha1 and internal types:
- `Convert_v1alpha1_RuntimeClass_To_node_RuntimeClass`: Converts nested spec struct to flat structure (v1alpha1 has `spec.runtimeHandler`, internal uses `handler`)
- `Convert_node_RuntimeClass_To_v1alpha1_RuntimeClass`: Reverse conversion

### Registration
- `Resource(resource string)`: Returns qualified GroupResource
- `AddToScheme`: Registers types with a scheme

## Code Generation Markers
- `+k8s:conversion-gen`: Generates conversion functions
- `+k8s:validation-gen`: Generates validation functions

## Notes
- External types sourced from `k8s.io/api/node/v1alpha1`
- v1alpha1 used a nested `spec` structure that was flattened in later versions
- Deprecated; superseded by v1beta1 and v1
