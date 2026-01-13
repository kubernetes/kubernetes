# Package: v1beta1

## Purpose
Provides the v1beta1 versioned API registration, defaults, and conversions for the resource.k8s.io API group.

## Key Constants
- `GroupName`: "resource.k8s.io"
- `SchemeGroupVersion`: resource.k8s.io/v1beta1

## Structural Differences
v1beta1 uses a flattened DeviceRequest structure:
- v1beta1: `spec.devices.requests[].deviceClassName`
- v1/v1beta2: `spec.devices.requests[].exactly.deviceClassName`

Conversion functions handle this mapping.

## Code Generation Markers
- `+k8s:conversion-gen`: Generates conversion to/from internal resource types
- `+k8s:defaulter-gen`: Generates defaulting functions
- `+k8s:validation-gen`: Generates validation functions

## Notes
- External types sourced from `k8s.io/api/resource/v1beta1`
- Has custom conversion for structural differences with v1
- ValidationNormalizationRules in validation package handle error path mapping
