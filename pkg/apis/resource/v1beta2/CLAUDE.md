# Package: v1beta2

## Purpose
Provides the v1beta2 versioned API registration, defaults, and conversions for the resource.k8s.io API group.

## Key Constants
- `GroupName`: "resource.k8s.io"
- `SchemeGroupVersion`: resource.k8s.io/v1beta2

## Code Generation Markers
- `+k8s:conversion-gen`: Generates conversion to/from internal resource types
- `+k8s:defaulter-gen`: Generates defaulting functions
- `+k8s:validation-gen`: Generates validation functions

## Notes
- External types sourced from `k8s.io/api/resource/v1beta2`
- Structure matches v1 (uses `exactly` field in DeviceRequest)
- Intermediate beta version between v1beta1 and v1
