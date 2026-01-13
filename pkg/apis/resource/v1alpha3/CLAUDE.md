# Package: v1alpha3

## Purpose
Provides the v1alpha3 versioned API registration and conversions for the resource.k8s.io API group.

## Key Constants
- `GroupName`: "resource.k8s.io"
- `SchemeGroupVersion`: resource.k8s.io/v1alpha3

## Code Generation Markers
- `+k8s:conversion-gen`: Generates conversion to/from internal resource types
- `+k8s:defaulter-gen`: Generates defaulting functions

## Notes
- External types sourced from `k8s.io/api/resource/v1alpha3`
- Early alpha version; many fields have been removed/tombstoned
- Some types like `Controller` and `SuitableNodes` were removed in later versions
- Maintained for backward compatibility during upgrade
