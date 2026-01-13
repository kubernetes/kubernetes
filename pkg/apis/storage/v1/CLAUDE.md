# Package: storage/v1

## Purpose
Provides versioned (v1) API types, conversion, defaulting, and validation for the storage.k8s.io API group.

## Key Files
- `register.go`: Registers v1 types with the scheme
- `defaults.go`: Provides default values for storage types
- `zz_generated.conversion.go`: Auto-generated conversion functions between internal and v1 types
- `zz_generated.defaults.go`: Auto-generated defaulting functions
- `zz_generated.validations.go`: Auto-generated validation functions

## Key Defaults
- CSIDriver.Spec.AttachRequired defaults to true
- CSIDriver.Spec.PodInfoOnMount defaults to false
- CSIDriver.Spec.VolumeLifecycleModes defaults to ["Persistent"]

## Design Notes
- This is the stable (GA) version of the storage API
- Follows Kubernetes versioned API package conventions
- Conversion and defaulting are primarily auto-generated from internal types
