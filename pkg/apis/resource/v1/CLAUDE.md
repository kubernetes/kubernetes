# Package: v1

## Purpose
Provides the v1 (stable) versioned API registration, defaults, and conversions for the resource.k8s.io API group.

## Key Constants
- `GroupName`: "resource.k8s.io"
- `SchemeGroupVersion`: resource.k8s.io/v1

## Key Defaulting Functions
- `SetDefaults_ExactDeviceRequest`: Defaults AllocationMode to ExactCount, Count to 1
- `SetDefaults_DeviceSubRequest`: Same defaults for subrequests
- `SetDefaults_DeviceTaint`: Sets TimeAdded to current time if nil

## Key Functions
- `Resource(resource string)`: Returns qualified GroupResource
- `AddToScheme`: Registers types with a scheme
- `addDefaultingFuncs`, `addConversionFuncs`: Register defaults and conversions

## Code Generation Markers
- `+k8s:conversion-gen`: Generates conversion to/from internal resource types
- `+k8s:defaulter-gen`: Generates defaulting functions
- `+k8s:validation-gen`: Generates validation functions

## Notes
- External types sourced from `k8s.io/api/resource/v1`
- GA (stable) version of the DRA API
- Uses structured DeviceRequest with `exactly` field
