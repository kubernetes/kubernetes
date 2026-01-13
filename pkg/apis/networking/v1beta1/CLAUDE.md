# Package: v1beta1

## Purpose
Provides the v1beta1 versioned API registration, defaults, and conversions for the networking.k8s.io API group, supporting backward compatibility with older Ingress resources.

## Key Constants
- `GroupName`: "networking.k8s.io"
- `SchemeGroupVersion`: networking.k8s.io/v1beta1

## Key Functions

### Conversion Functions
Handles differences between v1beta1 and internal Ingress representations:
- `Convert_v1beta1_IngressBackend_To_networking_IngressBackend`: Converts legacy ServiceName/ServicePort to new Service struct format
- `Convert_networking_IngressBackend_To_v1beta1_IngressBackend`: Reverse conversion for v1beta1 output
- `Convert_v1beta1_IngressSpec_To_networking_IngressSpec`: Handles Backend to DefaultBackend field mapping
- `Convert_networking_IngressSpec_To_v1beta1_IngressSpec`: Reverse conversion

### Defaulting Functions
- `SetDefaults_HTTPIngressPath`: Defaults PathType to "ImplementationSpecific" if not specified

### Registration
- `Resource(resource string)`: Returns qualified GroupResource
- `AddToScheme`: Registers types with a scheme

## Code Generation Markers
- `+k8s:conversion-gen`: Generates conversion to/from internal networking and extensions types
- `+k8s:defaulter-gen`: Generates defaulting functions
- `+k8s:validation-gen`: Generates validation functions

## Notes
- Exists primarily for backward compatibility with older Ingress resources
- v1beta1 used ServiceName/ServicePort directly; internal version uses nested Service struct
- External types sourced from `k8s.io/api/networking/v1beta1`
