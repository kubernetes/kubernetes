# Package: v1

## Purpose
Provides the v1 versioned API registration, defaults, and conversions for the networking.k8s.io API group.

## Key Constants
- `GroupName`: "networking.k8s.io"
- `SchemeGroupVersion`: networking.k8s.io/v1

## Key Functions

### Defaulting Functions
- `SetDefaults_NetworkPolicyPort`: Defaults Protocol to TCP if not specified
- `SetDefaults_NetworkPolicy`: Sets PolicyTypes to ["Ingress"] if empty; adds "Egress" if egress rules exist
- `SetDefaults_IngressClass`: Defaults Parameters.Scope to "Cluster" if Parameters exists but Scope is nil

### Registration
- `Resource(resource string)`: Returns qualified GroupResource
- `AddToScheme`: Registers types with a scheme

## Code Generation Markers
- `+k8s:conversion-gen`: Generates conversion to/from internal networking and extensions types
- `+k8s:defaulter-gen`: Generates defaulting functions
- `+k8s:validation-gen`: Generates validation functions

## Notes
- External types sourced from `k8s.io/api/networking/v1`
- Defaulting ensures API objects have sensible values when fields are omitted
