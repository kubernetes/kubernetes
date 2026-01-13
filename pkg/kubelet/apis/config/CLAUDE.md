# Package: config

## Purpose
This package defines the internal (unversioned) API types for kubelet configuration. It contains the KubeletConfiguration struct and related types used to configure the kubelet component.

## Key Types

- **KubeletConfiguration**: Main configuration struct for the kubelet
- **KubeletAuthentication**: Settings for kubelet server authentication
- **KubeletAuthorization**: Settings for kubelet server authorization
- **KubeletAuthorizationMode**: Enum for authorization modes (AlwaysAllow, Webhook)
- **HairpinMode**: Enum for hairpin packet handling modes
- **ResourceChangeDetectionStrategy**: Enum for secret/configmap change detection

## Key Configuration Areas

- Server and TLS settings (ports, certificates, cipher suites)
- Authentication and authorization configuration
- Resource management (cgroups, CPU manager, memory manager, topology manager)
- Eviction thresholds and policies
- Image garbage collection settings
- Logging and debugging options
- Node allocatable resources

## Design Notes

- Internal type - not versioned, not directly serialized
- External versions (v1, v1alpha1, v1beta1) convert to/from this type
- Uses +k8s:deepcopy-gen=package for automatic deep copy generation
- Part of the kubelet.config.k8s.io API group
