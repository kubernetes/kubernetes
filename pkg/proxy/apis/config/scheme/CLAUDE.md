# Package: scheme

This package provides the Kubernetes API scheme setup for kube-proxy configuration types. It creates and configures the runtime.Scheme used for encoding/decoding KubeProxyConfiguration objects.

## Key Types

- `Scheme` - The runtime.Scheme instance containing registered kube-proxy config types
- `Codecs` - Serializer factory for encoding/decoding config objects

## Key Functions

- `init()` - Registers kube-proxy configuration types with the scheme and sets up version conversions

## Design Notes

- Uses the standard Kubernetes API machinery pattern for scheme registration
- Imports and registers types from `pkg/proxy/apis/config` (internal) and `v1alpha1` (external)
- Provides default values application via `AddToScheme` from the v1alpha1 package
