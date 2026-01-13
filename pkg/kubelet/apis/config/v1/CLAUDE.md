# Package: v1

## Purpose
This package provides the v1 (GA) version of the kubelet configuration API. It contains conversion, defaulting, and registration code for the kubelet.config.k8s.io/v1 API version.

## Key Files

- **register.go**: Registers v1 types with the scheme
- **zz_generated.conversion.go**: Auto-generated conversion functions
- **zz_generated.defaults.go**: Auto-generated defaulting functions
- **zz_generated.deepcopy.go**: Auto-generated deep copy functions

## Code Generation Tags

- `+k8s:deepcopy-gen=package`: Generate deep copy for all types
- `+k8s:conversion-gen`: Generate conversion to internal types
- `+k8s:defaulter-gen`: Generate defaulting functions

## Design Notes

- GA version of KubeletConfiguration API
- Converts to/from internal kubeletconfig types
- External type definitions are in k8s.io/kubelet/config/v1
- Part of the kubelet.config.k8s.io API group
