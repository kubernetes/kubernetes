# Package: v1

## Purpose
Provides conversion and defaulting functions for the v1 external version of the admission API.

## Build Tags
- `+k8s:conversion-gen=k8s.io/kubernetes/pkg/apis/admission` - Generate conversions to internal
- `+k8s:conversion-gen-external-types=k8s.io/api/admission/v1` - External types location
- `+k8s:defaulter-gen=TypeMeta` - Generate defaulters
- `+groupName=admission.k8s.io` - API group name

## Generated Files
- `zz_generated.conversion.go` - Conversion functions between v1 and internal
- `zz_generated.defaults.go` - Defaulting functions

## Design Notes
- This is the stable version of the admission API
- Types are defined in `k8s.io/api/admission/v1`
- This package contains only the conversion/defaulting infrastructure
