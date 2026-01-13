# Package: v2beta1

## Purpose
Provides conversion, defaulting, and registration functions for the v2beta1 version of the apidiscovery API.

## Build Tags
- `+k8s:conversion-gen=k8s.io/kubernetes/pkg/apis/apidiscovery` - Generate conversions to internal
- `+k8s:conversion-gen-external-types=k8s.io/api/apidiscovery/v2beta1` - External types location
- `+k8s:defaulter-gen=TypeMeta` - Generate defaulters
- `+groupName=apidiscovery.k8s.io` - API group name

## Generated Files
- `zz_generated.conversion.go` - Conversion functions between v2beta1 and internal
- `zz_generated.defaults.go` - Defaulting functions

## Design Notes
- This is the beta version of the API discovery types (v2 is now stable)
- Types are defined in `k8s.io/api/apidiscovery/v2beta1`
- Provides backward compatibility for clients using the beta API
