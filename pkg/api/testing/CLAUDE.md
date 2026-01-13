# Package: testing

## Purpose
Provides test utilities for Kubernetes API types including fuzzer functions for property-based testing, scheme installation, and cross-API validation tests.

## Key Components

### Fuzzer Functions (fuzzer.go)
- `FuzzerFuncs` - Combined list of all fuzzer functions from all API groups
- `overrideGenericFuncs` - Overrides generic fuzzer functions for Kubernetes-specific behavior (runtime.Object, RawExtension)

### Scheme Installation (install.go)
- Blank imports all API group install packages to register types with the legacy scheme
- Includes: admission, admissionregistration, apps, authentication, authorization, autoscaling, batch, certificates, coordination, core, discovery, events, extensions, flowcontrol, imagepolicy, networking, node, policy, rbac, resource, scheduling, storage

### Test Files
- `backward_compatibility_test.go` - Tests for API backward compatibility
- `conversion_test.go` - Tests for type conversions between versions
- `deep_copy_test.go` - Tests for deep copy implementations
- `defaulting_test.go` - Tests for defaulting functions
- `serialization_test.go` / `serialization_proto_test.go` - Tests for JSON/protobuf serialization
- `validation_test.go` - Tests for validation functions
- `unstructured_test.go` - Tests for unstructured object handling

## Design Notes
This package is the central test infrastructure for all Kubernetes API types, ensuring consistent testing patterns across all API groups.
