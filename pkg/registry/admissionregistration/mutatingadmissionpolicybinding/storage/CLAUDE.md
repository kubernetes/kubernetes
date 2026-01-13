# Package: storage

This package provides the REST storage implementation for MutatingAdmissionPolicyBinding resources, backed by etcd.

## Key Types

- `REST` - Implements RESTStorage for MutatingAdmissionPolicyBinding
- `PolicyGetter` - Interface for retrieving MutatingAdmissionPolicy by name
- `DefaultPolicyGetter` - Default implementation using rest.Getter

## Key Functions

- `NewREST()` - Creates REST storage with authorizer, policy getter, and resource resolver
- `Categories()` - Returns ["api-extensions"] for kubectl grouping

## Design Notes

- Uses generic registry Store for CRUD operations
- Resource name: "mutatingadmissionpolicybindings"
- Singular: "mutatingadmissionpolicybinding"
- PolicyGetter enables validation that referenced policy exists
- Part of admissionregistration.k8s.io API group
