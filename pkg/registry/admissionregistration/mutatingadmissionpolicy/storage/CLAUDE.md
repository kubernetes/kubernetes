# Package: storage

This package provides the REST storage implementation for MutatingAdmissionPolicy resources, backed by etcd.

## Key Types

- `REST` - Implements RESTStorage for MutatingAdmissionPolicy against etcd

## Key Functions

- `NewREST()` - Creates a new REST storage with the specified options, authorizer, and resource resolver
- `Categories()` - Returns ["api-extensions"] for kubectl category grouping

## Design Notes

- Uses the generic registry Store for standard CRUD operations
- Resource name: "mutatingadmissionpolicies" (plural)
- Singular name: "mutatingadmissionpolicy"
- Strategy handles validation, defaulting, and authorization
- Supports table printing via TableConvertor
- Part of the admissionregistration.k8s.io API group
