# Package: storage

This package provides the REST storage implementation for MutatingWebhookConfiguration resources, backed by etcd.

## Key Types

- `REST` - Implements RESTStorage for MutatingWebhookConfiguration

## Key Functions

- `NewREST()` - Creates a new REST storage with the specified options
- `Categories()` - Returns ["api-extensions"] for kubectl category grouping

## Design Notes

- Uses generic registry Store for CRUD operations
- Resource name: "mutatingwebhookconfigurations"
- Singular: "mutatingwebhookconfiguration"
- Uses the mutatingwebhookconfiguration.Strategy for validation
- Supports table printing via TableConvertor
- Part of admissionregistration.k8s.io/v1 API
