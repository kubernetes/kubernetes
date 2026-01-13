# Package: storage

This package provides REST storage for ValidatingWebhookConfiguration resources.

## Key Types

- `REST` - RESTStorage implementation backed by etcd

## Key Functions

- `NewREST()` - Creates storage with specified options
- `Categories()` - Returns ["api-extensions"]

## Design Notes

- Uses generic registry Store for CRUD operations
- Resource: "validatingwebhookconfigurations"
- Singular: "validatingwebhookconfiguration"
- Uses validatingwebhookconfiguration.Strategy
- Supports table printing via TableConvertor
- Part of admissionregistration.k8s.io/v1 API
