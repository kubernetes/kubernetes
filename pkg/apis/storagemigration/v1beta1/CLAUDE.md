# Package: storagemigration/v1beta1

## Purpose
Provides versioned (v1beta1) API types and conversion logic for the storagemigration.k8s.io API group.

## Key Files
- `register.go`: Registers v1beta1 types with the scheme
- `doc.go`: Package documentation and code generation directives
- `zz_generated.conversion.go`: Auto-generated conversion functions

## Design Notes
- This is the beta version of the storage migration API
- Contains StorageVersionMigration type for managing data migrations
- Follows Kubernetes versioned API package conventions
- Conversion between internal and v1beta1 types is auto-generated
