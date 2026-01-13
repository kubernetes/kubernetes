# Package: serviceaccount

## Purpose
Provides the registry interface and REST strategy implementation for storing ServiceAccount API objects.

## Key Types

- **strategy**: Main strategy for ServiceAccount CRUD operations (namespace-scoped).

## Key Functions

- **Strategy** (var): Default logic for creating/updating ServiceAccounts.
- **NamespaceScoped()**: Returns `true` - ServiceAccounts are namespace-scoped.
- **PrepareForCreate()**: Cleans secret references (preserves only Name).
- **PrepareForUpdate()**: Cleans secret references.
- **Validate()**: Validates ServiceAccount.
- **WarningsOnCreate()**: Warns if EnforceMountableSecretsAnnotation is used (deprecated v1.32+).
- **WarningsOnUpdate()**: Warns if EnforceMountableSecretsAnnotation is newly added.
- **cleanSecretReferences()**: Strips all fields except Name from secret references.

## Design Notes

- Namespace-scoped resource with no subresources.
- Cleans secret ObjectReferences to only include Name field.
- Deprecation warning for kubernetes.io/enforce-mountable-secrets annotation (v1.32+).
- Recommends using separate namespaces instead of mountable secrets annotation.
