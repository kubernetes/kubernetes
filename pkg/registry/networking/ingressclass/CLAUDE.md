# Package: ingressclass

Implements the API server registry strategy for IngressClass resources.

## Key Types

- **ingressClassStrategy**: Implements create/update/delete strategies for IngressClass objects.

## Key Functions

- **PrepareForCreate**: Sets initial generation to 1.
- **PrepareForUpdate**: Increments generation on spec changes.
- **Validate / ValidateUpdate**: Validates using declarative validation with migration checks.

## Design Notes

- IngressClass is a cluster-scoped (non-namespaced) resource.
- Defines which ingress controller should handle Ingresses referencing this class.
- Uses declarative validation with migration checks for create and update operations.
- No status subresource (simple spec-only resource).
