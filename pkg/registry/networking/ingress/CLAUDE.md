# Package: ingress

Implements the API server registry strategy for Ingress resources in the `networking.k8s.io` API group.

## Key Types

- **ingressStrategy**: Implements create/update/delete strategies for Ingress objects.
- **ingressStatusStrategy**: Separate strategy for status subresource updates.

## Key Functions

- **PrepareForCreate**: Clears status and sets generation to 1.
- **PrepareForUpdate**: Preserves status, increments generation on spec changes.
- **Validate / ValidateUpdate**: Validates Ingress objects using networking validation.
- **WarningsOnCreate**: Warns about deprecated `kubernetes.io/ingress.class` annotation.
- **GetResetFields**: Defines reset fields for extensions/v1beta1, networking/v1beta1, and networking/v1.

## Design Notes

- Ingresses are namespace-scoped resources.
- The deprecated `kubernetes.io/ingress.class` annotation triggers a warning; use `spec.ingressClassName` instead.
- Status updates validate and warn about IP address formats in loadBalancer.ingress.
- Supports multiple API versions with appropriate field reset configuration.
