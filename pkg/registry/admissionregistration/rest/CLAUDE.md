# Package: rest

This package provides the REST storage provider for the admissionregistration.k8s.io API group, registering all admission-related resources.

## Key Types

- `RESTStorageProvider` - Implements genericapiserver.RESTStorageProvider for admission resources

## Key Functions

- `NewRESTStorage()` - Creates storage for all admission registration resources
- `v1Storage()` - Returns v1 API version storage (webhooks, validating policies)
- `v1alpha1Storage()` - Returns v1alpha1 storage (mutating admission policies)
- `v1beta1Storage()` - Returns v1beta1 storage (policies in beta)
- `GroupName()` - Returns "admissionregistration.k8s.io"

## Resources Registered

- `validatingwebhookconfigurations` - External validation webhooks
- `mutatingwebhookconfigurations` - External mutation webhooks
- `validatingadmissionpolicies` - CEL-based validation policies
- `validatingadmissionpolicybindings` - Policy bindings
- `mutatingadmissionpolicies` - CEL-based mutation policies (alpha)
- `mutatingadmissionpolicybindings` - Mutation policy bindings

## Design Notes

- Each API version has its own storage map
- Uses discovery client for GVK to GVR resolution
- Requires authorizer for policy paramKind authorization
