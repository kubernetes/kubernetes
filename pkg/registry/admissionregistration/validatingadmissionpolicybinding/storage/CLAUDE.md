# Package: storage

This package provides REST storage for ValidatingAdmissionPolicyBinding resources.

## Key Types

- `REST` - RESTStorage implementation for bindings
- `PolicyGetter` - Interface for retrieving policies by name
- `DefaultPolicyGetter` - Default implementation wrapping rest.Getter

## Key Functions

- `NewREST()` - Creates storage with authorizer, policy getter, resolver
- `Categories()` - Returns ["api-extensions"]
- `GetValidatingAdmissionPolicy()` - DefaultPolicyGetter method

## Design Notes

- Uses generic registry Store for CRUD
- Resource: "validatingadmissionpolicybindings"
- Singular: "validatingadmissionpolicybinding"
- PolicyGetter validates referenced policy exists
- Strategy handles paramRef authorization
- Part of admissionregistration.k8s.io API group
