# Package: networkpolicy

Implements the API server registry strategy for NetworkPolicy resources.

## Key Types

- **networkPolicyStrategy**: Implements create/update/delete strategies for NetworkPolicy objects.

## Key Functions

- **PrepareForCreate**: Sets initial generation to 1.
- **PrepareForUpdate**: Increments generation on spec changes using reflect.DeepEqual.
- **Validate / ValidateUpdate**: Validates using version-aware validation options.
- **WarningsOnCreate / WarningsOnUpdate**: Warns about malformed CIDR values in IPBlock rules.
- **networkPolicyWarnings**: Helper that checks ingress/egress IPBlock CIDR formats.

## Design Notes

- NetworkPolicies are namespace-scoped resources.
- Generates warnings for potentially problematic CIDR formats in both ingress and egress rules.
- Uses `ValidationOptionsForNetworking` for version-aware validation.
- No status subresource (spec-only resource).
