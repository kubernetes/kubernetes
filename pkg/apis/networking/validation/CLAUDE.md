# Package: validation

## Purpose
Provides comprehensive validation logic for all networking.k8s.io API types including NetworkPolicy, Ingress, IngressClass, IPAddress, and ServiceCIDR.

## Key Types

### NetworkPolicyValidationOptions
Options controlling NetworkPolicy validation behavior:
- `AllowInvalidLabelValueInSelector`: Backward compatibility for invalid label selectors
- `AllowCIDRsEvenIfInvalid`: List of existing CIDRs to allow during updates

### IngressValidationOptions
Options controlling Ingress validation:
- `AllowInvalidSecretName`: Allow invalid TLS secret names (backward compatibility)
- `AllowInvalidWildcardHostRule`: Allow invalid rules with wildcard hosts
- `AllowRelaxedServiceNameValidation`: Use DNS label validation instead of strict RFC 1035

## Key Functions

### NetworkPolicy Validation
- `ValidateNetworkPolicy`: Full validation for NetworkPolicy objects
- `ValidateNetworkPolicyUpdate`: Validates updates, preserving existing invalid CIDRs
- `ValidateNetworkPolicySpec`: Validates pod selectors, ingress/egress rules, policy types
- `ValidateNetworkPolicyPort`: Validates protocol (TCP/UDP/SCTP), port numbers/names, endPort
- `ValidateNetworkPolicyPeer`: Validates peer selectors, ensuring IPBlock exclusivity
- `ValidateIPBlock`: CIDR validation with except range checking

### Ingress Validation
- `ValidateIngressCreate` / `ValidateIngressUpdate`: Validates Ingress on create/update
- `ValidateIngressSpec`: Validates rules, TLS, defaultBackend, ingressClassName
- `validateHTTPIngressPath`: Validates path types (Exact/Prefix/ImplementationSpecific)
- `validateIngressBackend`: Validates service or resource backend references
- `ValidateIngressLoadBalancerStatus`: Validates status IP/hostname fields

### IngressClass Validation
- `ValidateIngressClass`: Validates controller name and parameters
- `ValidateIngressClassUpdate`: Ensures controller field immutability

### IPAddress/ServiceCIDR Validation
- `ValidateIPAddress`: Validates IP address names and parent references
- `ValidateServiceCIDR`: Validates CIDR ranges (max 2, one per IP family)
- `ValidateServiceCIDRUpdate`: Allows adding second CIDR for dual-stack conversion

## Key Constants
- `annotationIngressClass`: "kubernetes.io/ingress.class"
- `maxLenIngressClassController`: 250
- `invalidPathSequences`: ["//", "/./", "/../", "%2f", "%2F"]

## Notes
- Backward compatibility is maintained through validation options
- Feature gate `RelaxedServiceNameValidation` affects service name validation
- Updates preserve invalid values that existed in the old resource
