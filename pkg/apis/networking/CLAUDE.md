# Package: networking

## Purpose
Defines the internal (unversioned) API types for the networking.k8s.io API group, covering network policies, ingress, and IP address management.

## Key Types

### NetworkPolicy
Controls network traffic flow to/from pods using label selectors and rules.
- `NetworkPolicySpec`: Defines pod selector, ingress/egress rules, and policy types
- `NetworkPolicyIngressRule` / `NetworkPolicyEgressRule`: Define allowed traffic patterns
- `NetworkPolicyPort`: Specifies protocol, port, and optional port range
- `NetworkPolicyPeer`: Identifies traffic sources/destinations (pods, namespaces, IP blocks)
- `IPBlock`: CIDR-based IP address filtering

### Ingress
Manages external HTTP(S) access to cluster services.
- `IngressSpec`: Defines class, default backend, TLS config, and routing rules
- `IngressRule`: Maps hosts to backend services
- `IngressBackend`: References a Service or custom resource
- `IngressClass`: Defines which controller handles the Ingress
- `IngressTLS`: TLS certificate configuration

### IP Address Management
- `IPAddress`: Represents a single allocated IP address with parent reference
- `ServiceCIDR`: Defines IP ranges for ClusterIP allocation

## Key Constants
- `PolicyTypeIngress`, `PolicyTypeEgress`: Network policy directions
- `PathTypeExact`, `PathTypePrefix`, `PathTypeImplementationSpecific`: Ingress path matching modes

## Key Functions
- `Kind(kind string)`: Returns qualified GroupKind
- `Resource(resource string)`: Returns qualified GroupResource
- `AddToScheme`: Registers all types with a scheme

## Notes
- Uses `runtime.APIVersionInternal` for internal version
- All list types include standard `metav1.TypeMeta` and `metav1.ListMeta`
