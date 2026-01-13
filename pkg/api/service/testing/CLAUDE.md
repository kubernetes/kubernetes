# Package: testing

## Purpose
Test helper package for constructing Service API objects that pass validation, providing a fluent builder pattern for unit tests.

## Key Types
- `Tweak` - Function type `func(*api.Service)` for modifying Service objects

## Key Functions

### Service Construction
- `MakeService(name string, tweaks ...Tweak) *api.Service` - Creates a valid ClusterIP Service with defaults (single port, trivial selector)
- `MakeServicePort(name string, port int, tgtPort intstr.IntOrString, proto api.Protocol) api.ServicePort` - Creates a ServicePort

### Service Type Setters
- `SetTypeClusterIP(svc *api.Service)` - Sets type to ClusterIP, clears incompatible fields
- `SetTypeNodePort(svc *api.Service)` - Sets type to NodePort with Cluster traffic policy
- `SetTypeLoadBalancer(svc *api.Service)` - Sets type to LoadBalancer with node port allocation
- `SetTypeExternalName(svc *api.Service)` - Sets type to ExternalName, clears ClusterIP fields
- `SetHeadless(svc *api.Service)` - Sets as headless service (ClusterIP=None)

### Configuration Tweaks
- `SetPorts(ports ...api.ServicePort) Tweak` - Sets the ports list
- `SetSelector(sel map[string]string) Tweak` - Sets the selector
- `SetClusterIP(ip string) Tweak` / `SetClusterIPs(ips ...string) Tweak` - Sets cluster IPs
- `SetIPFamilies(families ...api.IPFamily) Tweak` - Sets IP families
- `SetIPFamilyPolicy(policy api.IPFamilyPolicy) Tweak` - Sets IP family policy
- `SetNodePorts(values ...int) Tweak` - Sets node port values
- `SetInternalTrafficPolicy` / `SetExternalTrafficPolicy` - Sets traffic policies
- `SetHealthCheckNodePort`, `SetSessionAffinity`, `SetExternalName` - Other configurations

## Design Pattern
Each type setter clears incompatible fields to maintain valid Service state.
