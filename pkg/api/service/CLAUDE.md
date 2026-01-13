# Package: service

## Purpose
Provides utilities for Service API objects including load balancer configuration, traffic policy checks, and deprecation warnings.

## Key Functions

### Load Balancer Utilities (util.go)
- `IsAllowAll(ipnets utilnet.IPNetSet) bool` - Checks if IPNet allows traffic from 0.0.0.0/0
- `GetLoadBalancerSourceRanges(service *api.Service) (utilnet.IPNetSet, error)` - Parses LoadBalancerSourceRanges field or annotation, returns default allow-all if unset

### Traffic Policy Checks (util.go)
- `ExternallyAccessible(service *api.Service) bool` - True for LoadBalancer, NodePort, or ClusterIP with ExternalIPs
- `RequestsOnlyLocalTraffic(service *api.Service) bool` - True for LoadBalancer/NodePort with ExternalTrafficPolicy=Local
- `NeedsHealthCheck(service *api.Service) bool` - True for LoadBalancer services requesting local traffic only

### Warnings (warnings.go)
- `GetWarningsForService(service, oldService *api.Service) []string` - Generates warnings for:
  - Deprecated topology-aware hints annotation
  - Invalid IP address formats in clusterIPs, externalIPs, loadBalancerIP
  - Ignored fields on headless services (loadBalancerIP, externalIPs, sessionAffinity)
  - Invalid CIDR formats in loadBalancerSourceRanges
  - Mismatched type/field combinations (externalIPs with ExternalName type)
  - Deprecated trafficDistribution value "PreferClose" (use "PreferSameZone")
