# Package: service

## Purpose
Provides utilities for v1.Service objects including load balancer configuration parsing and traffic policy checks. This is the v1 API version of the service utilities.

## Key Functions

### Load Balancer Utilities
- `IsAllowAll(ipnets utilnet.IPNetSet) bool` - Checks if IPNet allows traffic from 0.0.0.0/0
- `GetLoadBalancerSourceRanges(service *v1.Service) (utilnet.IPNetSet, error)` - Parses LoadBalancerSourceRanges field or annotation, returns default allow-all if unset

### Traffic Policy Checks
- `ExternallyAccessible(service *v1.Service) bool` - True for LoadBalancer, NodePort, or ClusterIP with ExternalIPs
- `ExternalPolicyLocal(service *v1.Service) bool` - True if externally accessible and ExternalTrafficPolicy=Local
- `InternalPolicyLocal(service *v1.Service) bool` - True if InternalTrafficPolicy=Local
- `NeedsHealthCheck(service *v1.Service) bool` - True for LoadBalancer services with ExternalPolicyLocal

## Constants
- `defaultLoadBalancerSourceRanges` = "0.0.0.0/0" - Default allow-all CIDR

## Design Notes
- GetLoadBalancerSourceRanges prefers the spec field over the annotation
- Traffic policy functions are used by kube-proxy and cloud providers for routing decisions
