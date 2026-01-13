# Package: core

This package provides resource quota evaluators for core Kubernetes resources (Pods, Services, PVCs, ResourceClaims).

## Key Types

- `podEvaluator` - Evaluates pod resource usage (CPU, memory, ephemeral storage, pod count)
- `serviceEvaluator` - Evaluates service quotas (LoadBalancer, NodePort counts)
- `pvcEvaluator` - Evaluates PersistentVolumeClaim storage quotas
- `resourceClaimEvaluator` - Evaluates DRA ResourceClaim quotas

## Key Functions

- `NewEvaluators()` - Returns all core resource quota evaluators
- `NewPodEvaluator()` - Creates a pod quota evaluator
- `NewServiceEvaluator()` - Creates a service quota evaluator
- `NewPersistentVolumeClaimEvaluator()` - Creates a PVC quota evaluator

## Design Notes

- Pods are only counted against quota if they are not terminal (Succeeded/Failed)
- CPU/memory quotas require containers to specify requests/limits (legacy behavior)
- Supports extended resources and hugepages quotas
- Service quotas can limit LoadBalancers and NodePorts per namespace
- PVC quotas support storage class-specific limits
- Integrates with DRA for dynamic resource allocation quotas
