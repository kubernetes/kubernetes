# Package: registry

This is the root package for Kubernetes API server registry implementations. It contains subpackages for each API group that implement REST storage for Kubernetes resources.

## Structure

The registry package is organized by API group:
- `admissionregistration/` - Webhook configs, admission policies
- `apps/` - Deployments, StatefulSets, DaemonSets, ReplicaSets
- `authentication/` - TokenReviews, SelfSubjectReviews
- `authorization/` - SubjectAccessReviews
- `autoscaling/` - HorizontalPodAutoscalers
- `batch/` - Jobs, CronJobs
- `certificates/` - CSRs
- `core/` - Pods, Services, ConfigMaps, Secrets, etc.
- `networking/` - NetworkPolicies, Ingresses
- `rbac/` - Roles, RoleBindings, ClusterRoles
- `storage/` - StorageClasses, VolumeAttachments

## Design Notes

- Each resource has a strategy (validation, defaulting) and storage (etcd backend)
- Uses the apiserver generic registry framework
- Implements REST verbs (GET, LIST, CREATE, UPDATE, DELETE, WATCH)
- Supports subresources (status, scale, etc.)
