# Package: extensions

## Purpose
Internal API types for the extensions API group. This is a legacy/experimental API group that aliases types from other groups (apps, networking, autoscaling).

## Registered Types

Types are aliased from other packages:
- **Deployment, DeploymentList, DeploymentRollback**: From apps package
- **DaemonSet, DaemonSetList**: From apps package
- **ReplicaSet, ReplicaSetList**: From apps package
- **Ingress, IngressList**: From networking package
- **NetworkPolicy, NetworkPolicyList**: From networking package
- **Scale**: From autoscaling package

## Key Functions

- **Kind(kind string)**: Returns Group-qualified GroupKind.
- **Resource(resource string)**: Returns Group-qualified GroupResource.
- **AddToScheme**: Registers aliased types with a scheme.

## Key Constants

- **GroupName**: "extensions"

## Design Notes

- This API group is deprecated; types have moved to dedicated API groups.
- Maintained for backward compatibility with older clients.
- No new types should be added to this group.
- types.go is mostly empty, containing only package documentation.
