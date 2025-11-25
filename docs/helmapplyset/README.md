# Helm ApplySet Integration

This document describes the Helm ApplySet Integration features added to Kubernetes, enabling seamless interoperability between Helm and kubectl through the ApplySet specification (KEP-3659).

## Overview

Helm is the de facto package manager for Kubernetes, but historically Kubernetes hasn't recognized Helm releases as first-class resources. This integration bridges that gap by:

1. **Feature 1: Helm ApplySet Adapter Controller** - Automatically creates ApplySet metadata for Helm releases
2. **Feature 2: Helm Release Status Aggregator** - Provides aggregated health status, events, and metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Helm ApplySet Controller                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Watcher    │───▶│    Parent    │───▶│   Labeler    │                  │
│  │              │    │   Manager    │    │              │                  │
│  │ Watches Helm │    │              │    │ Labels all   │                  │
│  │ release      │    │ Creates      │    │ resources    │                  │
│  │ Secrets      │    │ ApplySet     │    │ with         │                  │
│  │              │    │ parent       │    │ ApplySet ID  │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Main Controller Loop                             │   │
│  │  - Reconciles Helm releases with ApplySet metadata                   │   │
│  │  - Handles create/update/delete events                               │   │
│  │  - Implements exponential backoff on errors                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                        Status Aggregator                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  Aggregator  │───▶│  Condition   │───▶│   Events     │                  │
│  │              │    │   Manager    │    │  Generator   │                  │
│  │ Computes     │    │              │    │              │                  │
│  │ health from  │    │ Sets Ready,  │    │ Emits K8s    │                  │
│  │ all resources│    │ Progressing, │    │ events on    │                  │
│  │              │    │ Degraded     │    │ transitions  │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                                       │                           │
│         ▼                                       ▼                           │
│  ┌──────────────┐                        ┌──────────────┐                  │
│  │   Metrics    │                        │  Admission   │                  │
│  │   Exporter   │                        │   Webhook    │                  │
│  │              │                        │              │                  │
│  │ Prometheus   │                        │ Validates    │                  │
│  │ metrics      │                        │ ApplySet     │                  │
│  │ endpoint     │                        │ metadata     │                  │
│  └──────────────┘                        └──────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Feature 1: Helm ApplySet Adapter Controller

### Components

| Component | Location | Description |
|-----------|----------|-------------|
| Watcher | `pkg/controller/helmapplyset/watcher.go` | Watches Helm release Secrets (`sh.helm.release.v1.*`) |
| Parent Manager | `pkg/controller/helmapplyset/parent/manager.go` | Creates/updates ApplySet parent Secrets |
| Labeler | `pkg/controller/helmapplyset/labeler/labeler.go` | Adds ApplySet labels to managed resources |
| Controller | `pkg/controller/helmapplyset/controller.go` | Main reconciliation loop |
| Webhook | `pkg/admission/helmapplyset/webhook.go` | Validates ApplySet metadata |

### How It Works

1. **Detection**: The watcher monitors for Secrets with type `helm.sh/release.v1` and name prefix `sh.helm.release.v1.`

2. **Parsing**: When a Helm release Secret is detected, it:
   - Base64 decodes the `release` data field
   - Decompresses the gzip content
   - Unmarshals the Helm release JSON

3. **ApplySet Creation**: The parent manager creates an ApplySet parent Secret with:
   ```yaml
   labels:
     applyset.kubernetes.io/id: applyset-<hash>-v1
   annotations:
     applyset.kubernetes.io/tooling: helm/v3
     applyset.kubernetes.io/contains-group-kinds: Deployment.apps,Service,ConfigMap
   ```

4. **Resource Labeling**: All resources in the Helm manifest are labeled with:
   ```yaml
   labels:
     applyset.kubernetes.io/part-of: applyset-<hash>-v1
   ```

### ApplySet ID Computation

The ApplySet ID is computed as:
```
applyset-<base64url(sha256(releaseName + "/" + namespace))>-v1
```

This ensures:
- Deterministic IDs for the same release
- Unique IDs across different releases/namespaces
- Compliance with KEP-3659 format

## Feature 2: Helm Release Status Aggregator

### Components

| Component | Location | Description |
|-----------|----------|-------------|
| Aggregator | `pkg/controller/helmapplyset/status/aggregator.go` | Computes aggregate health |
| Conditions | `pkg/controller/helmapplyset/status/conditions.go` | Manages status conditions |
| Events | `pkg/controller/helmapplyset/events/generator.go` | Generates Kubernetes events |
| Metrics | `pkg/controller/helmapplyset/metrics/exporter.go` | Exports Prometheus metrics |

### Health Computation

The aggregator checks health for each resource type:

| Resource Type | Health Check |
|---------------|--------------|
| Deployment | `spec.replicas == status.readyReplicas` |
| StatefulSet | `spec.replicas == status.readyReplicas` |
| DaemonSet | `status.desiredNumberScheduled == status.numberReady` |
| Service | Endpoints exist and have ready addresses |
| PVC | `status.phase == "Bound"` |
| Job | `status.succeeded >= spec.completions` |

### Status Conditions

The controller sets standard Kubernetes conditions:

| Condition | Meaning |
|-----------|---------|
| `Ready` | All resources are healthy |
| `Progressing` | Rollout in progress |
| `Degraded` | Some resources are unhealthy |
| `Available` | Minimum availability met |

### Prometheus Metrics

```
# Gauge: Current status of the ApplySet
helm_applyset_status{name="my-app", namespace="default", status="healthy"} 1

# Gauge: Number of resources by GroupKind
helm_applyset_resource_total{name="my-app", namespace="default", gvk="apps/v1/Deployment"} 2

# Gauge: Number of healthy resources
helm_applyset_resource_healthy{name="my-app", namespace="default", gvk="apps/v1/Deployment"} 2

# Gauge: Age of the ApplySet in seconds
helm_applyset_age_seconds{name="my-app", namespace="default"} 3600
```

## kubectl Integration

### Commands

```bash
# List all ApplySets (including Helm releases)
kubectl get applysets

# Filter by Helm releases only
kubectl get applysets -l applyset.kubernetes.io/tooling=helm/v3

# Describe an ApplySet
kubectl describe applyset applyset-my-app

# Get detailed status
kubectl helmapplyset status my-app
```

### Example Output

```
$ kubectl helmapplyset status my-app
NAME: my-app
NAMESPACE: default
STATUS: Healthy

RESOURCES:
  KIND         NAME              STATUS    READY   MESSAGE
  Deployment   my-app-nginx      Healthy   3/3     All replicas ready
  Service      my-app-nginx      Healthy   1       Endpoints available
  ConfigMap    my-app-config     Healthy   -
  Secret       my-app-secret     Healthy   -

CONDITIONS:
  TYPE          STATUS   REASON                MESSAGE
  Ready         True     AllResourcesHealthy   All 4 resources are healthy
  Progressing   False    -
  Degraded      False    -
```

## Testing

### Running Tests

```bash
# Run all HelmApplySet tests
go test ./pkg/controller/helmapplyset/... -v

# Run admission webhook tests
go test ./pkg/admission/helmapplyset/... -v

# Run with coverage
go test ./pkg/controller/helmapplyset/... -coverprofile=coverage.out
go tool cover -func=coverage.out
```

### Test Coverage

The test suite covers:
- Watcher: Secret detection, parsing, event handling
- Parent Manager: ApplySet ID computation, Secret creation/update
- Labeler: Resource discovery, label patching
- Controller: Full reconciliation loop, error handling
- Status: Health aggregation, condition management
- Events: Event generation for state transitions
- Metrics: Prometheus metric updates
- Webhook: Validation and mutation logic

## Configuration

### RBAC Requirements

The controller requires these permissions:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: helm-applyset-controller
rules:
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get", "list", "watch", "create", "update", "delete"]
- apiGroups: [""]
  resources: ["events"]
  verbs: ["create", "patch"]
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["get", "list", "patch"]  # For labeling resources
```

## Related Documentation

- [KEP-3659: ApplySet Specification](https://github.com/kubernetes/enhancements/blob/master/keps/sig-cli/3659-kubectl-apply-prune/README.md)
- [Helm Release Storage](https://helm.sh/docs/topics/advanced/#storage-backends)
- [Kubernetes Controller Patterns](https://kubernetes.io/docs/concepts/architecture/controller/)
