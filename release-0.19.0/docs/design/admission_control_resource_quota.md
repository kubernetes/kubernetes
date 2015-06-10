# Admission control plugin: ResourceQuota

## Background

This document proposes a system for enforcing hard resource usage limits per namespace as part of admission control.

## Model Changes

A new resource, **ResourceQuota**, is introduced to enumerate hard resource limits in a Kubernetes namespace.

A new resource, **ResourceQuotaUsage**, is introduced to support atomic updates of a **ResourceQuota** status.

```go
// The following identify resource constants for Kubernetes object types
const (
  // Pods, number
  ResourcePods ResourceName = "pods"
  // Services, number
  ResourceServices ResourceName = "services"
  // ReplicationControllers, number
  ResourceReplicationControllers ResourceName = "replicationcontrollers"
  // ResourceQuotas, number
  ResourceQuotas ResourceName = "resourcequotas"
)

// ResourceQuotaSpec defines the desired hard limits to enforce for Quota
type ResourceQuotaSpec struct {
  // Hard is the set of desired hard limits for each named resource
  Hard ResourceList `json:"hard,omitempty"`
}

// ResourceQuotaStatus defines the enforced hard limits and observed use
type ResourceQuotaStatus struct {
  // Hard is the set of enforced hard limits for each named resource
  Hard ResourceList `json:"hard,omitempty"`
  // Used is the current observed total usage of the resource in the namespace
  Used ResourceList `json:"used,omitempty"`
}

// ResourceQuota sets aggregate quota restrictions enforced per namespace
type ResourceQuota struct {
  TypeMeta   `json:",inline"`
  ObjectMeta `json:"metadata,omitempty"`

  // Spec defines the desired quota
  Spec ResourceQuotaSpec `json:"spec,omitempty"`

  // Status defines the actual enforced quota and its current usage
  Status ResourceQuotaStatus `json:"status,omitempty"`
}

// ResourceQuotaUsage captures system observed quota status per namespace
// It is used to enforce atomic updates of a backing ResourceQuota.Status field in storage
type ResourceQuotaUsage struct {
  TypeMeta   `json:",inline"`
  ObjectMeta `json:"metadata,omitempty"`

  // Status defines the actual enforced quota and its current usage
  Status ResourceQuotaStatus `json:"status,omitempty"`
}

// ResourceQuotaList is a list of ResourceQuota items
type ResourceQuotaList struct {
  TypeMeta `json:",inline"`
  ListMeta `json:"metadata,omitempty"`

  // Items is a list of ResourceQuota objects
  Items []ResourceQuota `json:"items"`
}

```

## AdmissionControl plugin: ResourceQuota

The **ResourceQuota** plug-in introspects all incoming admission requests. 

It makes decisions by evaluating the incoming object against all defined **ResourceQuota.Status.Hard** resource limits in the request
namespace.  If acceptance of the resource would cause the total usage of a named resource to exceed its hard limit, the request is denied.

The following resource limits are imposed as part of core Kubernetes at the namespace level:

| ResourceName | Description |
| ------------ | ----------- |
| cpu | Total cpu usage |
| memory | Total memory usage |
| pods | Total number of pods  |
| services | Total number of services |
| replicationcontrollers | Total number of replication controllers |
| resourcequotas | Total number of resource quotas |

Any resource that is not part of core Kubernetes must follow the resource naming convention prescribed by Kubernetes.

This means the resource must have a fully-qualified name (i.e. mycompany.org/shinynewresource)

If the incoming request does not cause the total usage to exceed any of the enumerated hard resource limits, the plug-in will post a
**ResourceQuotaUsage** document to the server to atomically update the observed usage based on the previously read 
**ResourceQuota.ResourceVersion**.  This keeps incremental usage atomically consistent, but does introduce a bottleneck (intentionally)
into the system.

To optimize system performance, it is encouraged that all resource quotas are tracked on the same **ResourceQuota** document.  As a result,
its encouraged to actually impose a cap on the total number of individual quotas that are tracked in the **Namespace** to 1 by explicitly
capping it in **ResourceQuota** document.

## kube-apiserver

The server is updated to be aware of **ResourceQuota** objects.

The quota is only enforced if the kube-apiserver is started as follows:

```
$ kube-apiserver -admission_control=ResourceQuota
```

## kube-controller-manager

A new controller is defined that runs a synch loop to calculate quota usage across the namespace.

**ResourceQuota** usage is only calculated if a namespace has a **ResourceQuota** object.

If the observed usage is different than the recorded usage, the controller sends a **ResourceQuotaUsage** resource
to the server to atomically update.

The synchronization loop frequency will control how quickly DELETE actions are recorded in the system and usage is ticked down.

To optimize the synchronization loop, this controller will WATCH on Pod resources to track DELETE events, and in response, recalculate
usage.  This is because a Pod deletion will have the most impact on observed cpu and memory usage in the system, and we anticipate
this being the resource most closely running at the prescribed quota limits.

## kubectl

kubectl is modified to support the **ResourceQuota** resource.

```kubectl describe``` provides a human-readable output of quota.

For example,

```
$ kubectl namespace myspace
$ kubectl create -f examples/resourcequota/resource-quota.json
$ kubectl get quota
NAME
quota
$ kubectl describe quota quota
Name:                   quota
Resource                Used    Hard
--------                ----    ----
cpu                     0m      20
memory                  0       1Gi
pods                    5       10
replicationcontrollers  5       20
resourcequotas          1       1
services                3       5
```


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/admission_control_resource_quota.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/docs/design/admission_control_resource_quota.md?pixel)]()
