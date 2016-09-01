<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Admission control plugin: ResourceQuota

## Background

This document describes a system for enforcing hard resource usage limits per
namespace as part of admission control.

## Use cases

1. Ability to enumerate resource usage limits per namespace.
2. Ability to monitor resource usage for tracked resources.
3. Ability to reject resource usage exceeding hard quotas.

## Data Model

The **ResourceQuota** object is scoped to a **Namespace**.

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
  // ResourceSecrets, number
  ResourceSecrets ResourceName = "secrets"
  // ResourcePersistentVolumeClaims, number
  ResourcePersistentVolumeClaims ResourceName = "persistentvolumeclaims"
)

// ResourceQuotaSpec defines the desired hard limits to enforce for Quota
type ResourceQuotaSpec struct {
  // Hard is the set of desired hard limits for each named resource
  Hard ResourceList `json:"hard,omitempty" description:"hard is the set of desired hard limits for each named resource; see http://releases.k8s.io/release-1.4/docs/design/admission_control_resource_quota.md#admissioncontrol-plugin-resourcequota"`
}

// ResourceQuotaStatus defines the enforced hard limits and observed use
type ResourceQuotaStatus struct {
  // Hard is the set of enforced hard limits for each named resource
  Hard ResourceList `json:"hard,omitempty" description:"hard is the set of enforced hard limits for each named resource; see http://releases.k8s.io/release-1.4/docs/design/admission_control_resource_quota.md#admissioncontrol-plugin-resourcequota"`
  // Used is the current observed total usage of the resource in the namespace
  Used ResourceList `json:"used,omitempty" description:"used is the current observed total usage of the resource in the namespace"`
}

// ResourceQuota sets aggregate quota restrictions enforced per namespace
type ResourceQuota struct {
  TypeMeta   `json:",inline"`
  ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/release-1.4/docs/devel/api-conventions.md#metadata"`

  // Spec defines the desired quota
  Spec ResourceQuotaSpec `json:"spec,omitempty" description:"spec defines the desired quota; http://releases.k8s.io/release-1.4/docs/devel/api-conventions.md#spec-and-status"`

  // Status defines the actual enforced quota and its current usage
  Status ResourceQuotaStatus `json:"status,omitempty" description:"status defines the actual enforced quota and current usage; http://releases.k8s.io/release-1.4/docs/devel/api-conventions.md#spec-and-status"`
}

// ResourceQuotaList is a list of ResourceQuota items
type ResourceQuotaList struct {
  TypeMeta `json:",inline"`
  ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/release-1.4/docs/devel/api-conventions.md#metadata"`

  // Items is a list of ResourceQuota objects
  Items []ResourceQuota `json:"items" description:"items is a list of ResourceQuota objects; see http://releases.k8s.io/release-1.4/docs/design/admission_control_resource_quota.md#admissioncontrol-plugin-resourcequota"`
}
```

## Quota Tracked Resources

The following resources are supported by the quota system:

| Resource | Description |
| ------------ | ----------- |
| cpu | Total requested cpu usage |
| memory | Total requested memory usage |
| pods | Total number of active pods where phase is pending or active.  |
| services | Total number of services |
| replicationcontrollers | Total number of replication controllers |
| resourcequotas | Total number of resource quotas |
| secrets | Total number of secrets |
| persistentvolumeclaims | Total number of persistent volume claims |

If a third-party wants to track additional resources, it must follow the
resource naming conventions prescribed by Kubernetes. This means the resource
must have a fully-qualified name (i.e. mycompany.org/shinynewresource)

## Resource Requirements: Requests vs Limits

If a resource supports the ability to distinguish between a request and a limit
for a resource, the quota tracking system will only cost the request value
against the quota usage. If a resource is tracked by quota, and no request value
is provided, the associated entity is rejected as part of admission.

For an example, consider the following scenarios relative to tracking quota on
CPU:

| Pod | Container | Request CPU | Limit CPU | Result |
| --- | --------- | ----------- | --------- | ------ |
| X | C1 | 100m | 500m | The quota usage is incremented 100m |
| Y | C2 | 100m | none | The quota usage is incremented 100m |
| Y | C2 | none | 500m | The quota usage is incremented 500m since request will default to limit |
| Z | C3 | none | none | The pod is rejected since it does not enumerate a request. |

The rationale for accounting for the requested amount of a resource versus the
limit is the belief that a user should only be charged for what they are
scheduled against in the cluster. In addition, attempting to track usage against
actual usage, where request < actual < limit, is considered highly volatile.

As a consequence of this decision, the user is able to spread its usage of a
resource across multiple tiers of service.  Let's demonstrate this via an
example with a 4 cpu quota.

The quota may be allocated as follows:

| Pod | Container | Request CPU | Limit CPU | Tier | Quota Usage |
| --- | --------- | ----------- | --------- | ---- | ----------- |
| X | C1 | 1 | 4 | Burstable | 1 |
| Y | C2 | 2 | 2 | Guaranteed | 2 |
| Z | C3 | 1 | 3 | Burstable | 1 |

It is possible that the pods may consume 9 cpu over a given time period
depending on the nodes available cpu that held pod X and Z, but since we
scheduled X and Z relative to the request, we only track the requesting value
against their allocated quota. If one wants to restrict the ratio between the
request and limit, it is encouraged that the user define a **LimitRange** with
**LimitRequestRatio** to control burst out behavior. This would in effect, let
an administrator keep the difference between request and limit more in line with
tracked usage if desired.

## Status API

A REST API endpoint to update the status section of the **ResourceQuota** is
exposed. It requires an atomic compare-and-swap in order to keep resource usage
tracking consistent.

## Resource Quota Controller

A resource quota controller monitors observed usage for tracked resources in the
**Namespace**.

If there is observed difference between the current usage stats versus the
current **ResourceQuota.Status**, the controller posts an update of the
currently observed usage metrics to the **ResourceQuota** via the /status
endpoint.

The resource quota controller is the only component capable of monitoring and
recording usage updates after a DELETE operation since admission control is
incapable of guaranteeing a DELETE request actually succeeded.

## AdmissionControl plugin: ResourceQuota

The **ResourceQuota** plug-in introspects all incoming admission requests.

To enable the plug-in and support for ResourceQuota, the kube-apiserver must be
configured as follows:

```
$ kube-apiserver --admission-control=ResourceQuota
```

It makes decisions by evaluating the incoming object against all defined
**ResourceQuota.Status.Hard** resource limits in the request namespace. If
acceptance of the resource would cause the total usage of a named resource to
exceed its hard limit, the request is denied.

If the incoming request does not cause the total usage to exceed any of the
enumerated hard resource limits, the plug-in will post a
**ResourceQuota.Status** document to the server to atomically update the
observed usage based on the previously read **ResourceQuota.ResourceVersion**.
This keeps incremental usage atomically consistent, but does introduce a
bottleneck (intentionally) into the system.

To optimize system performance, it is encouraged that all resource quotas are
tracked on the same **ResourceQuota** document in a **Namespace**. As a result,
it is encouraged to impose a cap on the total number of individual quotas that
are tracked in the **Namespace** to 1 in the **ResourceQuota** document.

## kubectl

kubectl is modified to support the **ResourceQuota** resource.

`kubectl describe` provides a human-readable output of quota.

For example:

```console
$ kubectl create -f test/fixtures/doc-yaml/admin/resourcequota/namespace.yaml
namespace "quota-example" created
$ kubectl create -f test/fixtures/doc-yaml/admin/resourcequota/quota.yaml --namespace=quota-example
resourcequota "quota" created
$ kubectl describe quota quota --namespace=quota-example
Name:                    quota
Namespace:               quota-example
Resource                 Used      Hard
--------                 ----      ----
cpu                      0         20
memory                   0         1Gi
persistentvolumeclaims   0         10
pods                     0         10
replicationcontrollers   0         20
resourcequotas           1         1
secrets                  1         10
services                 0         5
```

## More information

See [resource quota document](../admin/resource-quota.md) and the [example of Resource Quota](../admin/resourcequota/) for more information.



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/admission_control_resource_quota.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
