<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/proposals/resource-quota-scoping.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Resource Quota - Scoping resources

## Problem Description

### Ability to limit compute requests and limits

The existing `ResourceQuota` API object constrains the total amount of compute
resource requests.  This is useful when a cluster-admin is interested in
controlling explicit resource guarantees such that there would be a relatively
strong guarantee that pods created by users who stay within their quota will find
enough free resources in the cluster to be able to schedule.  The end-user creating
the pod is expected to have intimate knowledge on their minimum required resource
as well as their potential limits.

There are many environments where a cluster-admin does not extend this level
of trust to their end-user because user's often request too much resource, and
they have trouble reasoning about what they hope to have available for their
application versus what their application actually needs.  In these environments,
the cluster-admin will often just expose a single value (the limit) to the end-user.
Internally, they may choose a variety of other strategies for setting the request.
For example, some cluster operators are focused on satisfying a particular over-commit
ratio and may choose to set the request as a factor of the limit to control for
over-commit.  Other cluster operators may defer to a resource estimation tool that
sets the request based on known historical trends.  In this environment, the
cluster-admin is interested in exposing a quota to their end-users that maps
to their desired limit instead of their request since that is the value the user
manages.

### Ability to limit impact to node and promote fair-use

The current `ResourceQuota` API object does not allow the ability
to quota best-effort pods separately from pods with resource guarantees.
For example, if a cluster-admin applies a quota that caps requested
cpu at 10 cores and memory at 10Gi, all pods in the namespace must
make an explicit resource request for cpu and memory to satisfy
quota.  This prevents a namespace with a quota from supporting best-effort
pods.

In practice, the cluster-admin wants to control the impact of best-effort
pods to the cluster, but not restrict the ability to run best-effort pods
altogether.

As a result, the cluster-admin requires the ability to control the
max number of active best-effort pods.  In addition, the cluster-admin
requires the ability to scope a quota that limits compute resources to
exclude best-effort pods.

### Ability to quota long-running vs bounded-duration compute resources

The cluster-admin may want to quota end-users separately
based on long-running vs bounded-duration compute resources.

For example, a cluster-admin may offer more compute resources
for long running pods that are expected to have a more permanent residence
on the node than bounded-duration pods.  Many batch style workloads
tend to consume as much resource as they can until something else applies
the brakes.  As a result, these workloads tend to operate at their limit,
while many traditional web applications may often consume closer to their
request if there is no active traffic.  An operator that wants to control
density will offer lower quota limits for batch workloads than web applications.

A classic example is a PaaS deployment where the cluster-admin may
allow a separate budget for pods that run their web application vs pods that
build web applications.

Another example is providing more quota to a database pod than a
pod that performs a database migration.

## Use Cases

* As a cluster-admin, I want the ability to quota
 * compute resource requests
 * compute resource limits
 * compute resources for terminating vs non-terminating workloads
 * compute resources for best-effort vs non-best-effort pods

## Proposed Change

### New quota tracked resources

Support the following resources that can be tracked by quota.

| Resource Name | Description |
| ------------- | ----------- |
| cpu | total cpu requests (backwards compatibility) |
| memory | total memory requests (backwards compatibility) |
| requests.cpu | total cpu requests |
| requests.memory | total memory requests |
| limits.cpu | total cpu limits |
| limits.memory | total memory limits |

### Resource Quota Scopes

Add the ability to associate a set of `scopes` to a quota.

A quota will only measure usage for a `resource` if it matches
the intersection of enumerated `scopes`.

Adding a `scope` to a quota limits the number of resources
it supports to those that pertain to the `scope`.  Specifying
a resource on the quota object outside of the allowed set
would result in a validation error.

| Scope | Description |
| ----- | ----------- |
| Terminating | Match `kind=Pod` where `spec.activeDeadlineSeconds >= 0` |
| NotTerminating | Match `kind=Pod` where `spec.activeDeadlineSeconds = nil` |
| BestEffort | Match `kind=Pod` where `status.qualityOfService in (BestEffort)` |
| NotBestEffort | Match `kind=Pod` where `status.qualityOfService not in (BestEffort)` |

A `BestEffort` scope restricts a quota to tracking the following resources:

* pod

A `Terminating`, `NotTerminating`, `NotBestEffort` scope restricts a quota to
tracking the following resources:

* pod
* memory, requests.memory, limits.memory
* cpu, requests.cpu, limits.cpu

## Data Model Impact

```
// The following identify resource constants for Kubernetes object types
const (
	// CPU request, in cores. (500m = .5 cores)
	ResourceRequestsCPU ResourceName = "requests.cpu"
	// Memory request, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
	ResourceRequestsMemory ResourceName = "requests.memory"
	// CPU limit, in cores. (500m = .5 cores)
	ResourceLimitsCPU ResourceName = "limits.cpu"
	// Memory limit, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
	ResourceLimitsMemory ResourceName = "limits.memory"
)

// A scope is a filter that matches an object
type ResourceQuotaScope string
const (
  ResourceQuotaScopeTerminating ResourceQuotaScope = "Terminating"
  ResourceQuotaScopeNotTerminating ResourceQuotaScope = "NotTerminating"
  ResourceQuotaScopeBestEffort ResourceQuotaScope = "BestEffort"
  ResourceQuotaScopeNotBestEffort ResourceQuotaScope = "NotBestEffort"
)

// ResourceQuotaSpec defines the desired hard limits to enforce for Quota
// The quota matches by default on all objects in its namespace.
// The quota can optionally match objects that satisfy a set of scopes.
type ResourceQuotaSpec struct {
  // Hard is the set of desired hard limits for each named resource
  Hard ResourceList `json:"hard,omitempty"`
  // A collection of filters that must match each object tracked by a quota.
  // If not specified, the quota matches all objects.
  Scopes []ResourceQuotaScope `json:"scopes,omitempty"`
}
```

## Rest API Impact

None.

## Security Impact

None.

## End User Impact

The `kubectl` commands that render quota should display its scopes.

## Performance Impact

This feature will make having more quota objects in a namespace
more common in certain clusters.  This impacts the number of quota
objects that need to be incremented during creation of an object
in admission control.  It impacts the number of quota objects
that need to be updated during controller loops.

## Developer Impact

None.

## Alternatives

This proposal initially enumerated a solution that leveraged a
`FieldSelector` on a `ResourceQuota` object.  A `FieldSelector`
grouped an `APIVersion` and `Kind` with a selector over its
fields that supported set-based requirements.  It would have allowed
a quota to track objects based on cluster defined attributes.

For example, a quota could do the following:

* match `Kind=Pod` where `spec.restartPolicy in (Always)`
* match `Kind=Pod` where `spec.restartPolicy in (Never, OnFailure)`
* match `Kind=Pod` where `status.qualityOfService in (BestEffort)`
* match `Kind=Service` where `spec.type in (LoadBalancer)`
 * see [#17484](https://github.com/kubernetes/kubernetes/issues/17484)

Theoretically, it would enable support for fine-grained tracking
on a variety of resource types.  While extremely flexible, there
are cons to to this approach that make it premature to pursue
at this time.

* Generic field selectors are not yet settled art
 * see [#1362](https://github.com/kubernetes/kubernetes/issues/1362)
 * see [#19084](https://github.com/kubernetes/kubernetes/pull/19804)
* Discovery API Limitations
 * Not possible to discover the set of field selectors supported by kind.
 * Not possible to discover if a field is readonly, readwrite, or immutable
 post-creation.

The quota system would want to validate that a field selector is valid,
and it would only want to select on those fields that are readonly/immutable
post creation to make resource tracking work during update operations.

The current proposal could grow to support a `FieldSelector` on a
`ResourceQuotaSpec` and support a simple migration path to convert
`scopes` to the matching `FieldSelector` once the project has identified
how it wants to handle `fieldSelector` requirements longer term.

This proposal previously discussed a solution that leveraged a
`LabelSelector` as a mechanism to partition quota.  This is potentially
interesting to explore in the future to allow `namespace-admins` to
quota workloads based on local knowledge.  For example, a quota
could match all kinds that match the selector
`tier=cache, environment in (dev, qa)` separately from quota that
matched `tier=cache, environment in (prod)`.  This is interesting to
explore in the future, but labels are insufficient selection targets
for `cluster-administrators` to control footprint.  In those instances,
you need fields that are cluster controlled and not user-defined.

## Example

### Scenario 1

The cluster-admin wants to restrict the following:

* limit 2 best-effort pods
* limit 2 terminating pods that can not use more than 1Gi of memory, and 2 cpu cores
* limit 4 long-running pods that can not use more than 4Gi of memory, and 4 cpu cores
* limit 6 pods in total, 10 replication controllers

This would require the following quotas to be added to the namespace:

```
$ cat quota-best-effort
apiVersion: v1
kind: ResourceQuota
metadata:
  name: quota-best-effort
spec:
  hard:
    pods: "2"
  scopes:
  - BestEffort

$ cat quota-terminating
apiVersion: v1
kind: ResourceQuota
metadata:
  name: quota-terminating
spec:
  hard:
    pods: "2"
    memory.limit: 1Gi
    cpu.limit: 2
  scopes:
  - Terminating
  - NotBestEffort

$ cat quota-longrunning
apiVersion: v1
kind: ResourceQuota
metadata:
  name: quota-longrunning
spec:
  hard:
    pods: "2"
    memory.limit: 4Gi
    cpu.limit: 4
  scopes:
  - NotTerminating
  - NotBestEffort 

$ cat quota
apiVersion: v1
kind: ResourceQuota
metadata:
  name: quota
spec:
  hard:
    pods: "6"
    replicationcontrollers: "10"
```

In the above scenario, every pod creation will result in its usage being
tracked by `quota` since it has no additional scoping.  The pod will then
be tracked by at 1 additional quota object based on the scope it
matches.  In order for the pod creation to succeed, it must not violate
the constraint of any matching quota.  So for example, a best-effort pod
would only be created if there was available quota in `quota-best-effort`
and `quota`.

## Implementation

### Assignee

@derekwaynecarr

### Work Items

* Add support for requests and limits
* Add support for scopes in quota-related admission and controller code

## Dependencies

None.

Longer term, we should evaluate what we want to do with `fieldSelector` as
the requests around different quota semantics will continue to grow.

## Testing

Appropriate unit and e2e testing will be authored.

## Documentation Impact

Existing resource quota documentation and examples will be updated.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/resource-quota-scoping.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
