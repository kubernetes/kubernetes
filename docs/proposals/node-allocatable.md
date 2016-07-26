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
[here](http://releases.k8s.io/release-1.3/docs/proposals/node-allocatable.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Node Allocatable Resources

**Issue:** https://github.com/kubernetes/kubernetes/issues/13984

## Overview

Currently Node.Status has Capacity, but no concept of node Allocatable. We need additional
parameters to serve several purposes:

1. Kubernetes metrics provides "/docker-daemon", "/kubelet",
   "/kube-proxy", "/system" etc. raw containers for monitoring system component resource usage
   patterns and detecting regressions. Eventually we want to cap system component usage to a certain
   limit / request. However this is not currently feasible due to a variety of reasons including:
       1. Docker still uses tons of computing resources (See
          [#16943](https://github.com/kubernetes/kubernetes/issues/16943))
       2. We have not yet defined the minimal system requirements, so we cannot control Kubernetes
          nodes or know about arbitrary daemons, which can make the system resources
          unmanageable. Even with a resource cap we cannot do a full resource management on the
          node, but with the proposed parameters we can mitigate really bad resource over commits
       3. Usage scales with the number of pods running on the node
2. For external schedulers (such as mesos, hadoop, etc.) integration, they might want to partition
   compute resources on a given node, limiting how much Kubelet can use. We should provide a
   mechanism by which they can query kubelet, and reserve some resources for their own purpose.

### Scope of proposal

This proposal deals with resource reporting through the [`Allocatable` field](#allocatable) for more
reliable scheduling, and minimizing resource over commitment. This proposal *does not* cover
resource usage enforcement (e.g. limiting kubernetes component usage), pod eviction (e.g. when
reservation grows), or running multiple Kubelets on a single node.

## Design

### Definitions

![image](node-allocatable.png)

1. **Node Capacity** - Already provided as
   [`NodeStatus.Capacity`](https://htmlpreview.github.io/?https://github.com/kubernetes/kubernetes/blob/HEAD/docs/api-reference/v1/definitions.html#_v1_nodestatus),
   this is total capacity read from the node instance, and assumed to be constant.
2. **System-Reserved** (proposed) - Compute resources reserved for processes which are not managed by
   Kubernetes. Currently this covers all the processes lumped together in the `/system` raw
   container.
3. **Kubelet Allocatable** - Compute resources available for scheduling (including scheduled &
   unscheduled resources). This value is the focus of this proposal. See [below](#api-changes) for
   more details.
4. **Kube-Reserved** (proposed) - Compute resources reserved for Kubernetes components such as the
   docker daemon, kubelet, kube proxy, etc.

### API changes

#### Allocatable

Add `Allocatable` (4) to
[`NodeStatus`](https://htmlpreview.github.io/?https://github.com/kubernetes/kubernetes/blob/HEAD/docs/api-reference/v1/definitions.html#_v1_nodestatus):

```
type NodeStatus struct {
  ...
  // Allocatable represents schedulable resources of a node.
  Allocatable ResourceList `json:"allocatable,omitempty"`
  ...
}
```

Allocatable will be computed by the Kubelet and reported to the API server. It is defined to be:

```
   [Allocatable] = [Node Capacity] - [Kube-Reserved] - [System-Reserved]
```

The scheduler will use `Allocatable` in place of `Capacity` when scheduling pods, and the Kubelet
will use it when performing admission checks.

*Note: Since kernel usage can fluctuate and is out of kubernetes control, it will be reported as a
 separate value (probably via the metrics API). Reporting kernel usage is out-of-scope for this
 proposal.*

#### Kube-Reserved

`KubeReserved` is the parameter specifying resources reserved for kubernetes components (4). It is
provided as a command-line flag to the Kubelet at startup, and therefore cannot be changed during
normal Kubelet operation (this may change in the [future](#future-work)).

The flag will be specified as a serialized `ResourceList`, with resources defined by the API
`ResourceName` and values specified in `resource.Quantity` format, e.g.:

```
--kube-reserved=cpu=500m,memory=5Mi
```

Initially we will only support CPU and memory, but will eventually support more resources. See
[#16889](https://github.com/kubernetes/kubernetes/pull/16889) for disk accounting.

If KubeReserved is not set it defaults to a sane value (TBD) calculated from machine capacity. If it
is explicitly set to 0 (along with `SystemReserved`), then `Allocatable == Capacity`, and the system
behavior is equivalent to the 1.1 behavior with scheduling based on Capacity.

#### System-Reserved

In the initial implementation, `SystemReserved` will be functionally equivalent to
[`KubeReserved`](#system-reserved), but with a different semantic meaning. While KubeReserved
designates resources set aside for kubernetes components, SystemReserved designates resources set
aside for non-kubernetes components (currently this is reported as all the processes lumped
together in the `/system` raw container).

## Issues

### Kubernetes reservation is smaller than kubernetes component usage

**Solution**: Initially, do nothing (best effort). Let the kubernetes daemons overflow the reserved
resources and hope for the best. If the node usage is less than Allocatable, there will be some room
for overflow and the node should continue to function. If the node has been scheduled to capacity
(worst-case scenario) it may enter an unstable state, which is the current behavior in this
situation.

In the [future](#future-work) we may set a parent cgroup for kubernetes components, with limits set
according to `KubeReserved`.

### Version discrepancy

**API server / scheduler is not allocatable-resources aware:** If the Kubelet rejects a Pod but the
  scheduler expects the Kubelet to accept it, the system could get stuck in an infinite loop
  scheduling a Pod onto the node only to have Kubelet repeatedly reject it. To avoid this situation,
  we will do a 2-stage rollout of `Allocatable`. In stage 1 (targeted for 1.2), `Allocatable` will
  be reported by the Kubelet and the scheduler will be updated to use it, but Kubelet will continue
  to do admission checks based on `Capacity` (same as today). In stage 2 of the rollout (targeted
  for 1.3 or later), the Kubelet will start doing admission checks based on `Allocatable`.

**API server expects `Allocatable` but does not receive it:** If the kubelet is older and does not
  provide `Allocatable` in the `NodeStatus`, then `Allocatable` will be
  [defaulted](../../pkg/api/v1/defaults.go) to
  `Capacity` (which will yield today's behavior of scheduling based on capacity).

### 3rd party schedulers

The community should be notified that an update to schedulers is recommended, but if a scheduler is
not updated it falls under the above case of "scheduler is not allocatable-resources aware".

## Future work

1. Convert kubelet flags to Config API - Prerequisite to (2). See
   [#12245](https://github.com/kubernetes/kubernetes/issues/12245).
2. Set cgroup limits according KubeReserved - as described in the [overview](#overview)
3. Report kernel usage to be considered with scheduling decisions.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/node-allocatable.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
