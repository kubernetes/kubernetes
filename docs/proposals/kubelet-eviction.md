<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubelet - Eviction Policy

**Author**: Derek Carr (@derekwaynecarr)

**Status**: Proposed

This document presents a specification for how the `kubelet` evicts pods when compute resources are too low.

## Goals

The node needs a mechanism to preserve stability when available compute resources are low.

This is especially important when dealing with incompressible compute resources such
as memory or disk.  If either resource is exhausted, the node would become unstable.

The `kubelet` has some support for influencing system behavior in response to a system OOM by
having the system OOM killer see higher OOM score adjust scores for containers that have consumed
the largest amount of memory relative to their request.  System OOM events are very compute
intensive, and can stall the node until the OOM killing process has completed.  In addition,
the system is prone to return to an unstable state since the containers that are killed due to OOM
are either restarted or a new pod is scheduled on to the node.

Instead, we would prefer a system where the `kubelet` can pro-actively monitor for
and prevent against total starvation of a compute resource, and in cases of where it
could appear to occur, pro-actively fail one or more pods, so the workload can get
moved and scheduled elsewhere when/if its backing controller creates a new pod.

## Scope of proposal

This proposal defines a pod eviction policy for reclaiming compute resources.

In the first iteration, it focuses on memory; later iterations are expected to cover
other resources like disk.  The proposal focuses on a simple default eviction strategy
intended to cover the broadest class of user workloads.

## Eviction Signals

The `kubelet` will support the ability to trigger eviction decisions on the following signals.

| Eviction Signal  | Description                                                                     |
|------------------|---------------------------------------------------------------------------------|
| memory.available | memory.available := node.status.capacity[memory] - node.stats.memory.workingSet |

## Eviction Thresholds

The `kubelet` will support the ability to specify eviction thresholds.

An eviction threshold is of the following form:

`<eviction-signal><operator><quantity>`

* valid `eviction-signal` tokens as defined above.
* valid `operator` tokens are `<`
* valid `quantity` tokens must match the quantity representation used by Kubernetes

If threhold criteria are met, the `kubelet` will take pro-active action to attempt
to reclaim the starved compute resource associated with the eviction signal.

The `kubelet` will support soft and hard eviction thresholds.

### Soft Eviction Thresholds

A soft eviction threshold pairs an eviction threshold with a required
administrator specified grace period.  No action is taken by the `kubelet`
to reclaim resources associated with the eviction signal until that grace
period has been exceeded.  If no grace period is provided, the `kubelet` will
error on startup.

In addition, if a soft eviction threshold has been met, an operator can
specify a maximum allowed pod termination grace period to use when evicting
pods from the node.  If specified, the `kubelet` will use the lesser value among
the `pod.Spec.TerminationGracePeriodSeconds` and the max allowed grace period.
If not specified, the `kubelet` will kill pods immediately with no graceful
termination.

To configure soft eviction thresholds, the following flags will be supported:

```
--eviction-soft="": A set of eviction thresholds (e.g. memory.available<1.5Gi) that if met over a corresponding grace period would trigger a pod eviction.
--eviction-soft-grace-period="": A set of eviction grace periods (e.g. memory.available=1m30s) that correspond to how long a soft eviction threshold must hold before triggering a pod eviction.
--eviction-max-pod-grace-period="0": Maximum allowed grace period (in seconds) to use when terminating pods in response to a soft eviction threshold being met.
```

### Hard Eviction Thresholds

A hard eviction threshold has no grace period, and if observed, the `kubelet`
will take immediate action to reclaim the associated starved resource.  If a
hard eviction threshold is met, the `kubelet` will kill the pod immediately
with no graceful termination.

To configure hard eviction thresholds, the following flag will be supported:

```
--eviction-hard="": A set of eviction thresholds (e.g. memory.available<1Gi) that if met would trigger a pod eviction.
```

## Eviction Monitoring Interval

The `kubelet` will initially evaluate eviction thresholds at the same
housekeeping interval as `cAdvisor` housekeeping.

In Kubernetes 1.2, this was defaulted to `10s`.

It is a goal to shrink the monitoring interval to a much shorter window.
This may require changes to `cAdvisor` to let alternate housekeeping intervals
be specified for selected data (https://github.com/google/cadvisor/issues/1247)

For the purposes of this proposal, we expect the monitoring interval to be no
more than `10s` to know when a threshold has been triggered, but we will strive
to reduce that latency time permitting.

## Node Conditions

The `kubelet` will support a node condition that corresponds to each eviction signal.

If a hard eviction threshold has been met, or a soft eviction threshold has been met
independent of its associated grace period, the `kubelet` will report a condition that
reflects the node is under pressure.

The following node conditions are defined that correspond to the specified eviction signal.

| Node Condition | Eviction Signal  | Description                                                      |
|----------------|------------------|------------------------------------------------------------------|
| MemoryPressure | memory.available | Available memory on the node has satisfied an eviction threshold |

The `kubelet` will continue to report node status updates at the frequency specified by
`--node-status-update-frequency` which defaults to `10s`.

## Eviction scenario

Let's assume the operator started the `kubelet` with the following:

```
--eviction-hard="memory.available<100Mi"
--eviction-soft="memory.available<300Mi"
--eviction-soft-grace-period="memory.available=30s"
```

The `kubelet` will run a sync loop that looks at the available memory
on the node as reported from `cAdvisor` by calculating (capacity - workingSet).
If available memory is observed to drop below 100Mi, the `kubelet` will immediately
initiate eviction. If available memory is observed as falling below `300Mi`,
it will record when that signal was observed internally in a cache.  If at the next
sync, that criteria was no longer satisfied, the cache is cleared for that
signal.  If that signal is observed as being satisfied for longer than the
specified period, the `kubelet` will initiate eviction to attempt to
reclaim the resource that has met its eviction threshold.

## Eviction of Pods

If an eviction threshold has been met, the `kubelet` will initiate the
process of evicting pods until it has observed the signal has gone below
its defined threshold.

The eviction sequence works as follows:

* for each monitoring interval, if eviction thresholds have been met
 * find candidate pod
 * fail the pod
 * block until pod is terminated on node

If a pod is not terminated because a container does not happen to die
(i.e. processes stuck in disk IO for example), the `kubelet` may select
an additional pod to fail instead.  The `kubelet` will invoke the `KillPod`
operation exposed on the runtime interface.  If an error is returned,
the `kubelet` will select a subsequent pod.

## Eviction Strategy

The `kubelet` will implement a default eviction strategy oriented around
the pod quality of service class.

It will target pods that are the largest consumers of the starved compute
resource relative to their scheduling request.  It ranks pods within a
quality of service tier in the following order.

* `BestEffort` pods that consume the most of the starved resource are failed
first.
* `Burstable` pods that consume the greatest amount of the starved resource
relative to their request for that resource are killed first.  If no pod
has exceeded its request, the strategy targets the largest consumer of the
starved resource.
* `Guaranteed` pods that consume the greatest amount of the starved resource
relative to their request are killed first.  If no pod has exceeded its request,
the strategy targets the largest consumer of the starved resource.

A guaranteed pod is guaranteed to never be evicted because of another pod's
resource consumption.  That said, guarantees are only as good as the underlying
foundation they are built upon.  If a system daemon
(i.e. `kubelet`, `docker`, `journald`, etc.) is consuming more resources than
were reserved via `system-reserved` or `kube-reserved` allocations, and the node
only has guaranteed pod(s) remaining, then the node must choose to evict a
guaranteed pod in order to preserve node stability, and to limit the impact
of the unexpected consumption to other guaranteed pod(s).

## Kubelet Admission Control

### Feasibility checks during kubelet admission

The `kubelet` will reject `BestEffort` pods if any of its associated
eviction thresholds have been exceeded independent of the configured
grace period.

Let's assume the operator started the `kubelet` with the following:

```
--eviction-soft="memory.available<256Mi"
--eviction-soft-grace-period="memory.available=30s"
```

If the `kubelet` sees that it has less than `256Mi` of memory available
on the node, but the `kubelet` has not yet initiated eviction since the
grace period criteria has not yet been met, the `kubelet` will still immediately
fail any incoming best effort pods.

The reasoning for this decision is the expectation that the incoming pod is
likely to further starve the particular compute resource and the `kubelet` should
return to a steady state before accepting new workloads.

## Scheduler

The node will report a condition when a compute resource is under pressure.  The
scheduler should view that condition as a signal to dissuade placing additional
best effort pods on the node.  In this case, the `MemoryPressure` condition if true
should dissuade the scheduler from placing new best effort pods on the node since
they will be rejected by the `kubelet` in admission.

## Best Practices

### DaemonSet

It is never desired for a `kubelet` to evict a pod that was derived from
a `DaemonSet` since the pod will immediately be recreated and rescheduled
back to the same node.

At the moment, the `kubelet` has no ability to distinguish a pod created
from `DaemonSet` versus any other object.  If/when that information is
available, the `kubelet` could pro-actively filter those pods from the
candidate set of pods provided to the eviction strategy.

In general, it should be strongly recommended that `DaemonSet` not
create `BestEffort` pods to avoid being identified as a candidate pod
for eviction.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/kubelet-eviction.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
