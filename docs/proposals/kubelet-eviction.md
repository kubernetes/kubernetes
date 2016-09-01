<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubelet - Eviction Policy

**Authors**: Derek Carr (@derekwaynecarr), Vishnu Kannan (@vishh)

**Status**: Proposed (memory evictions WIP)

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

As of now, memory and disk based evictions are supported.
The proposal focuses on a simple default eviction strategy
intended to cover the broadest class of user workloads.

## Eviction Signals

The `kubelet` will support the ability to trigger eviction decisions on the following signals.

| Eviction Signal  | Description                                                                     |
|------------------|---------------------------------------------------------------------------------|
| memory.available | memory.available := node.status.capacity[memory] - node.stats.memory.workingSet |
| nodefs.available   | nodefs.available := node.stats.fs.available |
| nodefs.inodesFree | nodefs.inodesFree := node.stats.fs.inodesFree |
| imagefs.available | imagefs.available := node.stats.runtime.imagefs.available |
| imagefs.inodesFree | imagefs.inodesFree := node.stats.runtime.imagefs.inodesFree |

Each of the above signals support either a literal or percentage based value.  The percentage based value
is calculated relative to the total capacity associated with each signal.

`kubelet` supports only two filesystem partitions.

1. The `nodefs` filesystem that kubelet uses for volumes, daemon logs, etc.
1. The `imagefs` filesystem that container runtimes uses for storing images and container writable layers.

`imagefs` is optional. `kubelet` auto-discovers these filesystems using cAdvisor.
`kubelet` does not care about any other filesystems. Any other types of configurations are not currently supported by the kubelet. For example, it is *not OK* to store volumes and logs in a dedicated `imagefs`.

## Eviction Thresholds

The `kubelet` will support the ability to specify eviction thresholds.

An eviction threshold is of the following form:

`<eviction-signal><operator><quantity | int%>`

* valid `eviction-signal` tokens as defined above.
* valid `operator` tokens are `<`
* valid `quantity` tokens must match the quantity representation used by Kubernetes
* an eviction threshold can be expressed as a percentage if ends with `%` token.

If threshold criteria are met, the `kubelet` will take pro-active action to attempt
to reclaim the starved compute resource associated with the eviction signal.

The `kubelet` will support soft and hard eviction thresholds.

For example, if a node has `10Gi` of memory, and the desire is to induce eviction
if available memory falls below `1Gi`, an eviction signal can be specified as either
of the following (but not both).

* `memory.available<10%`
* `memory.available<1Gi`

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
| DiskPressure | nodefs.available, nodefs.inodesFree, imagefs.available, or imagefs.inodesFree | Available disk space and inodes on either the node's root filesytem or image filesystem has satisfied an eviction threshold |

The `kubelet` will continue to report node status updates at the frequency specified by
`--node-status-update-frequency` which defaults to `10s`.

### Oscillation of node conditions

If a node is oscillating above and below a soft eviction threshold, but not exceeding
its associated grace period, it would cause the corresponding node condition to
constantly oscillate between true and false, and could cause poor scheduling decisions
as a consequence.

To protect against this oscillation, the following flag is defined to control how
long the `kubelet` must wait before transitioning out of a pressure condition.

```
--eviction-pressure-transition-period=5m0s: Duration for which the kubelet has to wait
before transitioning out of an eviction pressure condition.
```

The `kubelet` would ensure that it has not observed an eviction threshold being met
for the specified pressure condition for the period specified before toggling the
condition back to `false`.

## Eviction scenarios

### Memory

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

### Disk

Let's assume the operator started the `kubelet` with the following:

```
--eviction-hard="nodefs.available<1Gi,nodefs.inodesFree<1,imagefs.available<10Gi,imagefs.inodesFree<10"
--eviction-soft="nodefs.available<1.5Gi,nodefs.inodesFree<10,imagefs.available<20Gi,imagefs.inodesFree<100"
--eviction-soft-grace-period="nodefs.available=1m,imagefs.available=2m"
```

The `kubelet` will run a sync loop that looks at the available disk
on the node's supported partitions as reported from `cAdvisor`.
If available disk space on the node's primary filesystem is observed to drop below 1Gi
or the free inodes on the node's primary filesystem is less than 1,
the `kubelet` will immediately initiate eviction.
If available disk space on the node's image filesystem is observed to drop below 10Gi
or the free inodes on the node's primary image filesystem is less than 10,
the `kubelet` will immediately initiate eviction.

If available disk space on the node's primary filesystem is observed as falling below `1.5Gi`,
or if the free inodes on the node's primary filesystem is less than 10,
or if available disk space on the node's image filesystem is observed as falling below `20Gi`,
or if the free inodes on the node's image filesystem is less than 100,
it will record when that signal was observed internally in a cache.  If at the next
sync, that criterion was no longer satisfied, the cache is cleared for that
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

## Disk based evictions

### With Imagefs

If `nodefs` filesystem has met eviction thresholds, `kubelet` will free up disk space in the following order:

1. Delete logs
1. Evict Pods if required.

If `imagefs` filesystem has met eviction thresholds, `kubelet` will free up disk space in the following order:

1. Delete unused images
1. Evict Pods if required.

### Without Imagefs

If `nodefs` filesystem has met eviction thresholds, `kubelet` will free up disk space in the following order:

1. Delete logs
1. Delete unused images
1. Evict Pods if required.

Let's explore the different options for freeing up disk space.

### Delete logs of dead pods/containers

As of today, logs are tied to a container's lifetime. `kubelet` keeps dead containers around,
to provide access to logs.
In the future, if we store logs of dead containers outside of the container itself, then
`kubelet` can delete these logs to free up disk space.
Once the lifetime of containers and logs are split, kubelet can support more user friendly policies
around log evictions. `kubelet` can delete logs of the oldest containers first.
Since logs from the first and the most recent incarnation of a container is the most important for most applications,
kubelet can try to preserve these logs and aggresively delete logs from other container incarnations.

Until logs are split from container's lifetime, `kubelet` can delete dead containers to free up disk space.

### Delete unused images

`kubelet` performs image garbage collection based on thresholds today. It uses a high and a low watermark.
Whenever disk usage exceeds the high watermark, it removes images until the low watermark is reached.
`kubelet` employs a LRU policy when it comes to deleting images.

The existing policy will be replaced with a much simpler policy.
Images will be deleted based on eviction thresholds. If kubelet can delete logs and keep disk space availability
above eviction thresholds, then kubelet will not delete any images.
If `kubelet` decides to delete unused images, it will delete *all* unused images.

### Evict pods

There is no ability to specify disk limits for pods/containers today.
Disk is a best effort resource. When necessary, `kubelet` can evict pods one at a time.
`kubelet` will follow the [Eviction Strategy](#eviction-strategy) mentioned above for making eviction decisions.
`kubelet` will evict the pod that will free up the maximum amount of disk space on the filesystem that has hit eviction thresholds.
Within each QoS bucket, `kubelet` will sort pods according to their disk usage.
`kubelet` will sort pods in each bucket as follows:

#### Without Imagefs

If `nodefs` is triggering evictions, `kubelet` will sort pods based on their total disk usage
- local volumes + logs & writable layer of all its containers.

#### With Imagefs

If `nodefs` is triggering evictions, `kubelet` will sort pods based on the usage on `nodefs`
- local volumes + logs of all its containers.

If `imagefs` is triggering evictions, `kubelet` will sort pods based on the writable layer usage of all its containers.

## Minimum eviction reclaim

In certain scenarios, eviction of pods could result in reclamation of small amount of resources. This can result in
`kubelet` hitting eviction thresholds in repeated successions. In addition to that, eviction of resources like `disk`,
 is time consuming.

To mitigate these issues, `kubelet` will have a per-resource `minimum-reclaim`. Whenever `kubelet` observes
resource pressure, `kubelet` will attempt to reclaim at least `minimum-reclaim` amount of resource.

Following are the flags through which `minimum-reclaim` can be configured for each evictable resource:

`--eviction-minimum-reclaim="memory.available=0Mi,nodefs.available=500Mi,imagefs.available=2Gi"`

The default `eviction-minimum-reclaim` is `0` for all resources.

## Deprecation of existing features

`kubelet` has been freeing up disk space on demand to keep the node stable. As part of this proposal,
some of the existing features/flags around disk space retrieval will be deprecated in-favor of this proposal.

| Existing Flag | New Flag | Rationale |
| ------------- | -------- | --------- |
| `--image-gc-high-threshold` | `--eviction-hard` or `eviction-soft` | existing eviction signals can capture image garbage collection |
| `--image-gc-low-threshold` | `--eviction-minimum-reclaim` | eviction reclaims achieve the same behavior |
| `--maximum-dead-containers` | | deprecated once old logs are stored outside of container's context |
| `--maximum-dead-containers-per-container` | | deprecated once old logs are stored outside of container's context |
| `--minimum-container-ttl-duration` | | deprecated once old logs are stored outside of container's context |
| `--low-diskspace-threshold-mb` | `--eviction-hard` or `eviction-soft` | this use case is better handled by this proposal |
| `--outofdisk-transition-frequency` | `--eviction-pressure-transition-period` | make the flag generic to suit all compute resources |

## Kubelet Admission Control

### Feasibility checks during kubelet admission

#### Memory

The `kubelet` will reject `BestEffort` pods if any of the memory
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

#### Disk

The `kubelet` will reject all pods if any of the disk eviction thresholds have been met.

Let's assume the operator started the `kubelet` with the following:

```
--eviction-soft="nodefs.available<1500Mi"
--eviction-soft-grace-period="nodefs.available=30s"
```

If the `kubelet` sees that it has less than `1500Mi` of disk available
on the node, but the `kubelet` has not yet initiated eviction since the
grace period criteria has not yet been met, the `kubelet` will still immediately
fail any incoming pods.

The rationale for failing **all** pods instead of just best effort is because disk is currently
a best effort resource for all QoS classes.

Kubelet will apply the same policy even if there is a dedicated `image` filesystem.

## Scheduler

The node will report a condition when a compute resource is under pressure.  The
scheduler should view that condition as a signal to dissuade placing additional
best effort pods on the node.

In this case, the `MemoryPressure` condition if true should dissuade the scheduler
from placing new best effort pods on the node since they will be rejected by the `kubelet` in admission.

On the other hand, the `DiskPressure` condition if true should dissuade the scheduler from
placing **any** new pods on the node since they will be rejected by the `kubelet` in admission.

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
for eviction. Instead `DaemonSet` should ideally include Guaranteed pods only.

## Known issues

### kubelet may evict more pods than needed

The pod eviction may evict more pods than needed due to stats collection timing gap. This can be mitigated by adding
the ability to get root container stats on an on-demand basis (https://github.com/google/cadvisor/issues/1247) in the future.

### How kubelet ranks pods for eviction in response to inode exhaustion

At this time, it is not possible to know how many inodes were consumed by a particular container.  If the `kubelet` observes
inode exhaustion, it will evict pods by ranking them by quality of service.  The following issue has been opened in cadvisor
to track per container inode consumption (https://github.com/google/cadvisor/issues/1422) which would allow us to rank pods
by inode consumption.  For example, this would let us identify a container that created large numbers of 0 byte files, and evict
that pod over others.



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/kubelet-eviction.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
