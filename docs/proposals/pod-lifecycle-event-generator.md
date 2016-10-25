# Kubelet: Pod Lifecycle Event Generator (PLEG)

In Kubernetes, Kubelet is a per-node daemon that manages the pods on the node,
driving the pod states to match their pod specifications (specs). To achieve
this, Kubelet needs to react to changes in both (1) pod specs and (2) the
container states. For the former, Kubelet watches the pod specs changes from
multiple sources; for the latter, Kubelet polls the container runtime
periodically (e.g., 10s) for the latest states for all containers.

Polling incurs non-negligible overhead as the number of pods/containers increases,
and is exacerbated by Kubelet's parallelism -- one worker (goroutine) per pod, which
queries the container runtime individually. Periodic, concurrent, large number
of requests causes high CPU usage spikes (even when there is no spec/state
change), poor performance, and reliability problems due to overwhelmed container
runtime. Ultimately, it limits Kubelet's scalability.

(Related issues reported by users: [#10451](https://issues.k8s.io/10451),
[#12099](https://issues.k8s.io/12099), [#12082](https://issues.k8s.io/12082))

## Goals and Requirements

The goal of this proposal is to improve Kubelet's scalability and performance
by lowering the pod management overhead.
 - Reduce unnecessary work during inactivity (no spec/state changes)
 - Lower the concurrent requests to the container runtime.

The design should be generic so that it can support different container runtimes
(e.g., Docker and rkt).

## Overview

This proposal aims to replace the periodic polling with a pod lifecycle event
watcher.

![pleg](pleg.png)

## Pod Lifecycle Event

A pod lifecycle event interprets the underlying container state change at the
pod-level abstraction, making it container-runtime-agnostic. The abstraction
shields Kubelet from the runtime specifics.

```go
type PodLifeCycleEventType string

const (
    ContainerStarted      PodLifeCycleEventType = "ContainerStarted"
    ContainerStopped      PodLifeCycleEventType = "ContainerStopped"
    NetworkSetupCompleted PodLifeCycleEventType = "NetworkSetupCompleted"
    NetworkFailed         PodLifeCycleEventType = "NetworkFailed"
)

// PodLifecycleEvent is an event reflects the change of the pod state.
type PodLifecycleEvent struct {
    // The pod ID.
    ID types.UID
    // The type of the event.
    Type PodLifeCycleEventType
    // The accompanied data which varies based on the event type.
    Data interface{}
}
```

Using Docker as an example, starting of a POD infra container would be
translated to a NetworkSetupCompleted`pod lifecycle event.


## Detect Changes in Container States Via Relisting

In order to generate pod lifecycle events, PLEG needs to detect changes in
container states. We can achieve this by periodically relisting all containers
(e.g., docker ps). Although this is similar to Kubelet's polling today, it will
only be performed by a single thread (PLEG).  This means that we still
benefit from not having all pod workers hitting the container runtime
concurrently. Moreover, only the relevant pod worker would be woken up
to perform a sync.

The upside of relying on relisting is that it is container runtime-agnostic,
and requires no external dependency.

### Relist period

The shorter the relist period is, the sooner that Kubelet can detect the
change. Shorter relist period also implies higher cpu usage. Moreover, the
relist latency depends on the underlying container runtime, and usually
increases as the number of containers/pods grows. We should set a default
relist period based on measurements. Regardless of what period we set, it will
likely be significantly shorter than the current pod sync period (10s), i.e.,
Kubelet will detect container changes sooner.


## Impact on the Pod Worker Control Flow

Kubelet is responsible for dispatching an event to the appropriate pod
worker based on the pod ID. Only one pod worker would be woken up for
each event.

Today, the pod syncing routine in Kubelet is idempotent as it always
examines the pod state and the spec, and tries to drive to state to
match the spec by performing a series of operations. It should be
noted that this proposal does not intend to change this property --
the sync pod routine would still perform all necessary checks,
regardless of the event type. This trades some efficiency for
reliability and eliminate the need to build a state machine that is
compatible with different runtimes.

## Leverage Upstream Container Events

Instead of relying on relisting, PLEG can leverage other components which
provide container events, and translate these events into pod lifecycle
events. This will further improve Kubelet's responsiveness and reduce the
resource usage caused by frequent relisting.

The upstream container events can come from:

(1). *Event stream provided by each container runtime*

Docker's API exposes an [event
stream](https://docs.docker.com/reference/api/docker_remote_api_v1.17/#monitor-docker-s-events).
Nonetheless, rkt does not support this yet, but they will eventually support it
(see [coreos/rkt#1193](https://github.com/coreos/rkt/issues/1193)).

(2). *cgroups event stream by cAdvisor*

cAdvisor is integrated in Kubelet to provide container stats. It watches cgroups
containers using inotify and exposes an event stream. Even though it does not
support rkt yet, it should be straightforward to add such a support.

Option (1) may provide richer sets of events, but option (2) has the advantage
to be more universal across runtimes, as long as the container runtime uses
cgroups. Regardless of what one chooses to implement now, the container event
stream should be easily swappable with a clearly defined interface.

Note that we cannot solely rely on the upstream container events due to the
possibility of missing events. PLEG should relist infrequently to ensure no
events are missed.

## Generate Expected Events

*This is optional for PLEGs which performs only relisting, but required for
PLEGs that watch upstream events.*

A pod worker's actions could lead to pod lifecycle events (e.g.,
create/kill a container), which the worker would not observe until
later. The pod worker should ignore such events to avoid unnecessary
work.

For example, assume a pod has two containers, A and B. The worker

 - Creates container A
 - Receives an event `(ContainerStopped, B)`
 - Receives an event `(ContainerStarted, A)`


The worker should ignore the `(ContainerStarted, A)` event since it is
expected. Arguably, the worker could process `(ContainerStopped, B)`
as soon as it receives the event, before observing the creation of
A. However, it is desirable to wait until the expected event
`(ContainerStarted, A)` is observed to keep a consistent per-pod view
at the worker. Therefore, the control flow of a single pod worker
should adhere to the following rules:

1. Pod worker should process the events sequentially.
2. Pod worker should not start syncing until it observes the outcome of its own
   actions in the last sync to maintain a consistent view.

In other words, a pod worker should record the expected events, and
only wake up to perform the next sync until all expectations are met.

 - Creates container A, records an expected event `(ContainerStarted, A)`
 - Receives `(ContainerStopped, B)`; stores the event and goes back to sleep.
 - Receives `(ContainerStarted, A)`; clears the expectation. Proceeds to handle
   `(ContainerStopped, B)`.

We should set an expiration time for each expected events to prevent the worker
from being stalled indefinitely by missing events.

## TODOs for v1.2

For v1.2, we will add a generic PLEG which relists periodically, and leave
adopting container events for future work. We will also *not* implement the
optimization that generate and filters out expected events to minimize
redundant syncs.

- Add a generic PLEG using relisting. Modify the container runtime interface
  to provide all necessary information to detect container state changes
  in `GetPods()` (#13571).

- Benchmark docker to adjust relising frequency.

- Fix/adapt features that rely on frequent, periodic pod syncing.
    * Liveness/Readiness probing: Create a separate probing manager using
      explicitly container probing period [#10878](https://issues.k8s.io/10878).
    * Instruct pod workers to set up a wake-up call if syncing failed, so that
      it can retry.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/pod-lifecycle-event-generator.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
