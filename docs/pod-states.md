# The life of a pod

Updated: 9/22/2014

This document covers the intersection of pod states, the PodStatus type, the life-cycle of a pod, events, restart policies, and replication controllers.  It is not an exhaustive document, but an introduction to the topics.

## What is PodStatus?

While `PodStatus` represents the state of a pod, it is not intended to form a state machine. `PodStatus` is an observation of the current state of a pod.  As such, we discourage people from thinking about "transitions" or "changes" or "future states".

## Events

Since `PodStatus` is not a state machine, there are no edges which can be considered the "reason" for the current state.  Reasons can be determined by examining the events for the pod.  Events that affect containers, e.g. OOM, are reported as pod events.

TODO(@lavalamp) Event design

## Controllers and RestartPolicy

The only controller we have today is `ReplicationController`.  `ReplicationController` is *only* appropriate for pods with `RestartPolicy = Always`.  `ReplicationController` should refuse to instantiate any pod that has a different restart policy.

There is a legitimate need for a controller which keeps pods with other policies alive.  Both of the other policies (`OnFailure` and `Never`) eventually terminate, at which point the controller should stop recreating them.  Because of this fundamental distinction, let's hypothesize a new controller, called `JobController` for the sake of this document, which can implement this policy.

## Container termination

Containers can terminate with one of two statuses:
   1. success: The container exited voluntarily with a status code of 0.
   1. failure: The container exited with any other status code or signal, or was stopped by the system.

TODO(@dchen1107) Define ContainerStatus like PodStatus

## PodStatus values and meanings

The number and meanings of `PodStatus` values are tightly guarded.  Other than what is documented here, nothing should be assumed about pods with a given `PodStatus`.

### pending

The pod has been accepted by the system, but one or more of the containers has not been started.  This includes time before being schedule as well as time spent downloading images over the network, which could take a while.

### running

The pod has been bound to a node, and all of the containers have been started.  At least one container is still running (or is in the process of restarting).

### succeeded

All containers in the pod have terminated in success.

### failed

All containers in the pod have terminated, at least one container has terminated in failure.

## Pod lifetime

In general, pods which are created do not disappear until someone destroys them.  This might be a human or a `ReplicationController`.  The only exception to this rule is that pods with a `PodStatus` of `succeeded` or `failed` for more than some duration (determined by the master) will expire and be automatically reaped.

If a node dies or is disconnected from the rest of the cluster, some entity within the system (call it the NodeController for now) is responsible for applying policy (e.g. a timeout) and marking any pods on the lost node as `failed`.

## Examples

   * Pod is `running`, 1 container, container exits success
     * Log completion event
     * If RestartPolicy is:
       * Always: restart container, pod stays `running`
       * OnFailure: pod becomes `succeeded`
       * Never: pod becomes `succeeded`

   * Pod is `running`, 1 container, container exits failure
     * Log failure event
     * If RestartPolicy is:
       * Always: restart container, pod stays `running`
       * OnFailure: restart container, pod stays `running`
       * Never: pod becomes `failed`

   * Pod is `running`, 2 containers, container 1 exits failure
     * Log failure event
     * If RestartPolicy is:
       * Always: restart container, pod stays `running`
       * OnFailure: restart container, pod stays `running`
       * Never: pod stays `running`
     * When container 2 exits...
       * Log failure event
       * If RestartPolicy is:
         * Always: restart container, pod stays `running`
         * OnFailure: restart container, pod stays `running`
         * Never: pod becomes `failed`

   * Pod is `running`, container becomes OOM
     * Container terminates in failure
     * Log OOM event
     * If RestartPolicy is:
       * Always: restart container, pod stays `running`
       * OnFailure: restart container, pod stays `running`
       * Never: log failure event, pod becomes `failed`

   * Pod is `running`, a disk dies
     * All containers are killed
     * Log appropriate event
     * Pod becomes `failed`
     * If running under a controller, pod will be recreated elsewhere

   * Pod is `running`, its node is segmented out
     * NodeController waits for timeout
     * NodeController marks pod `failed`
     * If running under a controller, pod will be recreated elsewhere
