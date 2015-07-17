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

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/user-guide/pod-states.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# The life of a pod

Updated: 4/14/2015

This document covers the lifecycle of a pod.  It is not an exhaustive document, but an introduction to the topic.

## Pod Phase

As consistent with the overall [API convention](../devel/api-conventions.md#typical-status-properties), phase is a simple, high-level summary of the phase of the lifecycle of a pod. It is not intended to be a comprehensive rollup of observations of container-level or even pod-level conditions or other state, nor is it intended to be a comprehensive state machine.

The number and meanings of `PodPhase` values are tightly guarded.  Other than what is documented here, nothing should be assumed about pods with a given `PodPhase`.

* Pending: The pod has been accepted by the system, but one or more of the container images has not been created.  This includes time before being scheduled as well as time spent downloading images over the network, which could take a while.
* Running: The pod has been bound to a node, and all of the containers have been created.  At least one container is still running, or is in the process of starting or restarting.
* Succeeded: All containers in the pod have terminated in success, and will not be restarted.
* Failed: All containers in the pod have terminated, at least one container has terminated in failure (exited with non-zero exit status or was terminated by the system).

## Pod Conditions

A pod containing containers that specify readiness probes will also report the Ready condition. Condition status values may be `True`, `False`, or `Unknown`.

## Container Probes

A [Probe](https://godoc.org/github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1#Probe) is a diagnostic performed periodically by the kubelet on a container. Specifically the diagnostic is one of three [Handlers](https://godoc.org/github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1#Handler):

* `ExecAction`: executes a specified command inside the container expecting on success that the command exits with status code 0.
* `TCPSocketAction`: performs a tcp check against the container's IP address on a specified port expecting on success that the port is open.
* `HTTPGetAction`: performs an HTTP Get againsts the container's IP address on a specified port and path expecting on success that the response has a status code greater than or equal to 200 and less than 400.

Each probe will have one of three results:

* `Success`: indicates that the container passed the diagnostic.
* `Failure`: indicates that the container failed the diagnostic.
* `Unknown`: indicates that the diagnostic failed so no action should be taken.

Currently, the kubelet optionally performs two independent diagnostics on running containers which trigger action:

* `LivenessProbe`: indicates whether the container is *live*, i.e. still running. The LivenessProbe hints to the kubelet when a container is unhealthy. If the LivenessProbe fails, the kubelet will kill the container and the container will be subjected to it's [RestartPolicy](#restartpolicy). The default state of Liveness before the initial delay is `Success`. The state of Liveness for a container when no probe is provided is assumed to be `Success`.
* `ReadinessProbe`: indicates whether the container is *ready* to service requests. If the ReadinessProbe fails, the endpoints controller will remove the pod's IP address from the endpoints of all services that match the pod. Thus, the ReadinessProbe is sometimes useful to signal to the endpoints controller that even though a pod may be running, it should not receive traffic from the proxy (e.g. the container has a long startup time before it starts listening or the container is down for maintenance). The default state of Readiness before the initial delay is `Failure`. The state of Readiness for a container when no probe is provided is assumed to be `Success`.

## Container Statuses

More detailed information about the current (and previous) container statuses can be found in [ContainerStatuses](https://godoc.org/github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1#PodStatus). The information reported depends on the current [ContainerState](https://godoc.org/github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1#ContainerState), which may be Waiting, Running, or Terminated.

## RestartPolicy

The possible values for RestartPolicy are `Always`, `OnFailure`, or `Never`. If RestartPolicy is not set, the default value is `Always`. RestartPolicy applies to all containers in the pod. RestartPolicy only refers to restarts of the containers by the Kubelet on the same node. As discussed in the [pods document](pods.md#durability-of-pods-or-lack-thereof), once bound to a node, a pod will never be rebound to another node. This means that some kind of controller is necessary in order for a pod to survive node failure, even if just a single pod at a time is desired.

The only controller we have today is [`ReplicationController`](replication-controller.md).  `ReplicationController` is *only* appropriate for pods with `RestartPolicy = Always`.  `ReplicationController` should refuse to instantiate any pod that has a different restart policy.

There is a legitimate need for a controller which keeps pods with other policies alive. Pods having any of the other policies (`OnFailure` or `Never`) eventually terminate, at which point the controller should stop recreating them.  Because of this fundamental distinction, let's hypothesize a new controller, called [`JobController`](https://github.com/GoogleCloudPlatform/kubernetes/issues/1624) for the sake of this document, which can implement this policy.

## Pod lifetime

In general, pods which are created do not disappear until someone destroys them.  This might be a human or a `ReplicationController`.  The only exception to this rule is that pods with a `PodPhase` of `Succeeded` or `Failed` for more than some duration (determined by the master) will expire and be automatically reaped.

If a node dies or is disconnected from the rest of the cluster, some entity within the system (call it the NodeController for now) is responsible for applying policy (e.g. a timeout) and marking any pods on the lost node as `Failed`.

## Examples

   * Pod is `Running`, 1 container, container exits success
     * Log completion event
     * If RestartPolicy is:
       * Always: restart container, pod stays `Running`
       * OnFailure: pod becomes `Succeeded`
       * Never: pod becomes `Succeeded`

   * Pod is `Running`, 1 container, container exits failure
     * Log failure event
     * If RestartPolicy is:
       * Always: restart container, pod stays `Running`
       * OnFailure: restart container, pod stays `Running`
       * Never: pod becomes `Failed`

   * Pod is `Running`, 2 containers, container 1 exits failure
     * Log failure event
     * If RestartPolicy is:
       * Always: restart container, pod stays `Running`
       * OnFailure: restart container, pod stays `Running`
       * Never: pod stays `Running`
     * When container 2 exits...
       * Log failure event
       * If RestartPolicy is:
         * Always: restart container, pod stays `Running`
         * OnFailure: restart container, pod stays `Running`
         * Never: pod becomes `Failed`

   * Pod is `Running`, container becomes OOM
     * Container terminates in failure
     * Log OOM event
     * If RestartPolicy is:
       * Always: restart container, pod stays `Running`
       * OnFailure: restart container, pod stays `Running`
       * Never: log failure event, pod becomes `Failed`

   * Pod is `Running`, a disk dies
     * All containers are killed
     * Log appropriate event
     * Pod becomes `Failed`
     * If running under a controller, pod will be recreated elsewhere

   * Pod is `Running`, its node is segmented out
     * NodeController waits for timeout
     * NodeController marks pod `Failed`
     * If running under a controller, pod will be recreated elsewhere


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/pod-states.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
