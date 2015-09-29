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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/pods.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Pods

In Kubernetes, rather than individual application containers, _pods_ are the smallest deployable units that can be created, scheduled, and managed.

## What is a _pod_?

A _pod_ (as in a pod of whales or pea pod) corresponds to a colocated group of applications running with a shared context. Within that context, the applications may also have individual cgroup isolations applied. A pod models an application-specific "logical host" in a containerized environment. It may contain one or more applications which are relatively tightly coupled &mdash; in a pre-container world, they would have executed on the same physical or virtual host.

The context of the pod can be defined as the conjunction of several Linux namespaces:

* PID namespace (applications within the pod can see each other's processes)
* network namespace (applications within the pod have access to the same IP and port space)
* IPC namespace (applications within the pod can use SystemV IPC or POSIX message queues to communicate)
* UTS namespace (applications within the pod share a hostname)

Applications within a pod also have access to shared volumes, which are defined at the pod level and made available in each application's filesystem. Additionally, a pod may define top-level cgroup isolations which form an outer bound to any individual isolation applied to constituent applications.

In terms of [Docker](https://www.docker.com/) constructs, a pod consists of a colocated group of Docker containers with shared [volumes](volumes.md). PID namespace sharing is not yet implemented with Docker.

Like individual application containers, pods are considered to be relatively ephemeral rather than durable entities. As discussed in [life of a pod](pod-states.md), pods are scheduled to nodes and remain there until termination (according to restart policy) or deletion. When a node dies, the pods scheduled to that node are deleted. Specific pods are never rescheduled to new nodes; instead, they must be replaced (see [replication controller](replication-controller.md) for more details). (In the future, a higher-level API may support pod migration.)

## Motivation for pods

### Resource sharing and communication

Pods facilitate data sharing and communication among their constituents.

The applications in the pod all use the same network namespace/IP and port space, and can find and communicate with each other using localhost. Each pod has an IP address in a flat shared networking namespace that has full communication with other physical computers and containers across the network. The hostname is set to the pod's Name for the application containers within the pod. [More details on networking](../admin/networking.md).

In addition to defining the application containers that run in the pod, the pod specifies a set of shared storage volumes. Volumes enable data to survive container restarts and to be shared among the applications within the pod.

### Management

Pods also simplify application deployment and management by providing a higher-level abstraction than the raw, low-level container interface. Pods serve as units of deployment and horizontal scaling/replication. Co-location (co-scheduling), fate sharing, coordinated replication, resource sharing, and dependency management are handled automatically.

## Uses of pods

Pods can be used to host vertically integrated application stacks, but their primary motivation is to support co-located, co-managed helper programs, such as:

* content management systems, file and data loaders, local cache managers, etc.
* log and checkpoint backup, compression, rotation, snapshotting, etc.
* data change watchers, log tailers, logging and monitoring adapters, event publishers, etc.
* proxies, bridges, and adapters
* controllers, managers, configurators, and updaters

Individual pods are not intended to run multiple instances of the same application, in general.

## Alternatives considered

_Why not just run multiple programs in a single (Docker) container?_

1. Transparency. Making the containers within the pod visible to the infrastructure enables the infrastructure to provide services to those containers, such as process management and resource monitoring. This facilitates a number of conveniences for users.
2. Decoupling software dependencies. The individual containers may be rebuilt and redeployed independently. Kubernetes may even support live updates of individual containers someday.
3. Ease of use. Users don't need to run their own process managers, worry about signal and exit-code propagation, etc.
4. Efficiency. Because the infrastructure takes on more responsibility, containers can be lighter weight.

_Why not support affinity-based co-scheduling of containers?_

That approach would provide co-location, but would not provide most of the benefits of pods, such as resource sharing, IPC, guaranteed fate sharing, and simplified management.

## Durability of pods (or lack thereof)

Pods aren't intended to be treated as durable [pets](https://blog.engineyard.com/2014/pets-vs-cattle). They won't survive scheduling failures, node failures, or other evictions, such as due to lack of resources, or in the case of node maintenance.

In general, users shouldn't need to create pods directly. They should almost always use controllers (e.g., [replication controller](replication-controller.md)), even for singletons.  Controllers provide self-healing with a cluster scope, as well as replication and rollout management.

The use of collective APIs as the primary user-facing primitive is relatively common among cluster scheduling systems, including [Borg](https://research.google.com/pubs/pub43438.html), [Marathon](https://mesosphere.github.io/marathon/docs/rest-api.html), [Aurora](http://aurora.apache.org/documentation/latest/configuration-reference/#job-schema), and [Tupperware](http://www.slideshare.net/Docker/aravindnarayanan-facebook140613153626phpapp02-37588997).

Pod is exposed as a primitive in order to facilitate:

* scheduler and controller pluggability
* support for pod-level operations without the need to "proxy" them via controller APIs
* decoupling of pod lifetime from controller lifetime, such as for bootstrapping
* decoupling of controllers and services &mdash; the endpoint controller just watches pods
* clean composition of Kubelet-level functionality with cluster-level functionality &mdash; Kubelet is effectively the "pod controller"
* high-availability applications, which will expect pods to be replaced in advance of their termination and certainly in advance of deletion, such as in the case of planned evictions, image prefetching, or live pod migration [#3949](http://issue.k8s.io/3949)

The current best practice for pets is to create a replication controller with `replicas` equal to `1` and a corresponding service. If you find this cumbersome, please comment on [issue #260](http://issue.k8s.io/260).

## Termination of Pods

Because pods represent running processes on nodes in the cluster, it is important to allow those processes to gracefully terminate when they are no longer needed (vs being violently killed with a KILL signal and having no chance to clean up). Users should be able to request deletion and know when processes terminate, but also be able to ensure that deletes eventually complete. When a user requests deletion of a pod the system records the intended grace period before the pod is allowed to be forcefully killed, and a TERM signal is sent to the main process in each container. Once the grace period has expired the KILL signal is sent to those processes and the pod is then deleted from the API server. If the Kubelet or the container manager is restarted while waiting for processes to terminate, the termination will be retried with the full grace period.

An example flow:

1. User sends command to delete Pod, with default grace period (30s)
2. The Pod in the API server is updated with the time beyond which the Pod is considered "dead" along with the grace period.
3. Pod shows up as "Terminating" when listed in client commands
4. (simultaneous with 3) When the Kubelet sees that a Pod has been marked as terminating because the time in 2 has been set, it begins the pod shutdown process.
  1. If the pod has defined a [preStop hook](container-environment.md#hook-details), it is invoked inside of the pod. If the `preStop` hook is still running after the grace period expires, step 2 is then invoked with a small (2 second) extended grace period.
  2. The processes in the Pod are sent the TERM signal.
5. (simultaneous with 3), Pod is removed from endpoints list for service, and are no longer considered part of the set of running pods for replication controllers. Pods that shutdown slowly can continue to serve traffic as load balancers (like the service proxy) remove them from their rotations.
6. When the grace period expires, any processes still running in the Pod are killed with SIGKILL.
7. The Kubelet will finish deleting the Pod on the API server by setting grace period 0 (immediate deletion). The Pod disappears from the API and is no longer visible from the client.

By default, all deletes are graceful within 30 seconds. The `kubectl delete` command supports the `--grace-period=<seconds>` option which allows a user to override the default and specify their own value. The value `0` indicates that delete should be immediate, and removes the pod in the API immediately so a new pod can be created with the same name. On the node pods that are set to terminate immediately will still be given a small grace period before being force killed.

## Privileged mode for pod containers

From kubernetes v1.1, any container in a pod can enable privileged mode, using the `privileged` flag on the `SecurityContext` of the container spec. This is useful for containers that want to use linux capabilities like manipulating the network stack and accessing devices. Processes within the container get almost the same privileges that are available to processes outside a container. With privileged mode, it should be easier to write network and volume plugins as seperate pods that don't need to be compiled into the kubelet.

If the master is running kubernetes v1.1 or higher, and the nodes are running a version lower than v1.1, then new privileged pods will be accepted by api-server, but will not be launched. They will be pending state.
If user calls `kubectl describe pod FooPodName`, user can see the reason why the pod is in pending state. The events table in the describe command output will say:
`Error validating pod "FooPodName"."FooPodNamespace" from api, ignoring: spec.containers[0].securityContext.privileged: forbidden '<*>(0xc2089d3248)true'`


If the master is running a version lower than v1.1, then privileged pods cannot be created. If user attempts to create a pod, that has a privileged container, the user will get the following error:
`The Pod "FooPodName" is invalid.
spec.containers[0].securityContext.privileged: forbidden '<*>(0xc20b222db0)true'`

## API Object

Pod is a top-level resource in the kubernetes REST API. More details about the
API object can be found at: [Pod API
object](https://htmlpreview.github.io/?https://github.com/kubernetes/kubernetes/HEAD/docs/api-reference/definitions.html#_v1_pod).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/pods.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
