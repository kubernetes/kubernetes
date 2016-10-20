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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Image Exec

This proposal seeks to create a mechanism to execute a shell or other
troubleshooting tools inside a running Kubernetes pod, without requiring that
the associated container images include such tools.

## Motivation

Many developers of native Kubernetes applications wish to treat Kubernetes as an
execution platform for custom binaries produced by a build system. These users
can forgo the scripted OS install of traditional Dockerfiles and instead `COPY`
the output of their build system into a container image built `FROM scratch`.
This confers several advantages:

  1. **Minimal images** lower operational burden and reduce attack vectors.
  2. **Immutable images** improve correctness and reliability.
  3. **Smaller image size** reduces resource usage and speeds deployments.

The disadvantage of using containers built `FROM scratch` is the lack of system
binaries provided a base Operation System image makes it difficult to
troubleshoot running containers. Kubernetes should enable troubleshooting pods
regardless of the contents of the container image.

## Goals and Non-Goals

Goals include:
  - Enable troubleshooting of minimal container images
  - Improve supportability of native Kubernetes applications
  - Enable novel uses of `exec` by providing a pod-level abstraction

Non-goals include:
  - Introducing new abstractions to Kubernetes

## Requirements

A solution to troubleshoot minimal container images MUST:

  - troubleshoot arbitrary running containers with minimal prior configuration
  - allow access to all pod namespaces and mount namespaces of individual
    containers when supported by container storage driver
  - fetch troubleshooting utilities at debug time rather than at the time of pod
    initialization
  - have no inherent side effects to the running container image
  - respect admission restrictions

A good solution SHOULD:

  - have an excellent user experience (i.e. should be a feature of the platform
    rather than a config-time solution)
  - not require direct access to the node
  - re-use existing container image distribution channels
  - enable detection of pods that have been modified by an exec ("tainted" pods)

## User Stories

### Debugging

Samantha has a service that consists of a statically compiled Go binary running
in a minimal container that is suddenly having trouble connecting to an internal
service. Her troubleshooting session might resemble:

```
% kubectl get pods
NAME          READY     STATUS    RESTARTS   AGE
neato-5thn0   1/1       Running   0          1d
% kubectl exec -it -m gcr.io/neato/debug-image neato-5thn0 bash
root@debug-image-neato-5thn0:/# cat /etc/resolv.conf
search default.svc.cluster.local svc.cluster.local cluster.local
nameserver 10.155.240.10
options ndots:5
root@debug-image-neato-5thn0:/# dig @10.155.240.10 neato.svc.cluster.local.

; <<>> DiG 9.9.5-9+deb8u6-Debian <<>> @10.155.240.10 neato.svc.cluster.local.
; (1 server found)
;; global options: +cmd
;; connection timed out; no servers could be reached
```

Which leads Samantha to discover that the cluster's DNS service isn't
responding.

### Automation

Abe is a security engineer tasked with running security audits across all of his
company's running containers. Even though his company has no standard base
image, he's able to audit all containers using:

```
% for pod in $(kubectl get -o name pod); do
    kubectl exec -m gcr.io/neato/security-audit $pod /security-audit.sh
  done
```

## Image Exec

To troubleshoot a single container, one executes an ephemeral process inside the
container namespaces using `kubectl exec`. Since a pod is a collection of
containers, it follows that to debug a pod we could execute an ephemeral
container inside the pod namespaces. This is a logical extension of `kubectl
exec` enabled by adding a new flag to specify the name of the container image to
run inside the pod namespace.

The user will specify a debug image of her choice (e.g. `kubectl exec -it -m
gcr.io/myproj/debugimage podname`) and be connected to a process running inside
the pod. The processes of all other containers would be visible through the
shared PID namespace (if enabled). Network connections are visible through the
shared net namespace. The kubelet will bind mount the filesystems of other
containers in the pod into the debug container (e.g. in `/c/<container_name>/`),
potentially read-only.

The lifecycle of the debug container will model that of the current `kubectl
exec`, with the container being cleaned up after it exits. Since docker doesn't
support exec in a container that hasn't started, this is implemented under the
covers as create, start, attach, and remove in immediate succession.

## API Changes

The Exec handler will gain a new parameter to specify a debug image.

TODO: details of API changes

## Changes to kubelet

The kubelet will gain a new `ExecInPod` method to be called by `ServeExec`
instead of `ExecInContainer` when an container image name is provided. Image
exec could be accommodated by the current `ExecInContainer` with the addition of
a parameter for imageName, but it would the method name misleading.

A corresponding `ExecInPod` will be added to the kubelet's container.Runtime
interface.

### Container Runtime Interface

The [Container Runtime Interface](container-runtime-interface-v1.md) implements
a standard, imperative container interface across all container runtimes
supported by Kubernetes. This interface makes it easy to implement Image Exec.

The kubelet's Generic Runtime Manager can implement `ExecInPod` with almost no
changes to the CRI, which means we don't have to implement the logic in each of
the container runtimes.

To bind mount filesystems of other containers, the kubelet must be able to query
the mount path of a particular container. This will be added to the `Container`
message returned by `ListContainers`. If the container's filesystem is no longer
mounted in the node's mount namespace then it will not be made visible to the
troubleshooting container.

`ExecInPod` will build a map of container names to node filesystem mount points
for inclusion in the troubleshooting container's `ContainerConfig` with a
`container_path` of `/c/<container_name>`. This can only be accomplished when
the container is created.

We could instead modify CRI's `CreateContainer` to include a list of containers
IDs to "link" to the newly created container, but then we'd have to repeat the
logic in each of the container backends.

### Legacy Container Runtimes

Implementation is slightly more difficult for the legacy docker and rkt
runtimes.  We could implement the same create/start/attach/remove logic in rkt
and docker_manager, or we could save effort by linking this feature to the CRI
and return "Not Implemented" for the legacy runtimes.

## Open Questions

* Will Kubernetes evict the troubleshooting image for using additional
  resources? What happens if SyncPod() is called?
* What's the timeline for switching to the CRI?
* Docker 1.12 leaves mounted the merged container filesystem with the aufs and
  overlay drivers, but the current Google Cloud Image (GCI) does not. This will
  be significantly less useful when container filesystems are inaccessible.
* Docker 1.12 introduced a shared PID namespace. When will this be supported by
  Kubernetes?
* Is this compatible with Admission control plugins that restrict what images
  can run on a cluster?
* Would it be possible to Image Exec into pods that are in a terminal state to
  determine why they've failed?

## Alternatives Considered

### Inactive container

If Kubernetes supported the concept of an "inactive" container, we could
configure it as part of a pod and activate it at debug time. In order to avoid
coupling the debug tool versions with those of the running containers, we would
need to ensure the debug image was pulled at debug time. The container could
then be run with a TTY and attached using kubectl. We would need to figure out a
solution that allows access the filesystem of other containers.

The downside of this approach is that it requires prior configuration. In
addition to requiring prior consideration, it would increase boilerplate config.
A requirement for prior configuration makes it feel like a workaround rather
than a feature of the platform.

### Pod Mutation

If Kubernetes supported pod mutations, we could update a pod at debug time (via
kubectl patch or equivalent) to either add a debug container or a volume
containing debug binaries. The former would then be available via kubectl attach
while the latter would provide binaries for kubectl exec.

Adding a container to a running pod would be a large change to Kubernetes, and
there are several edge cases to consider such as resource limits and monitoring.
Adding a volume to a running container would be a smaller change, but it would
still be larger than the proposed change and we'd have to figure out how to
build and distribute a volume of binaries.

### Implicit Empty Volume

Kubernetes could implicitly create an EmptyDir volume for every pod which would
then be available as target for either the kubelet or a sidecar to extract a
package of binaries.

Users would have to be responsible for hosting a package build and distribution
infrastructure or rely on a public one. The complexity of this solution makes it
undesirable.

### Standalone Pod in Shared Namespace

Kubernetes could support starting a standalone pod that shares the namespace of
an existing pod.

This would be a small change to Kubernetes, but it would create edge cases in
the pod lifecycle that would have to be considered. For example, what happens to
the debugging pod when the target pod is destroyed?

### Exec from Node

The kubelet could support executing a troubleshooting binary from the node in
the namespaces of the container. Once executed this binary would lose access to
other binaries from the node, making it of limited utility and a confusing user
experience.

This couples the debug tools with the lifecycle of the node, which is worse than
coupling it with container images.

## References

  - [Tracking Issue](https://issues.k8s.io/27140)
  - [Container Runtime Interface Proposal](container-runtime-interface-v1.md)
  - [Container Runtime Interface proto](../../pkg/kubelet/api/v1alpha1/runtime/api.proto)
  - [CRI Initial PR](https://github.com/kubernetes/kubernetes/pull/25899)
  - [CRI Tracking Issue](https://issues.k8s.io/28789)
  - [CRI: expose optional runtime features](https://issues.k8s.io/32803)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/image-exec.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
