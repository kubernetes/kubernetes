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

# Troubleshooting Running Pods

This proposal seeks to add first class support for troubleshooting by creating a
mechanism to execute a shell or other troubleshooting tools inside a running pod
without requiring that the associated container images include such tools.

## Motivation

Many developers of native Kubernetes applications wish to treat Kubernetes as an
execution platform for custom binaries produced by a build system. These users
can forgo the scripted OS install of traditional Dockerfiles and instead `COPY`
the output of their build system into a container image built `FROM scratch`.
This confers several advantages:

1.  **Minimal images** lower operational burden and reduce attack vectors.
1.  **Immutable images** improve correctness and reliability.
1.  **Smaller image size** reduces resource usage and speeds deployments.

The disadvantage of using containers built `FROM scratch` is the lack of system
binaries provided a base Operation System image makes it difficult to
troubleshoot running containers. Kubernetes should enable troubleshooting pods
regardless of the contents of the container image.

## Goals and Non-Goals

Goals include:

*   Enable troubleshooting of minimal container images
*   Allow troubleshooting of containers in `CrashLoopBackoff`
*   Improve supportability of native Kubernetes applications
*   Enable novel uses of short-lived containers in pods

Non-Goals of this proposal are:

*   Fully design the ability to cross-mount container image filesystems. This is
    complex and should be covered in a separate doc.
*   Guarantee resources for ad-hoc troubleshooting. If troubleshooting causes a
    pod to exceed its resource limit it may be evicted.

## Requirements

A solution to troubleshoot minimal container images MUST:

*   troubleshoot arbitrary running containers with minimal prior configuration
*   allow access to all namespaces shared by containers in a pod and the mount
    namespaces of individual containers
*   fetch troubleshooting utilities at debug time rather than at the time of pod
    initialization
*   respect admission restrictions
*   allow introspection of pod state using existing tools (no hidden containers)
*   support arbitrary runtimes via the CRI (possibly with reduced feature set)

A good solution SHOULD:

*   have an excellent user experience (i.e. should be a feature of the platform
    rather than a config-time solution)
*   require no direct access to the node
*   have no inherent side effects to the running container image
*   re-use existing container image distribution channels
*   enable detection of pods that have been modified by an exec ("tainted" pods)

## kubectl debug

We will introduce an "execute a debug container in a pod" pattern that parallels
`kubectl exec`'s "execute a debug process in a container" in a new command,
`kubectl debug`, which has two modes of operation:

1.  *Copy Debug* mode creates a copy of a pod with minor changes to the pod spec
    specified on the command line.
1.  *Running Debug* mode causes Kubernetes to run a new *debug container* in the
    pod context. A *debug container* is not part of the pod spec and cannot be
    configured by the user. It is created by a debug operation on an existing
    pod and reported in `PodStatus`.

The end user need not be aware of the two modes. They should feel like a unified
debugging experience that has an option to create a copy (though initially a
copy will be the only option) and doesn't allow modifying existing containers
when not a copy (because they're immutable).

While both modes are useful, the primary motivation behind two modes is that
*Copy Debug* can be implemented client-side using existing APIs as part of a
long-term troubleshooting strategy that provides users some functionality in one
or two releases while the more complex *Running Debug* mode is implemented
server-side. If no one would find *Copy Debug* useful, it could be abandoned to
focus solely on *Running Debug*.

### Copy Debug Mode

*Copy Debug* mode works by fetching the spec of a running pod, modifying it
based on command line arguments, and creating a new pod. A common use might be
to modify the entrypoint of a pod that's crash looping:

```
% kubectl debug --copy-of=target-pod --pod=target-pod-copy -it --attach --container=crashing-container --command -- sh
```

The `--container` and `--attach` arguments would follow established conventions
for choosing a default container (kubectl exec) and attaching if stdin was
specified (kubectl run) and so are purely illustrative in the above example.
When `--copy-of` is provided, we can construct a reasonable default for
`--pod`(e.g. "${copy-of}-copy"), making the minimal form of this command:

```
% kubectl debug --copy-of=target-pod -it --command -- sh
```

One can also specify a container that doesn't currently exist in the pod spec to
create a new one:

```
% kubectl debug --copy-of=target-pod -it -c shell --image=debian
```

For more complex changes, such as modifying volume mounts, we can provide an
`--edit` workflow to fine tune the generated pod sec:

```
% kubectl debug --copy-of=target-pod -it -c shell --image=debian --edit
```

#### Scheduling conflicts

Attempting to duplicate some resources will cause the resulting pod to be
unschedulable, for example creating a second read-write volume mount of a
gcePersistentDisk will create a disk conflict. This is the same problem faced by
scaling a deployment, so one way to provide a better user experience is to
validate the generated config similar to `ValidatePodTemplateSpecForRC()` with
replicas > 1 as `kubectl scale` does.

Validating client side is not optimal, and currently on gcePersistentDisk
provides a check in `ValidatePodTemplateSpecForRC()`. Since *Running Debug* must
be implemented server-side, a better long term strategy may be to migrate *Copy
Debug* server-side as well to take advantage of easier validation against
current cluster state.

#### Stripping labels

Only the spec will be copied from the running pod. Copying metadata such as
labels might result in the pod copy receiving traffic or being killed
immediately by a replication controller. It would be trivial to provide a
`--copy-labels` option if this is desired behavior.

### Running Debug Mode

*Running Debug* mode is a `kubectl debug` invocation that results in a new
container being introduced in the running container. This *debug container* is
not part of the pod spec, which remains immutable, and is not restarted
according to the pod's restartPolicy.

The status of a *debug container* is reported in a new `DebugContainerStatuses`
field of `PodStatus`, which is a read-only list of `ContainerStatus` that
contains an entry for every *debug container* that has ever run in this pod.
Additionally, a new boolean `Tainted` in `PodCondition` changes from false to
true upon execution of a *debug container*. `Tainted` could also be flipped by
other operations which change a pod to a non-pristine state, such as `kubectl
exec`. A transition of `Tainted` from true to false is not possible.

Debug operations will generate an event so auditing utilities can reconstruct
what commands are run, an improvement over `kubectl exec`. A *Debug Containers*
section will display all debug containers in the output of `kubectl describe`
similar to the *Init Containers* section, and the status of `Tainted` will be
displayed in the *Conditions* section.

The following command would attach to a newly created container in a pod:

```
% kubectl debug -p target-pod -it -c debug-shell --image=debian -- bash
```

It would be reasonable to provide a default container name and image which,
combined with a default entrypoint, makes the minimal debug command:

```
% kubectl debug -p target-pod -it
```

If the specified container name already exists, `kubectl debug` will either
attach to the already running container or restart a container that has exited.
Specifying arguments is not allowed in the first case but is allowed in the
latter to support remote shell uses cases such as `kubectl debug -p
target-pod -- netstat`.

#### Additional kubelet complexity

Implementing debug requires no changes to the Container Runtime Interface as
it's the same operation as creating a regular container. The majority of the
complexity for this feature falls to the kubelet, but it's simplified by the
lack of configuration and restart policy. The kubelet's additional
responsibilities include:

1.  Start or restart a debug container in a pod when requested.
1.  Don't kill the debug container while the pod is alive.
1.  Report status on the debug container.
1.  Stop and delete the container when the pod is deleted.

Of these, preventing SyncPod() from killing the container is the most complex,
but most of this complexity has already been implemented by *init containers*.
For *debug containers* we need only amend computePodContainerChanges() to ignore
containers labeled

Additional implementation details, including prerequisites, are detailed in
Implementation Plan below.

## Implementation Plan

Functionality of `kubectl debug` will improve across multiple releases as
features are implemented, culminating in the ability to add a troubleshooting
container to a running pod. The following steps should be implemented serially
as independent changes.

### Copy Debug Implementation

The first phase of `kubectl debug` can be implemented entirely in kubectl as a
command that fetches the config of a current pod, modifies it, and creates a new
pod. This is automating the following manual workflow:

1.  `kubectl get -o yaml --export pod *pod-name* > pod.yaml`
1.  Remove `status` from `pod.yaml` and strip all values
    from `metadata`
1.  Add a `metadata.name`
1.  Modify `spec` to suit debugging needs (e.g. change
    `command` to `sh`)
1.  `kubectl create -f pod.yaml`
1.  `kubectl attach -it pod-copy-name`

### Related Features

*Running debug* doesn't depend on other features, but other features are
required to realize full troubleshooting functionality.

#### Shared PID Namespace

Tracked in [#1615](https://issues.k8s.io/1615), a shared PID namespace became
available in Docker 1.12 and requires plumbing through the docker drivers,
extending the infra container to reap orphaned zombies, and [rolling
out](https://pr.k8s.io/37404).

#### Pod Level cgroup

Tracked in [#26751](https://issues.k8s.io/26751), Kubernetes currently enforces
resource constraints at a container level. A container added by *Running Debug*
cannot allocate its own resources and must instead fit within previously
allocated pod resources.

A pod could potentially be evicted during debugging, particularly if there are
insufficient resources to allow for the troubleshooting process, but there's not
much we can do about this until Kubernetes adds support for vertical pod
resource scaling.

#### Container Volumes

Tracked in [#831](https://issues.k8s.io/831), we'll want to be able to make
volumes and eventually root filesystems of the other containers in the pod
available to the troubleshooting container. This is a complex issue and should
be covered in a separate proposal.

### *Running Debug* Implementation

At this stage we can extend `kubectl debug` to operate on running pods. Initial
implementation will not include mounting volumes or filesystems of other
containers.

#### kubelet changes

*Running Debug* will be implemented in the kubelet's generic runtime manager.
When the CRI is not enabled, performing this operation will result in a not
implemented error. State for pods is stored as labels in the container runtime,
so we will add a new label to differentiate a *debug container* from containers
that are part of the pod spec.

The `debug` handler for the kublet will create and start a new container with an
`io.kubernetes.container.type=DEBUG` label. This label will populate a new field
`Type` in `container.ContainerStatus`. `SyncPod()` will ignore containers of
type `DEBUG`. `convertStatusToAPIStatus()` will sort `DEBUG` containers into the
separate `PodStatus.DebugContainerStatuses` as it does currently for
`InitContainerStatuses`.

`DEBUG` containers will be excluded from calculation of pod phase and condition.
All containers will continue to be killed by `KillPod()`, which operates on
containers returned by the runtime and will not discriminate on type.

#### API changes

A client requests a debugging session by:

1.  `POST` to `/api/vX/namespaces/{namespace}/pods/{pod}/debug` including a
    `v1.Container`
1.  Validation for debug disallows some `Container` fields such as `resources`
1.  If successful, client is responsible for `attach` to container if
    appropriate

#### Changes to External Tools

Tools that examine pod state by inspecting the pod spec must be updated to also
inspect `PodStatus.DebugContainerStatuses`.

### Mounting Pod Volumes

When adding a container to the pod in either mode we can support an option to
mount all pod volumes by name to a user-configurable mount point defaulting to
`/mnt/volumes`. For *Copy Debug* this can be done when generating the new pod
spec. The kubelet's debug handler would perform a similar action for *Running
Debug*.

### Cross Mounting Container Images

Depending on resolution of [#831](https://issues.k8s.io/831), the last step will
be to implement cross-container filesystem image mounts. Implementing that
feature would almost certainly require amending the Container Runtime Interface.
Once implemented it should be easy to amend the debug handler to mount the other
container filesystems by name in (e.g.) `/mnt/containers`.

### Open Questions

*   How will affect SELinux labels and user namespace segregation?
*   what security context is used by the new container? how does that interact
    with PodSecurityPolicy?

## User Stories

### Debugging

Samantha has a service that consists of a statically compiled Go binary running
in a minimal container that is suddenly having trouble connecting to an internal
service. Her troubleshooting session might resemble:

```
% kubectl get pods
NAME          READY     STATUS    RESTARTS   AGE
neato-5thn0   1/1       Running   0          1d
% kubectl debug -it -m gcr.io/neato/debug-image -p neato-5thn0 -- bash
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

This leads Samantha to discover that the cluster's DNS service isn't responding.

### Automation

Abe is a security engineer tasked with running security audits across all of his
company's running containers. Even though his company has no standard base
image, he's able to audit all containers using:

```
% for pod in $(kubectl get -o name pod); do
    kubectl debug -m gcr.io/neato/security-audit -p $pod /security-audit.sh
  done
```

## Alternatives Considered

### Mutable Pod Spec

Rather than adding an operation to have Kubernetes attach a pod we could instead
make the pod spec mutable so the client can generate an update adding a
container. `SyncPod()` has no issues adding the container to the pod at that
point, but an immutable pod spec has been a basic assumption in Kubernetes thus
far and changing it carries risk. It's preferable to keep the pod spec immutable
as a best practice.

### Ephemeral container

An earlier version of this proposal suggested running an ephemeral container in
the pod namespaces. The container would not be added to the pod spec and would
exist only as long as the process it ran. This has the advantage of behaving
similarly to the current kubectl exec, but it is opaque and likely violates
design assumptions. We could add constructs to track and report on both
traditional exec process and exec containers, but this would probably be more
work than adding to the pod spec. Both are generally useful, and neither
precludes the other in the future, so we chose mutating the pod spec for
expedience.

### Attaching Container Type Volume

Combining container volumes ([#831](https://issues.k8s.io/831)) with the ability
to add volumes to the pod spec would get us most of the way there. One could
mount a volume of debug utilities at debug time. Docker does not allow adding a
volume to a running container, however, so this would require a container
restart. A restart doesn't meet our requirements for troubleshooting.

Rather than attaching the container at debug time, kubernetes could always
attach a volume at a random path at run time, just in case it's needed. Though
this simplifies the solution by working within the existing constraints of
`kubectl exec`, it has a sufficient list of minor limitations (detailed in
[#10834](https://issues.k8s.io/10834)) to result in a poor user experience.

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

*   [Pod Troubleshooting Tracking Issue](https://issues.k8s.io/27140)
*   [CRI Tracking Issue](https://issues.k8s.io/28789)
*   [CRI: expose optional runtime features](https://issues.k8s.io/32803)
*   [Resource QoS in
    Kubernetes](https://github.com/kubernetes/kubernetes/blob/master/docs/design/resource-qos.md)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/pod-troubleshooting.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
