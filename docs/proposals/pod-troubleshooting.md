# Troubleshooting Running Pods

This proposal seeks to add first class support for troubleshooting by creating a
mechanism to execute a shell or other troubleshooting tools inside a running pod
without requiring that the associated container images include such tools.

## Motivation

### Development

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

### Operations and Support

As Kubernetes gains in popularity, it's becoming the case that a person
troubleshooting an application is not necessarily the person who built it.
Operations staff and Support organizations want the ability to attach a "known
good" or automated debugging environment to a pod.

## Goals and Non-Goals

Goals include:

*   Enable troubleshooting of minimal container images
*   Allow troubleshooting of containers in `CrashLoopBackoff`
*   Improve supportability of native Kubernetes applications
*   Enable novel uses of short-lived containers in pods

Non-Goals of this proposal are:

*   Guarantee resources for ad-hoc troubleshooting. If troubleshooting causes a
    pod to exceed its resource limit it may be evicted.

## User Stories

These user stories are intended to give examples how this proposal addresses the
above requirements.

### Operations

Jonas runs a service "neato" that consists of a statically compiled Go binary
running in a minimal container image. One of the its pods is suddenly having
trouble connecting to an internal service. Being in operations, Jonas wants to
be able to inspect the running pod without restarting it, but he doesn't
necessarily need to enter the container itself. He wants to:

1.  Inspect the filesystem of target container
1.  Execute debugging utilities not included in the container image
1.  Initiate network requests from the pod network namespace

This is achieved by running a new "debug" container in the pod namespaces. His
troubleshooting session might resemble:

```
% kubectl debug -it -m debian neato-5thn0 -- bash
root@debug-image:~# ps x
  PID TTY      STAT   TIME COMMAND
    1 ?        Ss     0:00 /pause
   13 ?        Ss     0:00 bash
   26 ?        Ss+    0:00 /neato
  107 ?        R+     0:00 ps x
root@debug-image:~# cat /proc/26/root/etc/resolv.conf
search default.svc.cluster.local svc.cluster.local cluster.local
nameserver 10.155.240.10
options ndots:5
root@debug-image:~# dig @10.155.240.10 neato.svc.cluster.local.

; <<>> DiG 9.9.5-9+deb8u6-Debian <<>> @10.155.240.10 neato.svc.cluster.local.
; (1 server found)
;; global options: +cmd
;; connection timed out; no servers could be reached
```

Thus Jonas discovers that the cluster's DNS service isn't responding.

### Development

Eunice has noticed something strange with the production build of her
application. She wants to run the debug build in order to gather additional
data. She can create a copy of a running pod using a command like:

```
% kubectl run --copy-of=neato-5thn0 --name=neato-debug --image=gcr.io/neato/debug
```

Using `kubectl label`, Eunice adds the new `neato-debug` into the canary service
for the neato application just long enough to gather the data, and then she
deletes the debug pod.

### Debugging

Thurston is debugging a tricky issue that's difficult to reproduce. He can't
reproduce the issue with the debug build, so he attaches a debug container to
one of the pods exhibiting the problem:

```
% kubectl debug -it --image=gcr.io/neato/debugger neato-5x9k3 -- sh
Defaulting container name to debug.
/ # ps x
PID   USER     TIME   COMMAND
    1 root       0:00 /pause
   13 root       0:00 /neato
   26 root       0:00 sh
   32 root       0:00 ps x
/ # gdb -p 13
...
```

He discovers that he needs access to the actual container, which he can achieve
by installing busybox into the target container:

```
root@debug-image:~# cp /bin/busybox /proc/13/root
root@debug-image:~# nsenter -t 13 -m -u -p -n -r /busybox sh


BusyBox v1.22.1 (Debian 1:1.22.0-9+deb8u1) built-in shell (ash)
Enter 'help' for a list of built-in commands.

/ # ls -l /neato
-rwxr-xr-x    2 0        0           746888 May  4  2016 /neato
```

Note that running the commands referenced above require `CAP_SYS_ADMIN` and
`CAP_SYS_PTRACE`.

### Automation

Ginger is a security engineer tasked with running security audits across all of
her company's running containers. Even though his company has no standard base
image, she's able to audit all containers using:

```
% for pod in $(kubectl get -o name pod); do
    kubectl debug -m gcr.io/neato/security-audit -p $pod /security-audit.sh
  done
```

### Technical Support

Roy's team provides support for his company's multi-tenant cluster. He can
access the Kubernetes API (as a viewer) on behalf of the users he's supporting,
but he does not have administrative access to nodes or a say in how the
application image is constructed. When someone asks for help, Roy's first step
is to run his team's autodiagnose script:

```
% kubectl debug --image=gcr.io/google_containers/autodiagnose nginx-pod-1234
```

## Requirements

A solution to troubleshoot arbitrary container images MUST:

*   troubleshoot arbitrary running containers with minimal prior configuration
*   allow access to all pod namespaces and the file systems of individual
    containers
*   fetch troubleshooting utilities at debug time rather than at the time of pod
    initialization
*   respect admission restrictions
*   allow introspection of pod state using existing tools (no hidden containers)
*   support arbitrary runtimes via the CRI (possibly with reduced feature set)
*   require no direct access to the node

A good solution SHOULD:

*   have an excellent user experience (i.e. should be a feature of the platform
    rather than a config-time solution)
*   have no *inherent* side effects to the running container image

## Debug Patterns

There are two patterns that are useful when debugging a pod:

1.  *Copy Debug* creates a copy of a pod with minor changes to the pod spec
    specified on the command or complex changes specified using an editor.
1.  *Running Debug* mode causes Kubernetes to run a new *debug container* in the
    pod context. A *debug container* is not part of the pod spec and has only
    limited configuration by the user. It is created by a debug operation on an
    existing pod and reported in `PodStatus`.

### Copy Debug

*Copy Debug* works by fetching the spec of a running pod, modifying it based on
command line arguments, and creating a new pod. This fits well within the
existing `kubectl run` command, which exists to create new pods.

An example use might be to modify the entrypoint of a pod that's crash looping:

```
kubectl run target-pod-copy --copy-of=target-pod -it --attach --container=crashing-container --command -- sh
```

Command line arguments allow simple, single-container changes but are
insufficient for tasks like modifying volumes and their mount points. For more
complex changes, we can provide an `--edit` workflow to adjust the generated
spec before creation:

```
# Similar to the following workflow:
#   kubectl get -o yaml ... > $temp_file
#   $EDITOR $temp_file
#   kubectl create -f $temp_file
kubectl run --copy-of=target-pod -it -c shell --image=debian --edit
```

Note that the `--edit` workflow is useful for all generated config, not just
copies of pods.

#### Scheduling conflicts

Attempting to duplicate some resources will cause the resulting pod to be
unschedulable, for example creating a second read-write volume mount of a
gcePersistentDisk will create a disk conflict. This is the same problem faced by
scaling a deployment, so one way to provide a better user experience is to
validate the generated config similar to `ValidatePodTemplateSpecForRC()` with
replicas > 1 as `kubectl scale` does.

Validating client side is not optimal, and currently only gcePersistentDisk
provides a check in `ValidatePodTemplateSpecForRC()`, but this isn't really a
new problem introduced by this new functionality.

#### Stripping labels

Only the spec will be copied from the running pod. Copying metadata such as
labels might result in the pod copy receiving traffic or being killed
immediately by a replication controller. It would be trivial to provide a
`--copy-labels` option if this is desired behavior.

### Running Debug

*Running Debug* requires new functionality in the kubelet and a new `kubectl
debug` command that results in a new container being introduced in the running
pod. This *debug container* is not part of the pod spec, which remains
immutable, and is not restarted according to the pod's `restartPolicy`.

The status of a *debug container* is reported in a new `DebugContainerStatuses`
field of `PodStatus`, which is a read-only list of `ContainerStatus` that
contains an entry for every *debug container* that has ever run in this pod, and
is reported by `kubectl describe`. This mirrors how `InitContainers` are
implemented.

Debug operations will generate an event so auditing utilities can reconstruct
what commands are run, even though `kubectl exec` does not do this. A *Debug
Containers* section will display all debug containers in the output of `kubectl
describe` similar to the *Init Containers* section.

The following command would attach to a newly created container in a pod:

```
kubectl debug -it -c debug-shell --image=debian target-pod -- bash
```

It would be reasonable for Kubernetes to provide a default container image
which, combined with a default entrypoint, makes the minimal debug command:

```
kubectl debug -p target-pod -it
```

If the specified container name already exists, `kubectl debug` will kill the
existing container prior to starting a new one. It's legal to reuse container
names, in which case the exited containers will continue to be reported in
`DebugContainerStatuses`.

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
1.  Preserve and report state of exitted debug containers

Of these, preventing `SyncPod()` from killing the container is the most complex,
but most of this complexity has already been implemented by *init containers*.
For *debug containers* we need only amend `computePodContainerChanges()` to
ignore containers labeled as debug.

The simplest way to preserve state of debug containers is to exempt them from
garbage collection while the pod exists. Number of debug containers are limited
because they are started manually and are never restarted.

#### Changes to External Tools

Third party tools that examine pod state by inspecting the pod status must be
updated to also examine debug containers.

## Requirements Analysis

Many people have proposed alternate solutions to this problem. This section
discusses how the proposed solution meets all of the stated requirements and is
intended to contrast the alternatives listed below.

**Troubleshoot arbitrary running containers with minimal prior configuration.**
This solution requires no prior configuration.

**Access to all pod namespaces and the file systems of individual containers.**
This solution runs a container in the shared pod namespaces. It relies on the
behavior of /proc/<pid>/root (and therefore shared PID namespace) to provide
access to filesystems of individual containers.

**Fetch troubleshooting utilities at debug time**. This solution use normal
container image distribution mechanisms to fetch images when the debug command
is run.

**Respect admission restrictions.** Requests from kubectl are proxied through
the apiserver and so are available to existing [admission
controllers](https://kubernetes.io/docs/admin/admission-controllers/). Plugins
already exist to intercept `exec` and `attach` calls, but extending this to
support `debug` has not yet been scoped.

**Allow introspection of pod state using existing tools**. The list of
`DebugContainerStatuses` is never truncated. If a debug container has run in
this pod it will appear here.

**Support arbitrary runtimes via the CRI**. This proposal is implemented
entirely in the kubelet runtime manager and requires no changes in the
individual runtimes.

**Have an excellent user experience**. This solution is conceptually
straightforward and surfaced in a single `kubectl` command that "runs a thing in
a pod". Debug tools are distributed by container image, which is already well
understood by users. There is no automatic copying of files or hidden paths.

By using container images, users are empowered to create custom debug images.
Available images can be restricted by admission policy. Some examples of
possible debug images:

*   A script that automatically gathers a debugging snapshot and uploads it to a
    cloud storage bucket before killing the pod.
*   An image with a shell modified to log every statement to an audit API.

**Require no direct access to the node.** This solution uses the standard
streaming API.

**Have no inherent side effects to the running container image.** The target pod
is not modified by default, but resources used by the debug container will be
billed to the pod's cgroup, which means it could be evicted. A future
improvement could be to decrease the likelihood of eviction when there's an
active debug container.

## Implementation Plan

### Copy Debug Implementation

The changes to `kubectl run` can be implemented entirely client-side as a
command that fetches the config of a current pod, modifies it, and creates a new
pod. This is automating the following manual workflow:

1.  <code>kubectl get -o yaml --export pod <em>pod-name</em> > pod.yaml</code>
1.  Remove <code>status</code> from <code>pod.yaml</code> and strip all values
    from <code>metadata</code>
1.  Add a <code>metadata.name</code>
1.  Modify <code>spec</code> to suit debugging needs (e.g. change
    <code>command</code> to <code>sh</code>)
1.  <code>kubectl create -f pod.yaml</code>
1.  <code>kubectl attach -it pod-copy-name</code>

### Related Features

*Running Debug* doesn't depend on other features, but other features are
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
resource scaling ([#10782](https://issues.k8s.io/10782)).

### Implementation Plan

#### 1.7 Alpha Release

We're targeting an alpha release in Kubernetes 1.7 that includes the following
basic functionality:

*   Support in the kubelet for attaching debug containers to a running pod
*   A `kubectl debug` command to initiate a debug container
*   `kubectl describe pod` will list status of debug containers running in a pod

Functionality in the kubelet will be hidden behind an alpha feature flag and
disabled by default. The following are explicitly out of scope for the 1.7 alpha
release:

*   *Copy Debug* will not be implemented
*   `kubectl describe pod` will not report the running command, only the
    container status
*   Garbage collection of exitted debug containers cannot be tuned
*   Specific integration with cluster admission controller and pod security
    policy, though these may work by default

*Running Debug* is implemented in the kubelet's generic runtime manager.
Performing this operation with a legacy (non-CRI) runtime or a cluster without
the alpha feature enabled will result in a not implemented error. Implementation
will be broken into the following steps:

##### Step 1: Container Type

Debug containers exist within the kubelet as a `ContainerStatus` without a
container spec. As with other `kubecontainers`, their state is persisted in the
runtime via labels.

To distinguish debug containers, the runtime uses a new
`io.kubernetes.container.type` label. Existing containers will be started with a
type of `REGULAR`, `INIT` or `SANDBOX`. When added in a subsequent step, debug
containers will start with with the type `DEBUG`. The `type` label will populate
a new field `Type` in `container.ContainerStatus`.

##### Step 2: Creation and Handling of Debug Containers

This step adds methods for creating debug containers, but doesn't yet modify the
kubelet API. The kubelet will gain a `RunDebugContainer()` method which accepts
a `v1.Container` and creates a debug container. Fields that don't make sense in
the context of a debug container (e.g. probes, lifecycle, resources) will be
excluded by a white list and ignored.

The kubelet will treat `DEBUG` containers differently in the following ways:

1.  `SyncPod()` ignores containers of type `DEBUG`, since there is no
    configuration to sync.
1.  `DEBUG` containers will be excluded from calculation of pod phase and
    condition.
1.  `DEBUG` containers will not be garbage collected while the pod exists.

All containers will continue to be killed by `KillPod()`, which already operates
on containers returned by the runtime and does not discriminate on type.

##### Step 3: kubelet API changes

The kubelet will gain a new streaming endpoint `/debug` similar to `/exec`,
`/attach`, etc. The debug endpoint will create a new debug container and
automatically attach to it.

##### Step 4: API changes

A client requests a debugging session by:

1.  `POST` to `/api/vX/namespaces/{namespace}/pods/{pod}/debug` including a
    `v1.Container`
1.  Validation for debug disallows some `Container` fields (e.g. `resources)`
1.  If successful, the connection is upgraded to streaming and attached

To report debug container status back to the client, `PodStatus` gains a new
field, `DebugContainerStatuses`. `convertStatusToAPIStatus()` will sort `DEBUG`
container status into `DebugContainerStatuses` similar to as it does for
`InitContainerStatuses`.

##### Step 5: kubectl changes

A new top-level command `kubectl debug` is added to invoke *Running Debug*.
`kubectl describe pod` is extended to report the contents of
`DebugContainerStatuses` in addition to `ContainerStatuses and
InitContainerStatuses.`

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
