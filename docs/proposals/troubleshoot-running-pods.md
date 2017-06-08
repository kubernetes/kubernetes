# Troubleshoot Running Pods

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

## Requirements

A solution to troubleshoot arbitrary container images MUST:

*   troubleshoot arbitrary running containers with minimal prior configuration
*   allow access to shared pod namespaces and the file systems of individual
    containers
*   fetch troubleshooting utilities at debug time rather than at the time of pod
    creation
*   be compatible with admission controllers and audit logging
*   allow inspection of pod state using existing tools (no hidden containers)
*   support arbitrary runtimes via the CRI (possibly with reduced feature set)
*   require no access to the node
*   have an excellent user experience (i.e. should be a feature of the platform
    rather than config-time trickery)
*   have no *inherent* side effects to the running container image

## Feature Summary

Any new debugging functionality will require training users. We can ease the
transition by building on an existing usage pattern. We will create a new
command, `kubectl debug`, which parallels an existing command, `kubectl exec`.
Whereas `kubectl exec` runs a *process* in a *container*, `kubectl debug` will
be similar but run a *container* in a *pod*.

A container created by `kubectl debug` is a *Debug Container*. Just like a
process run by `kubectl exec`, a Debug Container is not part of the pod spec and
has no resource stored in the API. Unlike `kubectl exec`, a Debug Container
*does* have a `ContainerStatus` that is reported with `v1.PodStatus` and
displayed by `kubectl describe pod`.

For example, the following command would attach to a newly created container in
a pod:

```
kubectl debug -it -c debug-shell --image=debian target-pod -- bash
```

It would be reasonable for Kubernetes to provide a default container name and
image1, making the minimal possible debug command:

```
kubectl debug -it target-pod
```

This creates an interactive shell in a pod which can examine and signal all
processes in the pod. It has access to the same network and IPC as processes in
the pod. It can access the filesystem of other processes by `/proc/$PID/root`,
and enter aribitrary namespaces of another container via `nsenter` when
necessary.

*Please see the User Stories section for additional examples and Alternatives
Considered for the considerable list of other solutions we considered.*

## Implementation Details

The implementation of `kubectl debug` closely mirrors the implementation of
`kubectl exec`, with most of the complexity being implemented in the `kubelet`.
From the perspective of the client, there's a new subresource of pod, /`debug`,
that creates a Debug Container based on a provided `v1.Container` and attaches
to its console. Users give Debug Containers a name (e.g. "debug" or "shell")
which is used in subsequent interactions and is reported in `kubectl describe`.

### Creating Debug Containers

The process for creating a Debug Container is:

1.  `kubectl` constructs a `v1.Container` based on command line flags and
    `POST`s it to `/api/v1/namespaces/$NS/pods/$POD/debug`.
1.  The API server performs admission control and proxies the connection to the
    kubelet's `/podDebug/$NS/$POD/$DEBUG_CONTAINER_NAME` endpoint. `/podDebug`
    is used because `/debug` is already used by the kubelet. `/podDebug` was
    chosen to parallel existing endpoints like `/containerLogs`.
1.  The kubelet instructs the Generic Runtime Manager (this feature is only
    implemented for the CRI) to create a Debug Container.
1.  The runtime manager uses the existing `startContainer()` method to create a
    container in an existing pod. `startContainer()` has one modification for
    Debug Containers: it creates a new runtime label (e.g. a docker label) that
    identifies this container as a Debug Container.
1.  The kubelet performs an attach operation and upgrades the client connection
    to streaming.

It is an error to attempt to create a Debug Container with the same name as a
container that exists in the pod spec. There are no limits on the number of
Debug Containers that can be created in a pod, but exceeding a pod's resource
allocation may cause it to be evicted.

### Debug Container Status

The status of a *debug container* is reported in a new field in `v1.PodStatus`:

```
type PodStatus struct {
        ...
        DebugContainerStatuses []ContainerStatus
}
```

This list is populated by the kubelet from
`kubecontainer.PodStatus.ContainerStatuses` in the same way as regular and init
container statuses. This is sent to the API server and displayed by `kubectl
describe pod`.

Status of running Debug Containers will always be reported in this list, but
Debug Containers that exit will be removed from this list when they are garbage
collected. In order to satisfy inspection and audit requirements, we need to
fully understand the requirements and whether they will be satisfied by other
audit features currently under development. Until then, Debug Containers may
disappear from the list of Debug Container Statuses if the node is under
resource pressure. This is acceptable for the alpha but must be resolved to
graduate from alpha status.

### Restarting and Reattaching Debug Containers

Debug Containers will never be restarted automatically. It is possible to
replace a Debug Container that has exited by re-using a Debug Container name. It
is an error to attempt to replace a Debug Container that is still running.

One way in which `kubectl debug` differs from `kubectl exec` is the ability to
reattach to Debug Container if you've been disconnected. This is accomplished by
running `kubectl debug -r` with the name of an existing, running Debug
Container. The `-r` option to `kubectl debug` is translated to a `reattach=True`
parameter. It is an error to attempt to reattach to a Debug Container that does
not exist or is not running.

Action                   | Exiting Container State | Result
------------------------ | ----------------------- | ----------------
Create Debug Container   | Does Not Exist          | Create & Attach
"                        | Running                 | *ERROR*
"                        | Exited                  | Replace & Attach
Reattach Debug Container | Does Not Exist          | *ERROR*
"                        | Running                 | Attach
"                        | Exited                  | *ERROR*

When supported by a runtime, multiple clients can attach to a single debug
container and share the terminal. This is supported by Docker.

### Killing Debug Containers

Debug containers will not be killed automatically until the pod (specifically,
the pod sandbox) is destroyed. Unlike `kubectl exec`, Debug Containers will not
receive an EOF if their connection is interrupted. Instead, Debug Containers
must be reattached to exit a running process. This could be tricky if the
process does not allocate a TTY, in this case a second Debug Container could be
used to deliver a signal via `kill(1)`.

### Container Lifecycle Changes

Implementing debug requires no changes to the Container Runtime Interface as
it's the same operation as creating a regular container. The following changes
are necessary in the kubelet:

1.  `SyncPod()` must not kill any Debug Container even though it is not part of
    the pod spec.
1.  As an exception to the above, `SyncPod()` will kill Debug Containers when
    the pod sandbox changes since a Debug Container in a no-longer-used sandbox
    is not useful. Debug Containers are not automatically restarted in the new
    sandbox.
1.  `convertStatusToAPIStatus()` must sort Debug Containers status into
    `DebugContainerStatuses` similar to as it does for `InitContainerStatuses`
1.  The kubelet must preserve information on debug containers for reporting.
1.  Debug Containers must be excluded from calculation of pod phase and
    condition

It's worth noting some things that do not need to change:

1.  `KillPod()` already operates on all running containers returned by the
    runtime.
1.  Containers created prior to this feature being enabled will have a
    `containerType` of `""`. Since this does not match `"DEBUG"` the special
    handling of Debug Containers is backwards compatible.

### Additional Constraints

1.  Non-interactive workloads are explicitly supported. There are no plans to
    supported detached workloads, but doing so would be trivial with an
    `attach=false` flag.
1.  There are no guaranteed resources for ad-hoc troubleshooting. If
    troubleshooting causes a pod to exceed its resource limit it may be evicted.

## Implementation Plan

### Alpha Release

#### Goals and Non-Goals for Alpha Release

We're targeting an alpha release in Kubernetes 1.7 that includes the following
basic functionality:

*   Support in the kubelet for creating debug containers in a running pod
*   A `kubectl debug` command to initiate a debug container
*   `kubectl describe pod` will list status of debug containers running in a pod

Functionality in the kubelet will be hidden behind an alpha feature flag and
disabled by default. The following are explicitly out of scope for the 1.7 alpha
release:

*   `kubectl describe pod` will not report the running command, only container
    status
*   Exited Debug Containers will be garbage collected as regular containers and
    may disappear from the list of Debug Container Statuses.
*   There's no specific integration with admission controller and pod security
    policy.
*   Explicit reattaching isn't implemented. Instead a `kubectl debug` invocation
    will implicitly reattach if there is an existing, running container with the
    same name. In this case container configuration will be ignored.

Debug Containers are implemented in the kubelet's generic runtime manager.
Performing this operation with a legacy (non-CRI) runtime or a cluster without
the alpha feature enabled will result in a not implemented error. Implementation
will be split into the following steps:

##### Step 1: Container Type

The first step is to add a feature gate to ensure all changes are off by
default. This will be added in the `pkg/features` `DefaultFeatureGate`.

The runtime manager stores metadata about containers in the runtime via labels
(e.g. docker labels). These labels are used to populate the fields of
`kubecontainer.ContainerStatus`. Since the runtime manager needs to handle Debug
Containers differently in a few situations, we must add a new piece of metadata
to distinguish Debug Containers from regular containers.

`startContainer()` will be updated to write a new label
`io.kubernetes.container.type` to the runtime. Existing containers will be
started with a type of `REGULAR` or `INIT`. When added in a subsequent step,
Debug Containers will start with with the type `DEBUG`.

##### Step 2: Creation and Handling of Debug Containers

This step adds methods for creating debug containers, but doesn't yet modify the
kubelet API. Since the runtime manager discards runtime (e.g. docker) labels
after populating `kubecontainer.ContainerStatus`, the label value will be stored
in a the new field `ContainerStatus.Type` so it can be used by `SyncPod()`.

The kubelet gains a `RunDebugContainer()` method which accepts a `v1.Container`
and passes it on to the Runtime Manager's `RunDebugContainer()` if implemented.
Currently only the Generic Runtime Manager (i.e. the CRI) implements the
`DebugContainerRunner` interface.

The Generic Runtime Manager's `RunDebugContainer()` calls `startContainer()` to
create the Debug Container. Additionally, `SyncPod()` is modified to skip Debug
Containers unless the sandbox is restarted.

##### Step 3: kubelet API changes

The kubelet gains a new streaming endpoint `/debugPod` and `server.getDebug()`
which accepts a `v1.Container` via `POST` and calls `RunDebugContainer()` if
`reattach` is not specified. `getDebug()` then calls `getAttach()` to perform
the attach.

##### Step 4: Kubernetes API changes

The following changes will be made to the Kubernetes API:

1.  Add the subresource `/api/vX/namespaces/{namespace}/pods/{pod}/debug` and
    associated `PodDebugOptions`
1.  `v1.PodStatus` gains a `DebugContainerStatuses` field

Validation should disallow the following fields which are incompatible with
Debug Containers:

*   Resources
*   LivenessProbe
*   ReadinessProbe
*   Lifecycle

Note that it should be possible to further restrict allowed Container fields via
admission control, but this is out of scope for the alpha feature.

The kubelet will be updated to sort status of Debug Containers into
`DebugContainerStatuses`.

##### Step 5: kubectl changes

A new top-level command `kubectl debug` is added to create Debug Containers.
`kubectl describe pod` is extended to report the contents of
`DebugContainerStatuses` in addition to `ContainerStatuses and
InitContainerStatuses.`

Ideally there will be a way to hide this command when using clusters that do not
have alpha features enabled.

## Appendices

We've researched many options over the life of this proposal. These Appendices
are included as optional reference material. It's not necessary to read this
material in order to understand the proposal in its current form.

### Appendix 1: User Stories

These user stories are intended to give examples how this proposal addresses the
above requirements.

#### Operations

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

#### Debugging

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

#### Automation

Ginger is a security engineer tasked with running security audits across all of
her company's running containers. Even though his company has no standard base
image, she's able to audit all containers using:

```
% for pod in $(kubectl get -o name pod); do
    kubectl debug -m gcr.io/neato/security-audit -p $pod /security-audit.sh
  done
```

#### Technical Support

Roy's team provides support for his company's multi-tenant cluster. He can
access the Kubernetes API (as a viewer) on behalf of the users he's supporting,
but he does not have administrative access to nodes or a say in how the
application image is constructed. When someone asks for help, Roy's first step
is to run his team's autodiagnose script:

```
% kubectl debug --image=gcr.io/google_containers/autodiagnose nginx-pod-1234
```

### Appendix 2: Requirements Analysis

Many people have proposed alternate solutions to this problem. This section
discusses how the proposed solution meets all of the stated requirements and is
intended to contrast the alternatives listed below.

**Troubleshoot arbitrary running containers with minimal prior configuration.**
This solution requires no prior configuration.

**Access to all pod namespaces and the file systems of individual containers.**
This solution runs a container in the shared pod namespaces. It relies on the
behavior of `/proc/<pid>/root` (and therefore shared PID namespace) to provide
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

### Appendix 3: Alternatives Considered

#### Mutable Pod Spec

Rather than adding an operation to have Kubernetes attach a pod we could instead
make the pod spec mutable so the client can generate an update adding a
container. `SyncPod()` has no issues adding the container to the pod at that
point, but an immutable pod spec has been a basic assumption in Kubernetes thus
far and changing it carries risk. It's preferable to keep the pod spec immutable
as a best practice.

#### Ephemeral container

An earlier version of this proposal suggested running an ephemeral container in
the pod namespaces. The container would not be added to the pod spec and would
exist only as long as the process it ran. This has the advantage of behaving
similarly to the current kubectl exec, but it is opaque and likely violates
design assumptions. We could add constructs to track and report on both
traditional exec process and exec containers, but this would probably be more
work than adding to the pod spec. Both are generally useful, and neither
precludes the other in the future, so we chose mutating the pod spec for
expedience.

#### Attaching Container Type Volume

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

#### Inactive container

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

#### Implicit Empty Volume

Kubernetes could implicitly create an EmptyDir volume for every pod which would
then be available as target for either the kubelet or a sidecar to extract a
package of binaries.

Users would have to be responsible for hosting a package build and distribution
infrastructure or rely on a public one. The complexity of this solution makes it
undesirable.

#### Standalone Pod in Shared Namespace

Kubernetes could support starting a standalone pod that shares the namespace of
an existing pod.

This would be a small change to Kubernetes, but it would create edge cases in
the pod lifecycle that would have to be considered. For example, what happens to
the debugging pod when the target pod is destroyed?

#### Exec from Node

The kubelet could support executing a troubleshooting binary from the node in
the namespaces of the container. Once executed this binary would lose access to
other binaries from the node, making it of limited utility and a confusing user
experience.

This couples the debug tools with the lifecycle of the node, which is worse than
coupling it with container images.

## Reference

*   [Pod Troubleshooting Tracking Issue](https://issues.k8s.io/27140)
*   [CRI Tracking Issue](https://issues.k8s.io/28789)
*   [CRI: expose optional runtime features](https://issues.k8s.io/32803)
*   [Resource QoS in
    Kubernetes](https://github.com/kubernetes/kubernetes/blob/master/docs/design/resource-qos.md)
*   Related Features
    *   [#1615](https://issues.k8s.io/1615) - Shared PID Namespace across
        containers in a pod
    *   [#26751](https://issues.k8s.io/26751) - Pod-Level cgroup
    *   [#10782](https://issues.k8s.io/10782) - Vertical pod autoscaling
