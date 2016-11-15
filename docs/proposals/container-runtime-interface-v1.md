# Redefine Container Runtime Interface

The umbrella issue: [#22964](https://issues.k8s.io/22964)

## Motivation

Kubelet employs a declarative pod-level interface, which acts as the sole
integration point for container runtimes (e.g., `docker` and `rkt`). The
high-level, declarative interface has caused higher integration and maintenance
cost, and also slowed down feature velocity for the following reasons.
  1. **Not every container runtime supports the concept of pods natively**.
     When integrating with Kubernetes, a significant amount of work needs to
     go into implementing a shim of significant size to support all pod
     features. This also adds maintenance overhead (e.g., `docker`).
  2. **High-level interface discourages code sharing and reuse among runtimes**.
     E.g, each runtime today implements an all-encompassing `SyncPod()`
     function, with the Pod Spec as the input argument. The runtime implements
     logic to determine how to achieve the desired state based on the current
     status, (re-)starts pods/containers and manages lifecycle hooks
     accordingly.
  3. **Pod Spec is evolving rapidly**. New features are being added constantly.
     Any pod-level change or addition requires changing of all container
     runtime shims. E.g., init containers and volume containers.

## Goals and Non-Goals

The goals of defining the interface are to
 - **improve extensibility**: Easier container runtime integration.
 - **improve feature velocity**
 - **improve code maintainability**

The non-goals include
 - proposing *how* to integrate with new runtimes, i.e., where the shim
   resides. The discussion of adopting a client-server architecture is tracked
   by [#13768](https://issues.k8s.io/13768), where benefits and shortcomings of
   such an architecture is discussed.
 - versioning the new interface/API. We intend to provide API versioning to
   offer stability for runtime integrations, but the details are beyond the
   scope of this proposal.
 - adding support to Windows containers. Windows container support is a
   parallel effort and is tracked by [#22623](https://issues.k8s.io/22623).
   The new interface will not be augmented to support Windows containers, but
   it will be made extensible such that the support can be added in the future.
 - re-defining Kubelet's internal interfaces. These interfaces, though, may
   affect Kubelet's maintainability, is not relevant to runtime integration.
 - improving Kubelet's efficiency or performance, e.g., adopting event stream
   from the container runtime [#8756](https://issues.k8s.io/8756),
   [#16831](https://issues.k8s.io/16831).

## Requirements

 * Support the already integrated container runtime: `docker` and `rkt`
 * Support hypervisor-based container runtimes: `hyper`.

The existing pod-level interface will remain as it is in the near future to
ensure supports of all existing runtimes are continued. Meanwhile, we will
work with all parties involved to switching to the proposed interface.


## Container Runtime Interface

The main idea of this proposal is to adopt an imperative container-level
interface, which allows Kubelet to directly control the lifecycles of the
containers.

Pod is composed of a group of containers in an isolated environment with
resource constraints. In Kubernetes, pod is also the smallest schedulable unit.
After a pod has been scheduled to the node, Kubelet will create the environment
for the pod, and add/update/remove containers in that environment to meet the
Pod Spec. To distinguish between the environment and the pod as a whole, we
will call the pod environment **PodSandbox.**

The container runtimes may interpret the PodSandBox concept differently based
on how it operates internally. For runtimes relying on hypervisor, sandbox
represents a virtual machine naturally. For others, it can be Linux namespaces.

In short, a PodSandbox should have the following features.

 * **Isolation**: E.g., Linux namespaces or a full virtual machine, or even
   support additional security features.
 * **Compute resource specifications**: A PodSandbox should implement pod-level
   resource demands and restrictions.

*NOTE: The resource specification does not include externalized costs to
container setup that are not currently trackable as Pod constraints, e.g.,
filesystem setup, container image pulling, etc.*

A container in a PodSandbox maps to an application in the Pod Spec. For Linux
containers, they are expected to share at least network and IPC namespaces,
with sharing more namespaces discussed in [#1615](https://issues.k8s.io/1615).


Below is an example of the proposed interfaces.

```go
// PodSandboxManager contains basic operations for sandbox.
type PodSandboxManager interface {
    Create(config *PodSandboxConfig) (string, error)
    Delete(id string) (string, error)
    List(filter PodSandboxFilter) []PodSandboxListItem
    Status(id string) PodSandboxStatus
}

// ContainerRuntime contains basic operations for containers.
type ContainerRuntime interface {
    Create(config *ContainerConfig, sandboxConfig *PodSandboxConfig, PodSandboxID string) (string, error)
    Start(id string) error
    Stop(id string, timeout int) error
    Remove(id string) error
    List(filter ContainerFilter) ([]ContainerListItem, error)
    Status(id string) (ContainerStatus, error)
    Exec(id string, cmd []string, streamOpts StreamOptions) error
}

// ImageService contains image-related operations.
type ImageService interface {
    List() ([]Image, error)
    Pull(image ImageSpec, auth AuthConfig) error
    Remove(image ImageSpec) error
    Status(image ImageSpec) (Image, error)
    Metrics(image ImageSpec) (ImageMetrics, error)
}

type ContainerMetricsGetter interface {
    ContainerMetrics(id string) (ContainerMetrics, error)
}

All functions listed above are expected to be thread-safe.
```

### Pod/Container Lifecycle

The PodSandbox’s lifecycle is decoupled from the containers, i.e., a sandbox
is created before any containers, and can exist after all containers in it have
terminated.

Assume there is a pod with a single container C. To start a pod:

```
  create sandbox Foo --> create container C --> start container C
```

To delete a pod:

```
  stop container C --> remove container C --> delete sandbox Foo
```

The container runtime must not apply any transition (such as starting a new
container) unless explicitly instructed by Kubelet. It is Kubelet's
responsibility to enforce garbage collection, restart policy, and otherwise
react to changes in lifecycle.

The only transitions that are possible for a container are described below:

```
() -> Created        // A container can only transition to created from the
                     // empty, nonexistent state. The ContainerRuntime.Create
                     // method causes this transition.
Created -> Running   // The ContainerRuntime.Start method may be applied to a
                     // Created container to move it to Running
Running -> Exited    // The ContainerRuntime.Stop method may be applied to a running 
                     // container to move it to Exited.
                     // A container may also make this transition under its own volition 
Exited -> ()         // An exited container can be moved to the terminal empty
                     // state via a ContainerRuntime.Remove call.
```


Kubelet is also responsible for gracefully terminating all the containers
in the sandbox before deleting the sandbox. If Kubelet chooses to delete
the sandbox with running containers in it, those containers should be forcibly
deleted.

Note that every PodSandbox/container lifecycle operation (create, start,
stop, delete) should either return an error or block until the operation
succeeds. A successful operation should include a state transition of the
PodSandbox/container. E.g., if a `Create` call for a container does not
return an error, the container state should be "created" when the runtime is
queried.

### Updates to PodSandbox or Containers

Kubernetes support updates only to a very limited set of fields in the Pod
Spec.  These updates may require containers to be re-created by Kubelet. This
can be achieved through the proposed, imperative container-level interface.
On the other hand, PodSandbox update currently is not required.


### Container Lifecycle Hooks

Kubernetes supports post-start and pre-stop lifecycle hooks, with ongoing
discussion for supporting pre-start and post-stop hooks in
[#140](https://issues.k8s.io/140).

These lifecycle hooks will be implemented by Kubelet via `Exec` calls to the
container runtime. This frees the runtimes from having to support hooks
natively.

Illustration of the container lifecycle and hooks:

```
            pre-start post-start    pre-stop post-stop
               |        |              |       |
              exec     exec           exec    exec
               |        |              |       |
 create --------> start ----------------> stop --------> remove
```

In order for the lifecycle hooks to function as expected, the `Exec` call
will need access to the container's filesystem (e.g., mount namespaces).

### Extensibility

There are several dimensions for container runtime extensibility.
 - Host OS (e.g., Linux)
 - PodSandbox isolation mechanism (e.g., namespaces or VM)
 - PodSandbox OS (e.g., Linux)

As mentioned previously, this proposal will only address the Linux based
PodSandbox and containers. All Linux-specific configuration will be grouped
into one field. A container runtime is required to enforce all configuration
applicable to its platform, and should return an error otherwise.

### Keep it minimal

The proposed interface is experimental, i.e., it will go through (many) changes
until it stabilizes. The principle is to to keep the interface minimal and
extend it later if needed. This includes a several features that are still in
discussion and may be achieved alternatively:

 * `AttachContainer`: [#23335](https://issues.k8s.io/23335)
 * `PortForward`: [#25113](https://issues.k8s.io/25113)

## Alternatives

**[Status quo] Declarative pod-level interface**
 - Pros: No changes needed.
 - Cons: All the issues stated in #motivation

**Allow integration at both pod- and container-level interfaces**
 - Pros: Flexibility.
 - Cons: All the issues stated in #motivation

**Imperative pod-level interface**
The interface contains only CreatePod(), StartPod(), StopPod() and RemovePod().
This implies that the runtime needs to take over container lifecycle
management (i.e., enforce restart policy), lifecycle hooks, liveness checks,
etc. Kubelet will mainly be responsible for interfacing with the apiserver, and
can potentially become a very thin daemon.
 - Pros: Lower maintenance overhead for the Kubernetes maintainers if `Docker`
   shim maintenance cost is discounted.
 - Cons: This will incur higher integration cost because every new container
   runtime needs to implement all the features and need to understand the
   concept of pods. This would also lead to lower feature velocity because the
   interface will need to be changed, and the new pod-level feature will need
   to be supported in each runtime.

## Related Issues

 * Metrics: [#27097](https://issues.k8s.io/27097)
 * Log management: [#24677](https://issues.k8s.io/24677)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/container-runtime-interface-v1.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
