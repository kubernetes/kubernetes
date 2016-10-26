## Abstract

Presents a proposal for enhancing the security of Kubernetes clusters using
SELinux and simplifying the implementation of SELinux support within the
Kubelet by removing the need to label the Kubelet directory with an SELinux
context usable from a container.

## Motivation

The current Kubernetes codebase relies upon the Kubelet directory being
labeled with an SELinux context usable from a container.  This means that a
container escaping namespace isolation will be able to use any file within the
Kubelet directory without defeating kernel
[MAC (mandatory access control)](https://en.wikipedia.org/wiki/Mandatory_access_control).
In order to limit the attack surface, we should enhance the Kubelet to relabel
any bind-mounts into containers into a usable SELinux context without depending
on the Kubelet directory's SELinux context.

## Constraints and Assumptions

1.  No API changes allowed
2.  Behavior must be fully backward compatible
3.  No new admission controllers - make incremental improvements without huge
    refactorings

## Use Cases

1.  As a cluster operator, I want to avoid having to label the Kubelet
    directory with a label usable from a container, so that I can limit the
    attack surface available to a container escaping its namespace isolation
2.  As a user, I want to run a pod without an SELinux context explicitly
    specified and be isolated using MCS (multi-category security) on systems
    where SELinux is enabled, so that the pods on each host are isolated from
    one another
3.  As a user, I want to run a pod that uses the host IPC or PID namespace and
    want the system to do the right thing with regard to SELinux, so that no
    unnecessary relabel actions are performed

### Labeling the Kubelet directory

As previously stated, the current codebase relies on the Kubelet directory
being labeled with an SELinux context usable from a container.  The Kubelet
uses the SELinux context of this directory to determine what SELinux context
`tmpfs` mounts (provided by the EmptyDir memory-medium option) should receive.
The problem with this is that it opens an attack surface to a container that
escapes its namespace isolation; such a container would be able to use any
file in the Kubelet directory without defeating kernel MAC.

### SELinux when no context is specified

When no SELinux context is specified, Kubernetes should just do the right
thing, where doing the right thing is defined as isolating pods with a node-
unique set of categories.  Node-uniqueness means unique among the pods
scheduled onto the node.  Long-term, we want to have a cluster-wide allocator
for MCS labels. Node-unique MCS labels are a good middle ground that is
possible without a new, large, feature.

### SELinux and host IPC and PID namespaces

Containers in pods that use the host IPC or PID namespaces need access to
other processes and IPC mechanisms on the host.  Therefore, these containers
should be run with the `spc_t` SELinux type by the container runtime.  The
`spc_t` type is an unconfined type that other SELinux domains are allowed to
connect to.  In the case where a pod uses one of these host namespaces, it
should be unnecessary to relabel the pod's volumes.

## Analysis

### Libcontainer SELinux library

Docker and rkt both use the libcontainer SELinux library.  This library
provides a method, `GetLxcContexts`, that returns the a unique SELinux
contexts for container processes and files used by them.  `GetLxcContexts`
reads the base SELinux context information from a file at `/etc/selinux/<policy-
name>/contexts/lxc_contexts` and then adds a process-unique MCS label.

Docker and rkt both leverage this call to determine the 'starting' SELinux
contexts for containers.

### Docker

Docker's behavior when no SELinux context is defined for a container is to
give the container a node-unique MCS label.

#### Sharing IPC namespaces

On the Docker runtime, the containers in a Kubernetes pod share the IPC and
PID namespaces of the pod's infra container.

Docker's behavior for containers sharing these namespaces is as follows: if a
container B shares the IPC namespace of another container A, container B is
given the SELinux context of container A.  Therefore, for Kubernetes pods
running on docker, in a vacuum the containers in a pod should have the same
SELinux context.

[**Known issue**](https://bugzilla.redhat.com/show_bug.cgi?id=1377869): When
the seccomp profile is set on a docker container that shares the IPC namespace
of another container, that container will not receive the other container's
SELinux context.

#### Host IPC and PID namespaces

In the case of a pod that shares the host IPC or PID namespace, this flag is
simply ignored and the container receives the `spc_t` SELinux type.  The
`spc_t` type is unconfined, and so no relabeling needs to be done for volumes
for these pods.  Currently, however, there is code which relabels volumes into
explicitly specified SELinux contexts for these pods. This code is unnecessary
and should be removed.

#### Relabeling bind-mounts

Docker is capable of relabeling bind-mounts into containers using the `:Z`
bind-mount flag.  However, in the current implementation of the docker runtime
in Kubernetes, the `:Z` option is only applied when the pod's SecurityContext
contains an SELinux context.  We could easily implement the correct behaviors
by always setting `:Z` on systems where SELinux is enabled.

### rkt

rkt's behavior when no SELinux context is defined for a pod is similar to
Docker's -- an SELinux context with a node-unique MCS label is given to the
containers of a pod.

#### Sharing IPC namespaces

Containers (apps, in rkt terminology) in rkt pods share an IPC and PID
namespace by default.

#### Relabeling bind-mounts

Bind-mounts into rkt pods are automatically relabeled into the pod's SELinux
context.

#### Host IPC and PID namespaces

Using the host IPC and PID namespaces is not currently supported by rkt.

## Proposed Changes

### Refactor `pkg/util/selinux`

1.  The `selinux` package should provide a method `SELinuxEnabled` that returns
    whether SELinux is enabled, and is built for all platforms (the
    libcontainer SELinux is only built on linux)
2.  The `SelinuxContextRunner` interface should be renamed to `SELinuxRunner`
    and be changed to have the same method names and signatures as the
    libcontainer methods its implementations wrap
3.  The `SELinuxRunner` interface only needs `Getfilecon`, which is used by
    the rkt code

```go
package selinux

// Note: the libcontainer SELinux package is only built for Linux, so it is
// necessary to have a NOP wrapper which is built for non-Linux platforms to
// allow code that links to this package not to differentiate its own methods
// for Linux and non-Linux platforms.
//
// SELinuxRunner wraps certain libcontainer SELinux calls. For more
// information, see:
//
// https://github.com/opencontainers/runc/blob/master/libcontainer/selinux/selinux.go
type SELinuxRunner interface {
	// Getfilecon returns the SELinux context for the given path or returns an
	// error.
	Getfilecon(path string) (string, error)
}
```

### Kubelet Changes

1.  The `relabelVolumes` method in `kubelet_volumes.go` is not needed and can
    be removed
2.  The `GenerateRunContainerOptions` method in `kubelet_pods.go` should no
    longer call `relabelVolumes`
3.  The `makeHostsMount` method in `kubelet_pods.go` should set the
    `SELinuxRelabel` attribute of the mount for the pod's hosts file to `true`

### Changes to `pkg/kubelet/dockertools/`

1.  The `makeMountBindings` should be changed to:
  1.  No longer accept the `podHasSELinuxLabel` parameter
  2.  Always use the `:Z` bind-mount flag when SELinux is enabled and the mount
      has the `SELinuxRelabel` attribute set to `true`
2.  The `runContainer` method should be changed to always use the `:Z`
    bind-mount flag on the termination message mount when SELinux is enabled

### Changes to `pkg/kubelet/rkt`

The should not be any required changes for the rkt runtime; we should test to
ensure things work as expected under rkt.

### Changes to volume plugins and infrastructure

1.  The `VolumeHost` interface contains a method called `GetRootContext`; this
    is an artifact of the old assumptions about the Kubelet directory's SELinux
    context and can be removed
2.  The `empty_dir.go` file should be changed to be completely agnostic of
    SELinux; no behavior in this plugin needs to be differentiated when SELinux
    is enabled

### Changes to `pkg/controller/...`

The `VolumeHost` abstraction is used in a couple of PV controllers as NOP
implementations.  These should be altered to no longer include `GetRootContext`.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/selinux-enhancements.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
