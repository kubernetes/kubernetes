## Abstract

A proposal for enabling containers in a pod to share volumes using a pod level SELinux context.

## Motivation

Many users have a requirement to run pods on systems that have SELinux enabled.  Volume plugin
authors should not have to explicitly account for SELinux except for volume types that require
special handling of the SELinux context during setup.

Currently, each container in a pod has an SELinux context.  This is not an ideal factoring for
sharing resources using SELinux.

We propose a pod-level SELinux context and a mechanism to support SELinux labeling of volumes in a
generic way.

Goals of this design:

1.  Describe the problems with a container SELinux context
2.  Articulate a design for generic SELinux support for volumes using a pod level SELinux context
    which is backward compatible with the v1.0.0 API

## Constraints and Assumptions

1.  We will not support securing containers within a pod from one another
2.  Volume plugins should not have to handle setting SELinux context on volumes
3.  We will not deal with shared storage

## Current State Overview

### Docker

Docker uses a base SELinux context and calculates a unique MCS label per container.  The SELinux
context of a container can be overridden with the `SecurityOpt` api that allows setting the different
parts of the SELinux context individually.

Docker has functionality to relabel bind-mounts with a usable SElinux and supports two different
use-cases:

1.  The `:Z` bind-mount flag, which tells Docker to relabel a bind-mount with the container's
    SELinux context
2.  The `:z` bind-mount flag, which tells Docker to relabel a bind-mount with the container's
    SElinux context, but remove the MCS labels, making the volume shareable between containers

We should avoid using the `:z` flag, because it relaxes the SELinux context so that any container
(from an SELinux standpoint) can use the volume.

### rkt

rkt currently reads the base SELinux context to use from `/etc/selinux/*/contexts/lxc_contexts`
and allocates a unique MCS label per pod.

### Kubernetes


There is a [proposed change](https://github.com/kubernetes/kubernetes/pull/9844) to the
EmptyDir plugin that adds SELinux relabeling capabilities to that plugin, which is also carried as a
patch in [OpenShift](https://github.com/openshift/origin).  It is preferable to solve the problem
in general of handling SELinux in kubernetes to merging this PR.

A new `PodSecurityContext` type has been added that carries information about security attributes
that apply to the entire pod and that apply to all containers in a pod.  See:

1.  [Skeletal implementation](https://github.com/kubernetes/kubernetes/pull/13939)
1.  [Proposal for inlining container security fields](https://github.com/kubernetes/kubernetes/pull/12823)

## Use Cases

1.  As a cluster operator, I want to support securing pods from one another using SELinux when
    SELinux integration is enabled in the cluster
2.  As a user, I want volumes sharing to work correctly amongst containers in pods

#### SELinux context: pod- or container- level?

Currently, SELinux context is specifiable only at the container level.  This is an inconvenient
factoring for sharing volumes and other SELinux-secured resources between containers because there
is no way in SELinux to share resources between processes with different MCS labels except to
remove MCS labels from the shared resource.  This is a big security risk: _any container_ in the
system can work with a resource which has the same SELinux context as it and no MCS labels.  Since
we are also not interested in isolating containers in a pod from one another, the SELinux context
should be shared by all containers in a pod to facilitate isolation from the containers in other
pods and sharing resources amongst all the containers of a pod.

#### Volumes

Kubernetes volumes can be divided into two broad categories:

1.  Unshared storage:
    1.  Volumes created by the kubelet on the host directory: empty directory, git repo, secret,
        downward api.  All volumes in this category delegate to `EmptyDir` for their underlying
        storage.
    2.  Volumes based on network block devices: AWS EBS, iSCSI, RBD, etc, *when used exclusively
        by a single pod*.
2.  Shared storage:
    1.  `hostPath` is shared storage because it is necessarily used by a container and the host
    2.  Network file systems such as NFS, Glusterfs, Cephfs, etc.
    3.  Block device based volumes in `ReadOnlyMany` or `ReadWriteMany` modes are shared because
        they may be used simultaneously by multiple pods.

For unshared storage, SELinux handling for most volumes can be generalized into running a `chcon`
operation on the volume directory after running the volume plugin's `Setup` function.  For these
volumes, the Kubelet can perform the `chcon` operation and keep SELinux concerns out of the volume
plugin code.  Some volume plugins may need to use the SELinux context during a mount operation in
certain cases.  To account for this, our design must have a way for volume plugins to state that
a particular volume should or should not receive generic label management.

For shared storage, the picture is murkier.  Labels for existing shared storage will be managed
outside Kubernetes and administrators will have to set the SELinux context of pods correctly.
The problem of solving SELinux label management for new shared storage is outside the scope for
this proposal.

## Analysis

The system needs to be able to:

1.  Model correctly which volumes require SELinux label management
1.  Relabel volumes with the correct SELinux context when required

### Modeling whether a volume requires label management

#### Unshared storage: volumes derived from `EmptyDir`

Empty dir and volumes derived from it are created by the system, so Kubernetes must always ensure
that the ownership and SELinux context (when relevant) are set correctly for the volume to be
usable.

#### Unshared storage: network block devices

Volume plugins based on network block devices such as AWS EBS and RBS can be treated the same way
as local volumes.  Since inodes are written to these block devices in the same way as `EmptyDir`
volumes, permissions and ownership can be managed on the client side by the Kubelet when used
exclusively by one pod.  When the volumes are used outside of a persistent volume, or with the
`ReadWriteOnce` mode, they are effectively unshared storage.

When used by multiple pods, there are many additional use-cases to analyze before we can be
confident that we can support SELinux label management robustly with these file systems.  The right
design is one that makes it easy to experiment and develop support for ownership management with
volume plugins to enable developers and cluster operators to continue exploring these issues.

#### Shared storage: hostPath

The `hostPath` volume should only be used by effective-root users, and the permissions of paths
exposed into containers via hostPath volumes should always be managed by the cluster operator.  If
the Kubelet managed the SELinux labels for `hostPath` volumes, a user who could create a `hostPath`
volume could affect changes in the state of arbitrary paths within the host's filesystem.  This
would be a severe security risk, so we will consider hostPath a corner case that the kubelet should
never perform ownership management for.

#### Shared storage: network

Ownership management of shared storage is a complex topic.  SELinux labels for existing shared
storage will be managed externally from Kubernetes.  For this case, our API should make it simple to
express whether a particular volume should have these concerns managed by Kubernetes.

We will not attempt to address the concerns of new shared storage in this proposal.

When a network block device is used as a persistent volume in `ReadWriteMany` or `ReadOnlyMany`
modes, it is shared storage, and thus outside the scope of this proposal.

#### API requirements

From the above, we know that label management must be applied:

1.  To some volume types always
2.  To some volume types never
3.  To some volume types *sometimes*

Volumes should be relabeled with the correct SELinux context.  Docker has this capability today; it
is desirable for other container runtime implementations to provide similar functionality.

Relabeling should be an optional aspect of a volume plugin to accommodate:

1.  volume types for which generalized relabeling support is not sufficient
2.  testing for each volume plugin individually

## Proposed Design

Our design should minimize code for handling SELinux labelling required in the Kubelet and volume
plugins.

### Deferral: MCS label allocation

Our short-term goal is to facilitate volume sharing and isolation with SELinux and expose the
primitives for higher level composition; making these automatic is a longer-term goal.  Allocating
groups and MCS labels are fairly complex problems in their own right, and so our proposal will not
encompass either of these topics.  There are several problems that the solution for allocation
depends on:

1.  Users and groups in Kubernetes
2.  General auth policy in Kubernetes
3.  [security policy](https://github.com/kubernetes/kubernetes/pull/7893)

### API changes

The [inline container security attributes PR (12823)](https://github.com/kubernetes/kubernetes/pull/12823)
adds a `pod.Spec.SecurityContext.SELinuxOptions` field.  The change to the API in this proposal is
the addition of the semantics to this field:

* When the `pod.Spec.SecurityContext.SELinuxOptions` field is set, volumes that support ownership
management in the Kubelet have their SELinuxContext set from this field.

```go
package api

type PodSecurityContext struct {
    // SELinuxOptions captures the SELinux context for all containers in a Pod.  If a container's
    // SecurityContext.SELinuxOptions field is set, that setting has precedent for that container.
    //
    // This field will be used to set the SELinux of volumes that support SELinux label management
    // by the kubelet.
    SELinuxOptions *SELinuxOptions `json:"seLinuxOptions,omitempty"`
}
```

The V1 API is extended with the same semantics:

```go
package v1

type PodSecurityContext struct {
    // SELinuxOptions captures the SELinux context for all containers in a Pod.  If a container's
    // SecurityContext.SELinuxOptions field is set, that setting has precedent for that container.
    //
    // This field will be used to set the SELinux of volumes that support SELinux label management
    // by the kubelet.
    SELinuxOptions *SELinuxOptions `json:"seLinuxOptions,omitempty"`
}
```

#### API backward compatibility

Old pods that do not have the `pod.Spec.SecurityContext.SELinuxOptions` field set will not receive
SELinux label management for their volumes.  This is acceptable since old clients won't know about
this field and won't have any expectation of their volumes being managed this way.

The existing backward compatibility semantics for SELinux do not change at all with this proposal.

### Kubelet changes

The Kubelet should be modified to perform SELinux label management when required for a volume.  The
criteria to activate the kubelet SELinux label management for volumes are:

1.  SELinux integration is enabled in the cluster
2.  SELinux is enabled on the node
3.  The `pod.Spec.SecurityContext.SELinuxOptions` field is set
4.  The volume plugin supports SELinux label management

The `volume.Mounter` interface should have a new method added that indicates whether the plugin
supports SELinux label management:

```go
package volume

type Builder interface {
    // other methods omitted
    SupportsSELinux() bool
}
```

Individual volume plugins are responsible for correctly reporting whether they support label
management in the kubelet.  In the first round of work, only `hostPath` and `emptyDir` and its
derivations will be tested with ownership management support:

| Plugin Name             | SupportsOwnershipManagement   |
|-------------------------|-------------------------------|
| `hostPath`              | false                         |
| `emptyDir`              | true                          |
| `gitRepo`               | true                          |
| `secret`                | true                          |
| `downwardAPI`           | true                          |
| `gcePersistentDisk`     | false                         |
| `awsElasticBlockStore`  | false                         |
| `nfs`                   | false                         |
| `iscsi`                 | false                         |
| `glusterfs`             | false                         |
| `persistentVolumeClaim` | depends on underlying volume and PV mode |
| `rbd`                   | false                         |
| `cinder`                | false                         |
| `cephfs`                | false                         |

Ultimately, the matrix will theoretically look like:

| Plugin Name             | SupportsOwnershipManagement   |
|-------------------------|-------------------------------|
| `hostPath`              | false                         |
| `emptyDir`              | true                          |
| `gitRepo`               | true                          |
| `secret`                | true                          |
| `downwardAPI`           | true                          |
| `gcePersistentDisk`     | true                          |
| `awsElasticBlockStore`  | true                          |
| `nfs`                   | false                         |
| `iscsi`                 | true                          |
| `glusterfs`             | false                         |
| `persistentVolumeClaim` | depends on underlying volume and PV mode |
| `rbd`                   | true                          |
| `cinder`                | false                         |
| `cephfs`                | false                         |

In order to limit the amount of SELinux label management code in Kubernetes, we propose that it be a
function of the container runtime implementations.  Initially, we will modify the docker runtime
implementation to correctly set the `:Z` flag on the appropriate bind-mounts in order to accomplish
generic label management for docker containers.

Volume types that require SELinux context information at mount must be injected with and respect the
enablement setting for the labeling for the volume type.  The proposed `VolumeConfig` mechanism
will be used to carry information about label management enablement to the volume plugins that have
to manage labels individually.

This allows the volume plugins to determine when they do and don't want this type of support from
the Kubelet, and allows the criteria each plugin uses to evolve without changing the Kubelet.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/selinux.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
