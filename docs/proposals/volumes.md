## Abstract

A proposal for enabling generalized volume usage in advanced scenarios such as containers running
under non-root UIDs or with SELinux enabled.

## Motivation

Kubernetes volumes should be usable regardless of the UID a container runs as or whether SELinux
is enabled.  These scenarios cut across all volume types, so the system should be able to handle
them in a generalized way to provide uniform functionality across all volume types and lower the
barrier to new types.

Goals of this design:

1.  Enumerate the different use-cases for volumes
2.  Define the desired goal state for volumes in Kubernetes
3.  Describe a short-term approach for enabling achieveable use-cases
4.  Describe gaps that must be closed to achieve long-term goals

## Constraints and Assumptions

1.  When writing permissions in this proposal, `D` represents a don't-care value; example: `07D0`
    represents permissions where the owner has `7` permissions, all has `0` permissions, and group
    has a don't-care value
2.  Read-write usability of a volume from a container is defined as one of:
    1.  The volume is owned by the container's effective UID and has permissions `07D0`
    2.  The volume is owned by the container's effective GID or one of its supplemental groups and
        has permissions `0D70`
3.  Read-only usability of a volume from a container is defined as one of:
    1.  The volume is not owned by the container and has permissions `0DD5`
    2.  The volume is owned by the container's effective UID and has permissions `05DD`
    3.  The volume is owned the container's effective GID or one of its supplemental groups and has
        permissions `0D5D`
4.  Volume plugins should not have to handle setting permissions on volumes
5.  Volume plugins should not have to handle SELinux unless it is unavoidable during volume setup

## Current State Overview

### Kubernetes

Kubernetes volumes can be divided into two broad categories:

1.  Volumes created by the kubelet on the host directory: empty directory, git repo, secret,
    downward api ([proposed](https://github.com/GoogleCloudPlatform/kubernetes/pull/5093)).  All
    volumes in this category delegate to `EmptyDir` for their underlying storage.  These volumes are
    created with ownership `root:root`.

2.  Distributed filesystems: AWS EBS, iSCSI, RBD, NFS, Glusterfs, GCE PD.  For these volumes, the
    ownership is determined by the underlying filesystem.

The `EmptyDir` volume was recently modified to create the volume directory with `0777` permissions
from `0750` to support basic usability of that volume as a non-root UID.  The `EmptyDir` volume also
has basic SELinux support, in that it creates the volume directory using the SELinux context of the
Kubelet volume directory if SELinux is enabled.

There is a [proposed change](https://github.com/GoogleCloudPlatform/kubernetes/pull/9844) to the
EmptyDir plugin that adds SELinux relabeling capabilities to that plugin, which is also carried as a
patch in [OpenShift](https://github.com/openshift/origin).

### Docker

#### UID/GID

Docker recently added supplemental group support.  This adds the ability to specify additional
groups that a container should be part of, and will be released with Docker 1.8.

There is a [proposal](https://github.com/docker/docker/pull/14632) to add a bind-mount flag to tell
Docker to change the ownership of a volume to the effective UID and GID of a container, but this has
not yet been accepted.

#### SELinux

Docker uses a base SELinux context and calculates a unique MCS label per container.  The SELinux
context of a container can be overriden with the `SecurityOpt` api that allows setting the different
parts of the SELinux context individually.

Docker has functionality to relabel bind-mounts with a usable SElinux and supports two different
use-cases:

1.  The `:Z` bind-mount flag, which tells Docker to relabel a bind-mount with the container's
    SELinux context
2.  The `:z` bind-mount flag, which tells Docker to relabel a bind-mount with the container's
    SElinux context, but remove the MCS labels, making the volume shareable beween containers

### Rocket

#### UID/GID

Rocket
[image manifests](https://github.com/appc/spec/blob/master/spec/aci.md#image-manifest-schema) can
specify users and groups, similarly to how a Docker image can.  A Rocket
[pod manifest](https://github.com/appc/spec/blob/master/spec/pods.md#pod-manifest-schema) can also
override the default user and group specified by the image manifest.

Rocket does not currently support supplemental groups or changing the owning UID or
group of a volume.

#### SELinux

Rocket currently reads the base SELinux context to use from `/etc/selinux/*/contexts/lxc_contexts`
and allocates a unique MCS label per pod.

## Use Cases

1.  As a user, I want the system to set ownership and permissions on volumes correctly to enabled
the following scenarios:
    1.  All containers running as root
    4.  All containers running as the same non-root user
    5.  Multiple containers running as a mix of root and non-root users
2.  As a user, I want all use-cases to work properly on systems where SELinux is enabled

### Ownership and permissions

#### All containers running as root

For volumes that only need to be used by root, no action needs to be taken to change ownership or
permissions.  For situations where read-only access to a shared volume is required from one or more
containers, the `VolumeMount`s in those containers should have the `readOnly` field set.

#### All containers running as a single non-root user

In use cases whether a volume is used by a single non-root UID the volume ownership and permissions
should be set to enable read/write access.

Currently, a non-root UID will not have permissions to write to any but an `EmptyDir` volume.
Today, users that need this case to work can:

1.  Grant the container the necessary capabilities to `chown` and `chmod` the volume:
    - `CAP_FOWNER`
    - `CAP_CHOWN`
    - `CAP_DAC_OVERRIDE`
2.  Run a wrapper script that runs `chown` and `chmod` commands to set the desired ownership and
    permissions on the volume before starting their main process

This workaround has significant drawbacks:

1.  It grants powerful kernel capabilities to the code in the image
2.  The user experience is poor; it requires changing Dockerfile, adding a layer, or modifying the
    container's command

#### Containers running as a mix of root and non-root users

If the list of UIDs that need to use a volume includes both root and non-root users, supplemental
groups can be applied to enable sharing volumes between containers.  The ownership and permissions
`root:<supplemental group> 0770` will make a volume usable from both containers running as root and
running as a non-root UID and the supplemental group.

### Volumes and SELinux

Many users have a requirement to run pods on systems that have SELinux enabled.  Volume plugin
authors should not have to explicitly account for SELinux except for volume types that require
special handling of the SELinux context during setup.

SELinux handling for most volumes can be generalized into running a `chcon` operation on the volume
directory after running the volume plugin's `Setup` function, but there is at least one exception.
For NFS volumes, the `context` flag must be passed to `mount`, or the `virt_use_nfs` SELinux boolean
set.  If a system administrator does not wish to set `virt_use_nfs`, the correct context must be
passed to the `mount` operation in order for the volume to be usable from a container's SELinux
policy on certain systems.

We can generalize the requirements for SELinux handling as follows:

1.  For a volume used by a single container, the volume should be relabeled with the SELinux context
    of that container
2.  For a volume shared by multiple containers, the volume should be relabed with an SELinux context
    usable by all containers in the pod

## Community Design Discussion

- [kubernetes/2630](https://github.com/GoogleCloudPlatform/kubernetes/issues/2630)
- [kubernetes/11319](https://github.com/GoogleCloudPlatform/kubernetes/issues/11319)
- [kubernetes/9384](https://github.com/GoogleCloudPlatform/kubernetes/pull/9384)

## Analysis

The system needs to be able to:

1.  Determine the correct ownership and permissions of each volume in a pod
1.  Set the ownership and permissions on volumes
1.  Relabel volumes with the correct SELinux context

### Determining correct ownership

The Kubelet must analyze the pod spec to determine which UIDs need to use which volumes.  If a
container's security context's `RunAsUser` field is not set, the Kubelet must inspect the image via
the container runtime to determine which UID the image will run as.  Once the list of UIDs that need
to use a volume is known, the kubelet can determine which ownership and permissions should be used
to make the volume functional.

If a volume is used only by a single UID within the pod, the ownership can be set to that UID.
Otherwise, the volume should be owned by a group and the containers run in that group.

#### Handling non-numeric UIDs

If a non-numeric user is specified by an image, the behavior of container runtimes is to look up the
UID from the container's `/etc/passwd` file.  It is not feasible for the kubelet to make this
determination; we may not be able to correctly support non-numeric users in image metadata.  If the
kubelet encounters this scenario, it should be an error, and an event created.

### Setting ownership and permissions on volumes

Once the correct ownership and permissions are determined for the volume, the system must ensure
that these are in place on the volume.  Eventually, it would be desireable for the container runtime
to manage ownership and permissions on volumes.  In the short term, the Kubelet will have to perform
a chown and chmod on the volume directory.

#### `chown`, `chmod`, and distributed filesystems

The success of `chown` and `chmod` operations on distributed filesystems can depend on the
configuration of the server hosting the volumes.  It may not be appropriate to perform chown or
chmod operations on some distributed file systems.  For example, actions taken by root on an NFS
client are treated as `nobody` on the server unless the `no_squash_root` setting is enabled for that
volume's NFS export.  Enabling `no_squash_root` contradicts Red Hat's security guidance for NFS (as
an example), so there will need to be another mechanism that prepares some distributed fs volumes.

One possibility is that if the server hosting the filesystem is itself a Kubernetes node, a pod
could be scheduled onto that node that prepared the volume by setting the ownership, permissions,
and SELinux context.

### Relabeling volumes with correct SELinux context

On systems with SELinux enabled, volumes should be relabeled with the correct SELinux context.
Docker has this capability today; it is desireable for other container runtime implementations to
provide similar functionality.

For the docker runtime, the Kubelet should determine whether each volume is shared by more than one
container and set the correct bind-mount flags on the docker container configuration.  If a single
container uses a volume, the `:Z` flag should be used; if the volume is shared by multiple
containers, the `:z` flag should be used.

Relabeling should be an optional aspect of a volume plugin to accomodate:

1.  volume types for which generalized relabeling support is not desired
2.  testing for each volume plugin individually

#### SELinux and distributed file systems

Some distributed filesystems have complications where SELinux is involved.  Let's look at NFS as an
example.  Files in NFS volumes cannot have their contexts changed with `chcon` after the volume is
mounted; the `context` argument must be passed to the `mount` call.

For NFS, there is an SELinux boolean, `virt_use_nfs` that allows containers to use files with the
SELinux type `nfs_t`.  Setting `virt_use_nfs` on a system would allow containers to use *any* NFS
mount without the kubelet having to relabel the files.  However, the semantics would be different
from what might be expected, since SELinux will not provide any protection against containers from
one pod using files in an NFS mount belonging to another pod.

NFS is a case where the `chcon` approach isn't going to solve the whole problem.  Instead, the NFS
volume plugin will have to handle calling `mount` with the `context` argument in order to gain
cross-container protection from SELinux.

TODO: analyses of other file systems and SELinux
TODO: generalize to classes of situations:
TODO:  1.  chcon is ok
TODO:  2.  chcon won't work, mount flag required
TODO:  3.  chcon and mount flag won't work, SELinux boolean must be turned on

#### The `hostPath` volume and SELinux

One volume plugin that requires careful consideration with regard to SELinux support is `hostPath`.
This plugin allows a user to mount an arbitrary path on the host.  If `hostPath` supports
relabeling, it would effectively allow the users to force a change in arbitrary locations on a node.
It is safest to make `hostPath` not support relabelling and let administrators set the security
context on files and directories used with the `hostPath` plugin manually.

## Proposed Design

Our proposed design should minimize code for handling ownership and SELinux required in:

1.  volume plugins
2.  the kubelet

### Kubelet changes

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/volumes.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
    