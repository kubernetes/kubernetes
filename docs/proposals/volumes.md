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

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/proposals/volumes.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Abstract

A proposal for sharing volumes between containers in a pod using a special supplemental group.

## Motivation

Kubernetes volumes should be usable regardless of the UID a container runs as.  This concern cuts
across all volume types, so the system should be able to handle them in a generalized way to provide
uniform functionality across all volume types and lower the barrier to new plugins.

Goals of this design:

1.  Enumerate the different use-cases for volume usage in pods
2.  Define the desired goal state for ownership and permission management in Kubernetes
3.  Describe the changes necessary to achieve desired state

## Constraints and Assumptions

1.  When writing permissions in this proposal, `D` represents a don't-care value; example: `07D0`
    represents permissions where the owner has `7` permissions, all has `0` permissions, and group
    has a don't-care value
2.  Read-write usability of a volume from a container is defined as one of:
    1.  The volume is owned by the container's effective UID and has permissions `07D0`
    2.  The volume is owned by the container's effective GID or one of its supplemental groups and
        has permissions `0D70`
3.  Volume plugins should not have to handle setting permissions on volumes
5.  Preventing two containers within a pod from reading and writing to the same volume (by choosing
    different container UIDs) is not something we intend to support today
6.  We will not design to support multiple processes running in a single container as different
    UIDs; use cases that require work by different UIDs should be divided into different pods for
    each UID

## Current State Overview

### Kubernetes

Kubernetes volumes can be divided into two broad categories:

1.  Unshared storage:
    1.  Volumes created by the kubelet on the host directory: empty directory, git repo, secret,
        downward api.  All volumes in this category delegate to `EmptyDir` for their underlying
        storage.  These volumes are created with ownership `root:root`.
    2.  Volumes based on network block devices: AWS EBS, iSCSI, RBD, etc, *when used exclusively
        by a single pod*.
2.  Shared storage:
    1.  `hostPath` is shared storage because it is necessarily used by a container and the host
    2.  Network file systems such as NFS, Glusterfs, Cephfs, etc.  For these volumes, the ownership
        is determined by the configuration of the shared storage system.
    3.  Block device based volumes in `ReadOnlyMany` or `ReadWriteMany` modes are shared because
        they may be used simultaneously by multiple pods.

The `EmptyDir` volume was recently modified to create the volume directory with `0777` permissions
from `0750` to support basic usability of that volume as a non-root UID.

### Docker

Docker recently added supplemental group support.  This adds the ability to specify additional
groups that a container should be part of, and will be released with Docker 1.8.

There is a [proposal](https://github.com/docker/docker/pull/14632) to add a bind-mount flag to tell
Docker to change the ownership of a volume to the effective UID and GID of a container, but this has
not yet been accepted.

### rkt

rkt
[image manifests](https://github.com/appc/spec/blob/master/spec/aci.md#image-manifest-schema) can
specify users and groups, similarly to how a Docker image can.  A rkt
[pod manifest](https://github.com/appc/spec/blob/master/spec/pods.md#pod-manifest-schema) can also
override the default user and group specified by the image manifest.

rkt does not currently support supplemental groups or changing the owning UID or
group of a volume, but it has been [requested](https://github.com/coreos/rkt/issues/1309).

## Use Cases

1.  As a user, I want the system to set ownership and permissions on volumes correctly to enable
    reads and writes with the following scenarios:
    1.  All containers running as root
    2.  All containers running as the same non-root user
    3.  Multiple containers running as a mix of root and non-root users

### All containers running as root

For volumes that only need to be used by root, no action needs to be taken to change ownership or
permissions, but setting the ownership based on the supplemental group shared by all containers in a
pod will also work.  For situations where read-only access to a shared volume is required from one
or more containers, the `VolumeMount`s in those containers should have the `readOnly` field set.

### All containers running as a single non-root user

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

1.  It grants powerful kernel capabilities to the code in the image and thus is not securing,
    defeating the reason containers are run as non-root users
2.  The user experience is poor; it requires changing Dockerfile, adding a layer, or modifying the
    container's command

Some cluster operators manage the ownership of shared storage volumes on the server side.
In this scenario, the UID of the container using the volume is known in advance.  The ownership of
the volume is set to match the container's UID on the server side.

### Containers running as a mix of root and non-root users

If the list of UIDs that need to use a volume includes both root and non-root users, supplemental
groups can be applied to enable sharing volumes between containers.  The ownership and permissions
`root:<supplemental group> 2770` will make a volume usable from both containers running as root and
running as a non-root UID and the supplemental group.  The setgid bit is used to ensure that files
created in the volume will inherit the owning GID of the volume.

## Community Design Discussion

- [kubernetes/2630](https://github.com/kubernetes/kubernetes/issues/2630)
- [kubernetes/11319](https://github.com/kubernetes/kubernetes/issues/11319)
- [kubernetes/9384](https://github.com/kubernetes/kubernetes/pull/9384)

## Analysis

The system needs to be able to:

1.  Model correctly which volumes require ownership management
1.  Determine the correct ownership of each volume in a pod if required
1.  Set the ownership and permissions on volumes when required

### Modeling whether a volume requires ownership management

#### Unshared storage: volumes derived from `EmptyDir`

Since Kubernetes creates `EmptyDir` volumes, it should ensure the ownership is set to enable the
volumes to be usable for all of the above scenarios.

#### Unshared storage: network block devices

Volume plugins based on network block devices such as AWS EBS and RBS can be treated the same way
as local volumes.  Since inodes are written to these block devices in the same way as `EmptyDir`
volumes, permissions and ownership can be managed on the client side by the Kubelet when used
exclusively by one pod.  When the volumes are used outside of a persistent volume, or with the
`ReadWriteOnce` mode, they are effectively unshared storage.

When used by multiple pods, there are many additional use-cases to analyze before we can be
confident that we can support ownership management robustly with these file systems.  The right
design is one that makes it easy to experiment and develop support for ownership management with
volume plugins to enable developers and cluster operators to continue exploring these issues.

#### Shared storage: hostPath

The `hostPath` volume should only be used by effective-root users, and the permissions of paths
exposed into containers via hostPath volumes should always be managed by the cluster operator.  If
the Kubelet managed the ownership for `hostPath` volumes, a user who could create a `hostPath`
volume could affect changes in the state of arbitrary paths within the host's filesystem.  This
would be a severe security risk, so we will consider hostPath a corner case that the kubelet should
never perform ownership management for.

#### Shared storage

Ownership management of shared storage is a complex topic.  Ownership for existing shared storage
will be managed externally from Kubernetes.  For this case, our API should make it simple to express
whether a particular volume should have these concerns managed by Kubernetes.

We will not attempt to address the ownership and permissions concerns of new shared storage
in this proposal.

When a network block device is used as a persistent volume in `ReadWriteMany` or `ReadOnlyMany`
modes, it is shared storage, and thus outside the scope of this proposal.

#### Plugin API requirements

From the above, we know that some volume plugins will 'want' ownership management from the Kubelet
and others will not.  Plugins should be able to opt in to ownership management from the Kubelet.  To
facilitate this, there should be a method added to the `volume.Plugin` interface that the Kubelet
uses to determine whether to perform ownership management for a volume.

### Determining correct ownership of a volume

Using the approach of a pod-level supplemental group to own volumes solves the problem in any of the
cases of UID/GID combinations within a pod. Since this is the simplest approach that handles all
use-cases, our solution will be made in terms of it.

Eventually, Kubernetes should allocate a unique group for each pod so that a pod's volumes are
usable by that pod's containers, but not by containers of another pod.  The supplemental group used
to share volumes must be unique in a multitenant cluster.  If uniqueness is enforced at the host
level, pods from one host may be able to use shared filesystems meant for pods on another host.

Eventually, Kubernetes should integrate with external identity management systems to populate pod
specs with the right supplemental groups necessary to use shared volumes.  In the interim until the
identity management story is far enough along to implement this type of integration, we will rely
on being able to set arbitrary groups.  (Note: as of this writing, a PR is being prepared for
setting arbitrary supplemental groups).

An admission controller could handle allocating groups for each pod and setting the group in the
pod's security context.

#### A note on the root group

Today, by default, all docker containers are run in the root group (GID 0).  This is relied on by
image authors that make images to run with a range of UIDs: they set the group ownership for
important paths to be the root group, so that containers running as GID 0 *and* an arbitrary UID
can read and write to those paths normally.

It is important to note that the changes proposed here will not affect the primary GID of
containers in pods.  Setting the `pod.Spec.SecurityContext.FSGroup` field will not
override the primary GID and should be safe to use in images that expect GID 0.

### Setting ownership and permissions on volumes

For `EmptyDir`-based volumes and unshared storage, `chown` and `chmod` on the node are sufficient to
set ownership and permissions.  Shared storage is different because:

1.  Shared storage may not live on the node a pod that uses it runs on
2.  Shared storage may be externally managed

## Proposed design:

Our design should minimize code for handling ownership required in the Kubelet and volume plugins.

### API changes

We should not interfere with images that need to run as a particular UID or primary GID.  A pod
level supplemental group allows us to express a group that all containers in a pod run as in a way
that is orthogonal to the primary UID and GID of each container process.

```go
package api

type PodSecurityContext struct {
    // FSGroup is a supplemental group that all containers in a pod run under.  This group will own
    // volumes that the Kubelet manages ownership for.  If this is not specified, the Kubelet will
    // not set the group ownership of any volumes.
    FSGroup *int64 `json:"fsGroup,omitempty"`
}
```

The V1 API will be extended with the same field:

```go
package v1

type PodSecurityContext struct {
    // FSGroup is a supplemental group that all containers in a pod run under.  This group will own
    // volumes that the Kubelet manages ownership for.  If this is not specified, the Kubelet will
    // not set the group ownership of any volumes.
    FSGroup *int64 `json:"fsGroup,omitempty"`
}
```

The values that can be specified for the `pod.Spec.SecurityContext.FSGroup` field are governed by
[pod security policy](https://github.com/kubernetes/kubernetes/pull/7893).

#### API backward compatibility

Pods created by old clients will have the `pod.Spec.SecurityContext.FSGroup` field unset;
these pods will not have their volumes managed by the Kubelet.  Old clients will not be able to set
or read the `pod.Spec.SecurityContext.FSGroup` field.

### Volume changes

The `volume.Mounter` interface should have a new method added that indicates whether the plugin
supports ownership management:

```go
package volume

type Mounter interface {
    // other methods omitted

    // SupportsOwnershipManagement indicates that this volume supports having ownership
    // and permissions managed by the Kubelet; if true, the caller may manipulate UID
    // or GID of this volume.
    SupportsOwnershipManagement() bool
}
```

In the first round of work, only `hostPath` and `emptyDir` and its derivations will be tested with
ownership management support:

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

### Kubelet changes

The Kubelet should be modified to perform ownership and label management when required for a volume.

For ownership management the criteria are:

1.  The `pod.Spec.SecurityContext.FSGroup` field is populated
2.  The volume builder returns `true` from `SupportsOwnershipManagement`

Logic should be added to the `mountExternalVolumes` method that runs a local `chgrp` and `chmod` if
the pod-level supplemental group is set and the volume supports ownership management:

```go
package kubelet

type ChgrpRunner interface {
    Chgrp(path string, gid int) error
}

type ChmodRunner interface {
    Chmod(path string, mode os.FileMode) error
}

type Kubelet struct {
    chgrpRunner ChgrpRunner
    chmodRunner ChmodRunner
}

func (kl *Kubelet) mountExternalVolumes(pod *api.Pod) (kubecontainer.VolumeMap, error) {
    podFSGroup = pod.Spec.PodSecurityContext.FSGroup
    podFSGroupSet := false
    if podFSGroup != 0 {
        podFSGroupSet = true
    }

    podVolumes := make(kubecontainer.VolumeMap)

    for i := range pod.Spec.Volumes {
        volSpec := &pod.Spec.Volumes[i]

        rootContext, err := kl.getRootDirContext()
        if err != nil {
            return nil, err
        }

        // Try to use a plugin for this volume.
        internal := volume.NewSpecFromVolume(volSpec)
        builder, err := kl.newVolumeMounterFromPlugins(internal, pod, volume.VolumeOptions{RootContext: rootContext}, kl.mounter)
        if err != nil {
            glog.Errorf("Could not create volume builder for pod %s: %v", pod.UID, err)
            return nil, err
        }
        if builder == nil {
            return nil, errUnsupportedVolumeType
        }
        err = builder.SetUp()
        if err != nil {
            return nil, err
        }

        if builder.SupportsOwnershipManagement() &&
           podFSGroupSet {
            err = kl.chgrpRunner.Chgrp(builder.GetPath(), podFSGroup)
            if err != nil {
                return nil, err
            }

            err = kl.chmodRunner.Chmod(builder.GetPath(), os.FileMode(1770))
            if err != nil {
                return nil, err
            }
        }

        podVolumes[volSpec.Name] = builder
    }

    return podVolumes, nil
}
```

This allows the volume plugins to determine when they do and don't want this type of support from
the Kubelet, and allows the criteria each plugin uses to evolve without changing the Kubelet.

The docker runtime will be modified to set the supplemental group of each container based on the
`pod.Spec.SecurityContext.FSGroup` field.  Theoretically, the `rkt` runtime could support this
feature in a similar way.

### Examples

#### EmptyDir

For a pod that has two containers sharing an `EmptyDir` volume:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  securityContext:
    fsGroup: 1001
  containers:
  - name: a
    securityContext:
      runAsUser: 1009
    volumeMounts:
      - mountPath: "/example/hostpath/a"
        name: empty-vol
  - name: b
    securityContext:
      runAsUser: 1010
    volumeMounts:
      - mountPath: "/example/hostpath/b"
        name: empty-vol
  volumes:
    - name: empty-vol
```

When the Kubelet runs this pod, the `empty-vol` volume will have ownership root:1001 and permissions
`0770`.  It will be usable from both containers a and b.

#### HostPath

For a volume that uses a `hostPath` volume with containers running as different UIDs:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  securityContext:
    fsGroup: 1001
  containers:
  - name: a
    securityContext:
      runAsUser: 1009
    volumeMounts:
      - mountPath: "/example/hostpath/a"
        name: host-vol
  - name: b
    securityContext:
      runAsUser: 1010
    volumeMounts:
      - mountPath: "/example/hostpath/b"
        name: host-vol
  volumes:
    - name: host-vol
      hostPath:
        path: "/tmp/example-pod"
```

The cluster operator would need to manually `chgrp` and `chmod` the `/tmp/example-pod` on the host
in order for the volume to be usable from the pod.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/volumes.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
