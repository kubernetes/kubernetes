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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/persistent-volumes.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Persistent Volumes and Claims

This document describes the current state of `PersistentVolumes` in Kubernetes.  Familiarity with [volumes](volumes.md) is suggested.

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Persistent Volumes and Claims](#persistent-volumes-and-claims)
  - [Introduction](#introduction)
  - [Lifecycle of a volume and claim](#lifecycle-of-a-volume-and-claim)
    - [Provisioning](#provisioning)
    - [Binding](#binding)
    - [Using](#using)
    - [Releasing](#releasing)
    - [Reclaiming](#reclaiming)
  - [Types of Persistent Volumes](#types-of-persistent-volumes)
  - [Persistent Volumes](#persistent-volumes)
    - [Capacity](#capacity)
    - [Access Modes](#access-modes)
    - [Recycling Policy](#recycling-policy)
    - [Phase](#phase)
  - [PersistentVolumeClaims](#persistentvolumeclaims)
    - [Access Modes](#access-modes)
    - [Resources](#resources)
  - [Claims As Volumes](#claims-as-volumes)

<!-- END MUNGE: GENERATED_TOC -->

## Introduction

Managing storage is a distinct problem from managing compute. The `PersistentVolume` subsystem provides an API for users and administrators that abstracts details of how storage is provided from how it is consumed.  To do this we introduce two new API resources:  `PersistentVolume` and `PersistentVolumeClaim`.

A `PersistentVolume` (PV) is a piece of networked storage in the cluster that has been provisioned by an administrator.  It is a resource in the cluster just like a node is a cluster resource.   PVs are volume plugins like Volumes, but have a lifecycle independent of any individual pod that uses the PV.  This API object captures the details of the implementation of the storage, be that NFS, iSCSI, or a cloud-provider-specific storage system.

A `PersistentVolumeClaim` (PVC) is a request for storage by a user.  It is similar to a pod.  Pods consume node resources and PVCs consume PV resources.  Pods can request specific levels of resources (CPU and Memory).  Claims can request specific size and access modes (e.g, can be mounted once read/write or many times read-only).

Please see the [detailed walkthrough with working examples](persistent-volumes/).


## Lifecycle of a volume and claim

PVs are resources in the cluster.  PVCs are requests for those resources and also act as claim checks to the resource.  The interaction between PVs and PVCs follows this lifecycle:

### Provisioning

A cluster administrator will create a number of PVs. They carry the details of the real storage which is available for use by cluster users.  They exist in the Kubernetes API and are available for consumption.

### Binding

A user creates a `PersistentVolumeClaim` with a specific amount of storage requested and with certain access modes.  A control loop in the master watches for new PVCs, finds a matching PV (if possible), and binds them together.  The user will always get at least what they asked for, but the volume may be in excess of what was requested.

Claims will remain unbound indefinitely if a matching volume does not exist.  Claims will be bound as matching volumes become available.  For example, a cluster provisioned with many 50Gi PVs would not match a PVC requesting 100Gi.  The PVC can be bound when a 100Gi PV is added to the cluster.

### Using

Pods use claims as volumes. The cluster inspects the claim to find the bound volume and mounts that volume for a pod.  For volumes which support multiple access modes, the user specifies which mode desired when using their claim as a volume in a pod.

Once a user has a claim and that claim is bound, the bound PV belongs to the user for as long as they need it. Users schedule Pods and access their claimed PVs by including a persistentVolumeClaim in their Pod's volumes block. [See below for syntax details](#claims-as-volumes).

### Releasing

When a user is done with their volume, they can delete the PVC objects from the API which allows reclamation of the resource.  The volume is considered "released" when the claim is deleted, but it is not yet available for another claim.  The previous claimant's data remains on the volume which must be handled according to policy.

### Reclaiming

The reclaim policy for a `PersistentVolume` tells the cluster what to do with the volume after it has been released.  Currently, volumes can either be Retained or Recycled.  Retention allows for manual reclamation of the resource.  For those volume plugins that support it, recycling performs a basic scrub (`rm -rf /thevolume/*`) on the volume and makes it available again for a new claim.

## Types of Persistent Volumes

`PersistentVolume` types are implemented as plugins.  Kubernetes currently supports the following plugins:

* GCEPersistentDisk
* AWSElasticBlockStore
* NFS
* iSCSI
* RBD (Ceph Block Device)
* Glusterfs
* HostPath (single node testing only -- local storage is not supported in any way and WILL NOT WORK in a multi-node cluster)


## Persistent Volumes

Each PV contains a spec and status, which is the specification and status of the volume.


```yaml
  apiVersion: v1
  kind: PersistentVolume
  metadata:
    name: pv0003
  spec:
    capacity:
      storage: 5Gi
    accessModes:
      - ReadWriteOnce
    persistentVolumeReclaimPolicy: Recycle
    nfs:
      path: /tmp
      server: 172.17.0.2
```

### Capacity

Generally, a PV will have a specific storage capacity.  This is set using the PV's `capacity` attribute.  See the Kubernetes [Resource Model](../design/resources.md) to understand the units expected by `capacity`.

Currently, storage size is the only resource that can be set or requested.  Future attributes may include IOPS, throughput, etc.

### Access Modes

A `PersistentVolume` can be mounted on a host in any way supported by the resource provider.  Providers will have different capabilities and each PV's access modes are set to the specific modes supported by that particular volume.  For example, NFS can support multiple read/write clients, but a specific NFS PV might be exported on the server as read-only.  Each PV gets its own set of access modes describing that specific PV's capabilities.

The access modes are:

* ReadWriteOnce -- the volume can be mounted as read-write by a single node
* ReadOnlyMany -- the volume can be mounted read-only by many nodes
* ReadWriteMany -- the volume can be mounted as read-write by many nodes

In the CLI, the access modes are abbreviated to:

* RWO - ReadWriteOnce
* ROX - ReadOnlyMany
* RWX - ReadWriteMany

> __Important!__ A volume can only be mounted using one access mode at a time, even if it supports many.  For example, a GCEPersistentDisk can be mounted as ReadWriteOnce by a single node or ReadOnlyMany by many nodes, but not at the same time.


### Recycling Policy

Current recycling policies are:

* Retain -- manual reclamation
* Recycle -- basic scrub ("rm -rf /thevolume/*")

Currently, NFS and HostPath support recycling.

### Phase

A volume will be in one of the following phases:

* Available -- a free resource that is not yet bound to a claim
* Bound -- the volume is bound to a claim
* Released -- the claim has been deleted, but the resource is not yet reclaimed by the cluster
* Failed -- the volume has failed its automatic reclamation

The CLI will show the name of the PVC bound to the PV.

## PersistentVolumeClaims

Each PVC contains a spec and status, which is the specification and status of the claim.

```yaml
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: myclaim
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 8Gi
```

### Access Modes

Claims use the same conventions as volumes when requesting storage with specific access modes.

### Resources

Claims, like pods, can request specific quantities of a resource.  In this case, the request is for storage.  The same [resource model](../design/resources.md) applies to both volumes and claims.

## Claims As Volumes

Pods access storage by using the claim as a volume.  Claims must exist in the same namespace as the pod using the claim.  The cluster finds the claim in the pod's namespace and uses it to get the `PersistentVolume` backing the claim.  The volume is then mounted to the host and into the pod.

```yaml
kind: Pod
apiVersion: v1
metadata:
  name: mypod
spec:
  containers:
    - name: myfrontend
      image: dockerfile/nginx
      volumeMounts:
      - mountPath: "/var/www/html"
        name: mypd
  volumes:
    - name: mypd
      persistentVolumeClaim:
        claimName: myclaim
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/persistent-volumes.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
