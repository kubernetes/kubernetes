# Persistent Storage

This document proposes a model for managing persistent, cluster-scoped storage
for applications requiring long lived data.

### Abstract

Two new API kinds:

A `PersistentVolume` (PV) is a storage resource provisioned by an administrator.
It is analogous to a node. See [Persistent Volume Guide](../user-guide/persistent-volumes/)
for how to use it.

A `PersistentVolumeClaim` (PVC) is a user's request for a persistent volume to
use in a pod. It is analogous to a pod.

One new system component:

`PersistentVolumeClaimBinder` is a singleton running in master that watches all
PersistentVolumeClaims in the system and binds them to the closest matching
available PersistentVolume. The volume manager watches the API for newly created
volumes to manage.

One new volume:

`PersistentVolumeClaimVolumeSource` references the user's PVC in the same
namespace. This volume finds the bound PV and mounts that volume for the pod. A
`PersistentVolumeClaimVolumeSource` is, essentially, a wrapper around another
type of volume that is owned by someone else (the system).

Kubernetes makes no guarantees at runtime that the underlying storage exists or
is available. High availability is left to the storage provider.

### Goals

* Allow administrators to describe available storage.
* Allow pod authors to discover and request persistent volumes to use with pods.
* Enforce security through access control lists and securing storage to the same
namespace as the pod volume.
* Enforce quotas through admission control.
* Enforce scheduler rules by resource counting.
* Ensure developers can rely on storage being available without being closely
bound to a particular disk, server, network, or storage device.

#### Describe available storage

Cluster administrators use the API to manage *PersistentVolumes*. A custom store
`NewPersistentVolumeOrderedIndex` will index volumes by access modes and sort by
storage capacity. The `PersistentVolumeClaimBinder` watches for new claims for
storage and binds them to an available volume by matching the volume's
characteristics (AccessModes and storage size) to the user's request.

PVs are system objects and, thus, have no namespace.

Many means of dynamic provisioning will be eventually be implemented for various
storage types.


##### PersistentVolume API

| Action | HTTP Verb | Path | Description |
| ---- | ---- | ---- | ---- |
| CREATE | POST | /api/{version}/persistentvolumes/ | Create instance of PersistentVolume |
| GET | GET | /api/{version}persistentvolumes/{name} | Get instance of PersistentVolume with {name} |
| UPDATE | PUT | /api/{version}/persistentvolumes/{name} | Update instance of PersistentVolume with {name} |
| DELETE | DELETE | /api/{version}/persistentvolumes/{name} | Delete instance of PersistentVolume with {name} |
| LIST | GET | /api/{version}/persistentvolumes | List instances of PersistentVolume |
| WATCH | GET | /api/{version}/watch/persistentvolumes | Watch for changes to a PersistentVolume |


#### Request Storage

Kubernetes users request persistent storage for their pod by creating a
```PersistentVolumeClaim```. Their request for storage is described by their
requirements for resources and mount capabilities.

Requests for volumes are bound to available volumes by the volume manager, if a
suitable match is found. Requests for resources can go unfulfilled.

Users attach their claim to their pod using a new
```PersistentVolumeClaimVolumeSource``` volume source.


##### PersistentVolumeClaim API


| Action | HTTP Verb | Path | Description |
| ---- | ---- | ---- | ---- |
| CREATE | POST | /api/{version}/namespaces/{ns}/persistentvolumeclaims/ | Create instance of PersistentVolumeClaim in namespace {ns} |
| GET | GET | /api/{version}/namespaces/{ns}/persistentvolumeclaims/{name} | Get instance of PersistentVolumeClaim in namespace {ns} with {name} |
| UPDATE | PUT | /api/{version}/namespaces/{ns}/persistentvolumeclaims/{name} | Update instance of PersistentVolumeClaim in namespace {ns} with {name} |
| DELETE | DELETE | /api/{version}/namespaces/{ns}/persistentvolumeclaims/{name} | Delete instance of PersistentVolumeClaim in namespace {ns} with {name} |
| LIST | GET | /api/{version}/namespaces/{ns}/persistentvolumeclaims | List instances of PersistentVolumeClaim in namespace {ns} |
| WATCH | GET | /api/{version}/watch/namespaces/{ns}/persistentvolumeclaims | Watch for changes to PersistentVolumeClaim in namespace {ns} |



#### Scheduling constraints

Scheduling constraints are to be handled similar to pod resource constraints.
Pods will need to be annotated or decorated with the number of resources it
requires on a node. Similarly, a node will need to list how many it has used or
available.

TBD


#### Events

The implementation of persistent storage will not require events to communicate
to the user the state of their claim. The CLI for bound claims contains a
reference to the backing persistent volume. This is always present in the API
and CLI, making an event to communicate the same unnecessary.

Events that communicate the state of a mounted volume are left to the volume
plugins.

### Example

#### Admin provisions storage

An administrator provisions storage by posting PVs to the API. Various ways to
automate this task can be scripted. Dynamic provisioning is a future feature
that can maintain levels of PVs.

```yaml
POST:

kind: PersistentVolume
apiVersion: v1
metadata:
  name: pv0001
spec:
  capacity:
    storage: 10
  persistentDisk:
    pdName: "abc123"
    fsType: "ext4"
```

```console
$ kubectl get pv

NAME                LABELS              CAPACITY            ACCESSMODES         STATUS              CLAIM              REASON
pv0001              map[]               10737418240         RWO                 Pending    
```

#### Users request storage

A user requests storage by posting a PVC to the API. Their request contains the
AccessModes they wish their volume to have and the minimum size needed.

The user must be within a namespace to create PVCs.

```yaml
POST: 

kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: myclaim-1
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 3
```

```console
$ kubectl get pvc

NAME                LABELS              STATUS              VOLUME
myclaim-1           map[]               pending                         
```


#### Matching and binding

The ```PersistentVolumeClaimBinder``` attempts to find an available volume that
most closely matches the user's request. If one exists, they are bound by
putting a reference on the PV to the PVC. Requests can go unfulfilled if a
suitable match is not found.

```console
$ kubectl get pv

NAME                LABELS              CAPACITY            ACCESSMODES         STATUS              CLAIM                                                        REASON
pv0001              map[]               10737418240         RWO                 Bound               myclaim-1 / f4b3d283-c0ef-11e4-8be4-80e6500a981e


kubectl get pvc

NAME                LABELS              STATUS              VOLUME
myclaim-1           map[]               Bound               b16e91d6-c0ef-11e4-8be4-80e6500a981e
```

A claim must request access modes and storage capacity. This is because internally PVs are
indexed by their `AccessModes`, and target PVs are, to some degree, sorted by their capacity.
A claim may request one of more of the following attributes to better match a PV: volume name, selectors,
and volume class (currently implemented as an annotation).

A PV may define a `ClaimRef` which can greatly influence (but does not absolutely guarantee) which
PVC it will match.
A PV may also define labels, annotations, and a volume class (currently implemented as an
annotation) to better target PVCs.

As of Kubernetes version 1.4, the following algorithm describes in more details how a claim is
matched to a PV:

1. Only PVs with `accessModes` equal to or greater than the claim's requested `accessModes` are considered.
"Greater" here means that the PV has defined more modes than needed by the claim, but it also defines
the mode requested by the claim.

1. The potential PVs above are considered in order of the closest access mode match, with the best case
being an exact match, and a worse case being more modes than requested by the claim.

1. Each PV above is processed. If the PV has a `claimRef` matching the claim, *and* the PV's capacity
is not less than the storage being requested by the claim then this PV will bind to the claim. Done.

1. Otherwise, if the PV has the "volume.alpha.kubernetes.io/storage-class" annotation defined then it is
skipped and will be handled by Dynamic Provisioning.

1. Otherwise, if the PV has a `claimRef` defined, which can specify a different claim or simply be a
placeholder, then the PV is skipped.

1. Otherwise, if the claim is using a selector but it does *not* match the PV's labels (if any) then the
PV is skipped. But, even if a claim has selectors which match a PV that does not guarantee a match
since capacities may differ.

1. Otherwise, if the PV's "volume.beta.kubernetes.io/storage-class" annotation (which is a placeholder
for a volume class) does *not* match the claim's annotation (same placeholder) then the PV is skipped.
If the annotations for the PV and PVC are empty they are treated as being equal.

1. Otherwise, what remains is a list of PVs that may match the claim. Within this list of remaining PVs,
the PV with the smallest capacity that is also equal to or greater than the claim's requested storage
is the matching PV and will be bound to the claim. Done. In the case of two or more PVCs matching all
of the above criteria, the first PV (remember the PV order is based on `accessModes`) is the winner.

*Note:* if no PV matches the claim and the claim defines a `StorageClass` (or a default
`StorageClass` has been defined) then a volume will be dynamically provisioned.

#### Claim usage

The claim holder can use their claim as a volume.  The ```PersistentVolumeClaimVolumeSource``` knows to fetch the PV backing the claim
and mount its volume for a pod.

The claim holder owns the claim and its data for as long as the claim exists.
The pod using the claim can be deleted, but the claim remains in the user's
namespace. It can be used again and again by many pods.

```yaml
POST: 

kind: Pod
apiVersion: v1
metadata:
  name: mypod
spec:
  containers:
    - image: nginx
      name: myfrontend
      volumeMounts:
      - mountPath: "/var/www/html"
        name: mypd
  volumes:
    - name: mypd
      source:
        persistentVolumeClaim:
         accessMode: ReadWriteOnce
         claimRef:
           name: myclaim-1
```

#### Releasing a claim and Recycling a volume

When a claim holder is finished with their data, they can delete their claim.

```console
$ kubectl delete pvc myclaim-1
```

The ```PersistentVolumeClaimBinder``` will reconcile this by removing the claim
reference from the PV and change the PVs status to 'Released'.

Admins can script the recycling of released volumes. Future dynamic provisioners
will understand how a volume should be recycled.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/persistent-storage.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
