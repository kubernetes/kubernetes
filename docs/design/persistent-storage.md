# PersistentVolume

This document proposes a model for managing persistent, cluster-scoped storage for applications requiring long lived data.

### tl;dr

Two new API kinds:

A `PersistentVolume` is created by a cluster admin and is a piece of persistent storage exposed as a volume.  It is analogous to a node.

A `PersistentVolumeClaim` is a user's request for a persistent volume to use in a pod. It is analogous to a pod.  

One new system component:

`PersistentVolumeManager` watches for new volumes to manage in the system, analogous to the node controller.  The volume manager also watches for claims by users and binds them to available volumes.

Kubernetes makes no guarantees at runtime that the underlying storage exists or is available.  High availability is left to the storage provider.

### Goals

* Allow administrators to describe available storage
* Allow pod authors to discover and request persistent volumes to use with pods
* Enforce security through access control lists and securing storage to the same namespace as the pod volume
* Enforce quotas through admission control
* Enforce scheduler rules by resource counting
* Ensure developers can rely on storage being available without being closely bound to a particular disk, server, network, or storage device.


#### Describe available storage

Cluster adminstrators use the API to manage *PersistentVolumes*.  A manager watches for new volumes and adds them to the available supply of volumes.  All persistent volumes are managed and made available by the volume manager.  The manager also watches for new claims for storage and binds them to an available, matching volume.

Many means of dynamic provisioning will be eventually be implemented for various storage types. 

```

	$ cluster/kubectl.sh get pv

```

##### API Implementation:

| Action | HTTP Verb | Path | Description |
| ---- | ---- | ---- | ---- |
| CREATE | POST | /api/{version}/persistentvolumes/ | Create instance of PersistentVolume in system namespace  |
| GET | GET | /api/{version}persistentvolumes/{name} | Get instance of PersistentVolume in system namespace with {name} |
| UPDATE | PUT | /api/{version}/persistentvolumes/{name} | Update instance of PersistentVolume in system namespace with {name} |
| DELETE | DELETE | /api/{version}/persistentvolumes/{name} | Delete instance of PersistentVolume in system namespace with {name} |
| LIST | GET | /api/{version}/persistentvolumes | List instances of PersistentVolume in system namespace |
| WATCH | GET | /api/{version}/watch/persistentvolumes | Watch for changes to a PersistentVolume in system namespace |



#### Request Storage


Kubernetes users request a persistent volume for their pod by creating a *PersistentVolumeClaim*.  Their request for storage is described by their requirements for resource and mount capabilities.

Requests for volumes are bound to available volumes by the volume manager, if a suitable match is found.  Requests for resources can go unfulfilled.

Users attach their claim to their pod using a new *PersistentVolumeClaimVolumeSource* volume source.


##### Users require a full API to manage their claims.


| Action | HTTP Verb | Path | Description |
| ---- | ---- | ---- | ---- |
| CREATE | POST | /api/{version}/ns/{ns}/persistentvolumeclaims/ | Create instance of PersistentVolumeClaim in namespace {ns} |
| GET | GET | /api/{version}/ns/{ns}/persistentvolumeclaims/{name} | Get instance of PersistentVolumeClaim in namespace {ns} with {name} |
| UPDATE | PUT | /api/{version}/ns/{ns}/persistentvolumeclaims/{name} | Update instance of PersistentVolumeClaim in namespace {ns} with {name} |
| DELETE | DELETE | /api/{version}/ns/{ns}/persistentvolumeclaims/{name} | Delete instance of PersistentVolumeClaim in namespace {ns} with {name} |
| LIST | GET | /api/{version}/ns/{ns}/persistentvolumeclaims | List instances of PersistentVolumeClaim in namespace {ns} |
| WATCH | GET | /api/{version}/watch/ns/{ns}/persistentvolumeclaims | Watch for changes to PersistentVolumeClaim in namespace {ns} |



#### Scheduling constraints

Scheduling constraints are to be handled similar to pod resource constraints.  Pods will need to be annotated or decorated with the number of resources it requires on a node.  Similarly, a node will need to list how many it has used or available.

TBD

