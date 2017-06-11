# Emptydir persistent volume

This document proposes a model for managing emptydir persistent volumes backed by node local storage.

## Abstract
We currently have:
* Various different types of "local volumes", like emptyDir, hostPath and gitRepo
* A way to "claim" storage, such that the pod referencing the claim can continue using the storage across pod/node restarts
* A way to dynamically provision volumes to match un-bound/pending claims

The second and third features allow us to say: provision a volume on demand, that survives pod/node restarts by using network attached storage (NFS, network block device from cloudprovider, distributed file system like gluster). This proposal is about a new `emptyDir` persistent volume that combines all three features. The idea is that we can sacrifice some durability to get a similar abstraction using *node local storage*. We expect this new PV source to be useful to applications that:
* Prioritize latency over durability (eg: async replicated database using local ssd)
* Prioritize cost over durability (eg: dev/test prototyping)
* Bring their own out of band save/restore snapshotting

## Goals
* Define a new persistent volume source that is backed by local storage and survives pod restarts
* Every pod referencing this PV must land on the same node (this concept has been referred to as "data gravity").
* Once this PV is bound to a node, it cannot get unbound till a human intervenes or the node is deleted
* The PV must require low user interaction. It should be provisionable dynamically.
* The directory backing the PV must get deleted when the PV is deleted.

## Non goals
* A local storage abstraction that survives extended node outages
* Snapshotting or simply copying the local volume to another node

## A new PV storage source

As described above, the storage source for the PV must be a local volume. While we already allow hostPath as a PersistentVolumeSource, it is insufficient because:
* user needs to specify the exact path, thus requires knowledge of the filesystem
* hard to dynamically provision a hostpath
* hostpath volumes are not cleaned up

The `emptydir` abstraction solves these problems:
* it is allocated in `/var/lib/kubelet/resource/uid/`, a location that is predictable but unknow till `resource` creation time
* The node takes care of allocation, a dynamic provisioner just needs to create the PV
* it is cleaned up when the pod dies (or in this case when the PV is deleted)

We can create a new [PersistetVolumeSource](https://github.com/kubernetes/kubernetes/blob/master/pkg/api/types.go#L240) for EmptyDir. Users will request storage by creating PersistetVolumeClaims. If the PVC isn't bound to a PV, one will be created dynamically. The user can't control which node the storage for this PV comes from, or where in the node's filesystem it is allocated. Users can also hand create PVs bound to specific nodes (TODO: elaborate on QoS and what happens if there's no space on the node).

## Scheduling constraints for data gravity

The scheduler will schedule the pod paying attention to any referenced PVs. If an empty dir PV attached to an existing node is found, that trumps all scheduler predicates. Otherwise the scheduler makes a scheduling decision taking into account pod cpu/memory requests (and in the future, requested storage), attaches it to a node, and all subsequent scheduling decisions for any pod referencing the same PV evaluate to the same node. The scheduler "Attaches" a pv to a node by writing an annotation (TBD what actually writes this annotation, scheduler or kubelet, scheduler will at least bind the pod to the node). All pods referencing the PV that enter the "Running" state on the give node will be able to write to the volume. If the node enters the `NotReady` state, all pods referencing the PV attached to the node will remain in `Pending` till such time when an admin either removes the annotation, or rewrites it to point to another node. The directory allocated to the emptydir volume will be exported through the status field of the PV, so admins can yank the disk from a NotReady node and rsync data to the same directory on another node before rewriting the annotation.

The annotations used by the scheduler to attach volumes to nodes will be: `volume.alpha.kubernetes.io/sticky-host`. The kubelet will reject any pod at admission time if it finds that the pod references a PV with this annotation set to a value other than its own node name.

## Creation and Deletion

The directory path allocated to an emptydir PV will be `/var/lib/kubelet/volumes/pv-uuid/`. This path will only get torn down when the PV is delete, or the `sticky-host` annotation on the PV is re-written to point to another node AND no pods are referencing the PV on the current node. It will NOT get deleted if all pods referencing the PV are deleted.

## AccesModes and Capacity

Capacity on the PV will be ignored till the scheduler can make decisions based on disk. AccessModes will default to `ReadWriteMany`. In the future the kubelet may enforce other AccessModes, but initially they will be ignored (TODO: readonly bind mounts?).

## Implementation

### Overview

A brief overview of the existing volume subsystem on the node:
```
                                 updates
                           asw <- - - - - -
                            |             |                -----
    populator -> dsw -> reconciler -> operation executor -| storage plugin      - attach/detach
                                                          | (emptydir, gce...)  - mount/unmount
                                                           -----
```

### Proposed flow

At a high level:
* Kubelet watches volumes (new)
* Populator observes pod attached to emptydir PV and adds volume spec to dsw (unchanged)
* Reconciler notices new volume spec in dsw not in asw, invokes new plugin (unchanged)
* New plugin attaches, mounts (new)
* Populator observes pod delete (unchanged)
* Populator deletes pod from dsw volume map, keeps volume (new, today it deletes volume from map when no pods are using it)
* Reconciler notices volume in asw with no matching pod in dsw, invokes new plugin (unchanged)
* New plugin unmounts (new)
* ...
* Kubelet notices pv deletion (new)
* Populator deletes volume from map (new)
* Reconciler notices volume in asw with no matching volume in dsw, invokes new plugin (unchanged)
* New plugin detaches (new)

### New storage plugin

We will write an attachable volume plugin that wraps emptydir. The directory is created and deleted in the Attach/Detach stage, BUT these operations are carried out through the same functions that implement emptydir Mount/Unmount. Attach must be idempotent, after creating the `/var/lib/kubelet/volumes/pv-uuid` directory, we record the volume as attached through the node status. Subsequent calls to Attach will no-op. Detach will only get called when the volume is removed from the desired state of the world, which by our design won't happen till the PV is deleted. The permissions on this directory will be the same as emptydir (i.e 777).

This storage plugin will respond "yes" when asked if it owns a volume spec that contains a PV + `sticky-host` annotation, and "no" in every other case. (TODO: do we need to bind mount the directory for every pod?).

## Example

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: test
  annotations:
    # Set by scheduler or admin
    volume.alpha.kubernetes.io/sticky-host: ""
spec:
  capacity:
    storage: 10Mi
  accessModes:
    - ReadWriteMany
  emptyDir: {}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: test
  annotations:
    # Mentioned for the sake of clarity
    volume.alpha.kubernetes.io/storage-class: free
spec:
  accessModes: [ "ReadWriteMany" ]
  resources:
    requests:
      storage: 10Mi
---
apiVersion: v1
kind: Pod
metadata:
  generateName: test-
spec:
  terminationGracePeriodSeconds: 0
  containers:
  - name: busybox
    image: busybox
    command:
    - sh
    - -c
    - touch /data/`hostname`; while true; do sleep 3600; done
    volumeMounts:
    - name: data
      mountPath: /data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: test
```

Creating this will give you a pod pinned to the node the scheduler initally picks, with a directory mounted at /data backed by `/var/lib/kubelet/volumes/pv-uuid`. Creating the same thing without the PV section will invoke a dynamic storage provisioner to create an identical PV.

## Alternatives

I (@bprashanth) belive that a sticky emptydir volume as proposed strikes a good balance between hostPath local storage and a spectrum of other possibilities such as:
* Hostpath dynamic provisioner: The petset team is still considering a short term solution that involves a dynamic provisioner creating a PV backed by hostPath, pointing to the same `/var/lib/kubelet/volumes/pv-uuid` location. This requires 0 code modifications in the node volume subsystem, but requires the scheduler and kubelet admission time changes (relatively simple). This also means the sticky volumes will not get cleaned up, so it is not a viable long term plan.
* Streamline distributed filesystem deployment on Kubernetes: If we handwave the performance benefits of using local storage (writing to local disk is ~ going over a LAN), we could in theory make it really easy to deploy eg: Gluster and leverage local storage. While this is still a possibility, it introduces another failure mode, and distributed filesystems are far from being easy to deploy. It also significantly raises the bar for the dev/test use case.
* Use an NFS RC: This is already possible, but still doesn't give us the performance benefit, and is tedious to setup (though not as tedious as a distributed filesystem).
