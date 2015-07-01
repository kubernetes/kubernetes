# Persistent Volumes and Claims

This document describes the current state of Persistent Volumes in Kubernetes.  Familiarity with [volumes](./volumes.md) is suggested.

A Persistent Volume (PV) is a piece of networked storage in the cluster that has been provisioned by an administrator.  It is a resource in the cluster just like a node is a cluster resource.   PVs are volume plugins like Volumes, but have a lifecycle independent of any individual pod that uses the PV.

A Persistent Volume Claim (PVC) is a request for storage by a user.  It is similar to a pod.  Pods consume node resources and PVCs consume PV resources.  Pods can request specific levels of resources (CPU and Memory).  Claims can request specific size and access modes (e.g, can be mounted once read/write or many times read-only).  

Please see the [detailed walkthrough with working examples](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/examples/persistent-volumes).


## Lifecycle of a volume and claim

PVs are resources in the cluster.  PVC are requests for those resources and also act as claim checks to the resource.  The interaction between PVs and PVCs follows this lifecycle:

### Provisioning
	
The volume is created by an administrator.  It becomes a cluster resource available for consumption.

### Binding

A persistent volume claim is created by a user requesting a specific amount of storage and with certain access modes.  There is a process watching for new claims that binds them to an available volume if a match is available.  The user will always get at least what they asked for, but the volume may be in excess of what was requested.

Claims will remain unbound indefinitely if a matching volume does not exist.  Claims will be bound as matching volumes become available.  For example, a cluster provisioned with many 50Gi volumes would not match a PVC requesting 100Gi.  The PVC can be bound when a 100Gi PV is added to the cluster.

### Using

Pods use their claim as a volume.  The cluster uses the claim to find the bound volume bound and mounts that volume for the user.  For those volumes that support multiple access modes, the user specifies which mode desired when using their claim as a volume in a pod.

### Releasing

When a user is done with their volume, they can delete their claim which allows reclamation of the resource.  The volume is considered "released" when the claim is deleted, but it is not yet available for another claim.  The previous claimant's data remains on the volume which must be handled according to policy.

### Reclaiming

A persistent volume's reclaim policy tells the cluster what to do with the volume after it's released.  Currently, volumes can either be Retained or Recycled.  Retention allows for manual reclamation of the resource.  For those volume plugins that support it, recycling performs a basic scrub ("rm -rf /thevolume/*") on the volume and makes it available again for a new claim.

## Types of Persistent Volumes

Persistent volumes are implemented as plugins.  Kubernetes currently supports the following plugins:

* GCEPersistentDisk
* AWSElasticBlockStore
* NFS
* ISCSI
* RBD
* Glusterfs
* HostPath (single node testing only)


## Persistent Volumes

Each PV contains a spec and status, which is the specification and status of the volume.  

```

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
  status:
      phase: Bound
	
```

### Capacity

Generally, a PV will have a specific storage capacity.  This is set using the PV's ```capacity``` attribute.  See the Kubernetes [Resource Model](./resources.md) to understand the units expected by ```capacity```.

Currently, storage size is the only resource that can be set or requested.  Future attributes may include IOPS, throughput, etc.

### Access Modes

Persistent Volumes can be mounted on a host in any way supported by the resource provider.  Providers will have different capabilities and each PV's access modes are set to the specific modes supported by that particular volume.  For example, NFS can support multiple read/write clients, but a specific NFS PV might be exported on the server as read-only.  Each PV gets its own set of access modes describing that specific PV's capabilities.

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

* Available -- a free resource resource that is not yet bound to a claim
* Bound -- the volume is bound to a claim
* Released -- the claim has been deleted, but the resource is not yet reclaimed by the cluster
* Failed -- the volume has failed its automatic reclamation

The CLI will show the name of the PVC bound to the PV.

## Persistent Volume Claims

Each PVC contains a spec and status, which is the specification and status of the claim.

```

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

Claims, like pods, can request specific quantities of a resource.  In this case, the request is for storage.  The same [resource model](./resources.md) applies to both volumes and claims.

## Claims As Volumes

Pods access storage by using the claim as a volume.  Claims must exist in the same namespace as the pod using the claim.  The cluster finds the claim in the pod's namespace and uses it to get the persistent volume backing the claim.  The volume is then mounted to the host and into the pod.

```

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

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/persistent-volumes.md?pixel)]()
