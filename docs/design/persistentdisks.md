# Persistent Disks

## Abstract

This document defines persistent, cluster-scoped storage for applications that require long lived data.  Developers will request storage as an intention ("I'd like 3gb of storage, please, with 3k min IOPS") and Kubernetes will use the configured cloud provider implementation to create, manage, and secure that storage.  Names for storage will vary by cloud provider, but each shall be known internally to Kubernetes as a disk. A new entity `PersistentDisk` is introduced to provide longevity to data that outlasts pods.
   
PersistentDisks with mounting history provides a means for durable or node-specific data as a `NodePersistentDisk` [(see Eric Tune's discussion of this topic - #1515)](https://github.com/GoogleCloudPlatform/kubernetes/pull/1515).

Kubernetes makes no guarantees at runtime that these volumes exist or are available. High availability is left to the storage provider.
 
Disks are secured by restricting disks to the same namespace as the pods that request them.  [ACLs](../authorization.md) can restrict users to specific namespaces to prevent inadvertent access to a disk. Legacy disks are supported by administrative action.
 
## Goals

* Divorce pod authors from ops
	* Provide a means for pod authors to request disks and storage without requiring actions from an ops team.
* Implement dynamic storage for cloud providers (using GCE/AWS APIs to create disks/volumes)
* Provide working examples for each type of storage and cloud provider.

## Future

A means of mounting NFS shares to a host is a future requirement.  It is important to ensure this proposal does nothing to limit the ability of mounting NFS in future iterations.

## Constraints and Assumptions

* NodePersistentDisk can crash a host
    * Without quotas in place, any pod can fill the filesystem via an EmptyDir, HostDir, or NodePersistentDisk.
    * A filesystem like XFS provides project quotas.
 * All disks are considered "new" in Kubernetes
 	* Migrating legacy data is not in scope
 	* Existing named disks are supported by administrative action (adding disk objects in etcd)


## Disk performance attributes -- Requesting a disk 

Pod authors can request storage by its characteristics, specifically size, performance, and filesystem.  Kubernetes maps the request to the underlying cloud provider and chooses the best match.

> Example: Alice requests 1gb of space for a WordPress installation expected to have low traffic.  She requests "1gb, ext4, 100 IOPS".   Kubernetes is running in AWS and matches this request to an EBS Magnetic Volume (max oops 40-200).  K8s uses the AWS API to create the disk, attach it to the host, format the filesystem, mount the volume, and then deploys Alice's pod.

> Example:  Ralph requests 10gb of high performance disk space for a MySQL database.  He requests "10gb, xfs, 10000 IOPS".  Cabernets is running in GCE and matches this request to a 
SSD Persistent Disk (read/write 10k/15k IOPS, respectively).  K8s uses the GCE API to create Ralph's disk, just as it did in Alice's example above.

    
## DiskController

Add a new daemon to master that watches for changes to PersistentDisk in etcd.

New disk requests are posted to the API server and are stored in etcd with "pending" status.  The DiskController sees the
change and creates the disk using the cloud provider under which Kubernetes is running (i.e, AWS and GCE). Each provider knows how to make disks for its infrastructure.
  
After creation, the disk phase goes from "pending" to "created".

## Formatting Options

Disks created in GCE and AWS require formatting before being mounted as a filesystem on the host.
 
`NodePersistentDisk` would either require creating a partion and formatting it with the filesystem of choice or only allow the filesystem that is currently on the host.
 
> XFS is a filesystem that supports project quotas with regards to size and disk usage.  Without a quota in place, any pod can fill the entire filesystem via an EmptyDir or NodePersistentDisk.

Formatting must happen on the host.  Kubelet attaches a disk and mounts it.  If the disk is not formatted, Kubelet will perform this task in between attaching and mounting.  

## Security and Disk Ownership

It is important to secure disks only to those pods authorized to access them.  Disks and pods must belong to the same namespace.  Securing namespaces to users is required to keep pods from accessing disks
that don't belong to them.  Security is implemented via an access control list whereby an administrator specifies which usernames can access which namespaces.  

## Behavior

* PersistentDisk.Status.Mounts must be updated as disks are attached and detached from hosts.
* NodePersistentDisks are never detached and remain in PersistentDisk.Status.Mounts until the disk is deleted.

## Volumes Framework

See [Tim Hockin's volumes framework](https://github.com/GoogleCloudPlatform/kubernetes/pull/2598)

## <a name=schema></a>Schema

* Add new top-level PersistentDisk REST object
    * Disks have identity. Disks outlive pods. Their IDs must be persistent for future re-mounting.
    * Disks requests are not expected to be POSTed to the API server.  
    * Top level REST object is required to query for lists of disks, see their condition, and for deletion (as necessary)
* Separate Spec and Status for disks
    * Status keeps mount history for X days, allows pods to preferentially schedule onto previously used hosts

> **See types.go for structs**


 
## Disk implementations

### All *PersistentDisks

1. All *PersistentDisks rules are enforced by the scheduler and predicates.
1. Disks have identity that can outlive a pod.
1. Disks must share the same namespace as a pod.
1. Pods can have many volumes of many types except for cloud provider disks (GCE, AWS).
1. Pods can have only a single type of cloud provider disk (GCE, AWS, not both).

### GCEPersistentDisk

1. A GCEPersistentDisk must already be created and formatted before it can be used.
1. GCEPD must exist in the same availability zone as the VM.
1. GCEPDs can be attached once as read/write or many times as read-only, not both.
1. Hard limit 16 disk attachments per VM.
1. High-availability subject to the GCE Service Level Agreement.

### AWSPersistentDisk 

1. An AWS EBS volume must already be created and formatted before it can be used.
1. Must exist in the same availability zone as the VM.
1. AWSPDs can be attached once as read/write.
1. AWS Encrypted disks can only be attached to VMs that support EBS encryption
1. Attachments limited by Amazon account, not ES2 instance limit.
1. High-availability subject to the AWS Service Level Agreement.

### NodePersistentDisk

1. Local node directory similar to EmptyDir but is not deleted upon pod exit.
1. NodePD has long lived identity but local storage cannot be relied upon (e.g, disk failure, host crash)
1. The scheduler will attempt to place a pod on the host where the NodePD lives.
1. High availability of data is left to the cluster administrator.  It is not guaranteed the host (and disk) is available.




## Kubelet enhancements

***See [https://github.com/GoogleCloudPlatform/kubernetes/pull/2598](https://github.com/GoogleCloudPlatform/kubernetes/pull/2598) for a plugin architecture idea by @thockin***     
Much of the kubelet's attach/mount functionality will be handled by the plugin for that specific provider (GCE, AWS, etc)     


## Code changes

_**likely changing due to #2598 above**_

### pkg/kublet

#### kubelet.go

* line 341 - mountExternalVolumes(pod *api.BoundPod) -- uses interface Builder.SetUp()
  * Refactor from pod.Spec.Volumes to pod.Spec.Volumes.Selector to find the volumes to attach/mount 
* line 820 - reconcileVolumes(pods []api.BoundPod) -- uses interface Cleaner.TearDown()

#### server.go

* add REST handlers

### pkg/volume/volume.go

* interfaces and factories 
* CreateVolumeBuilder & CreateVolumeCleaner
  * genericize AttachDisk and DetachDisk interface (currently is gcePersistentDiskUtil)
  * add AWS, NFS, NodeLocal implementations

### pkg/api/

#### types.go

* add AWSPersistentDisk, NFSPersistentDisk, NodePersistentDisk structs
* add new types to VolumeSource struct
* add new PersistentDisk and related structs
* repeat in pkg/api/v1beta3/types.go

#### validation/validation.go

* add ValidateDisk



### pkg/scheduler/predicates.go

* add FitPredicates for all PersistentDisk rules
* add FitPredicate for matching NodeLocalDisk to existing host, if available.
      * if host exists, return true for host and false for others
      * if host !exist, return true so any host can match
      * this disk impl won't work while in pkg/volume (all orphans removed) but will with PersistentDisk stateful storage of mount history.

### pkg/cloudprovider/plugins.go

* Continue to use API credentials for cloud providers read via io.Reader, as implemented for GCE.
* cloudprovider/aws & gce both seem to pass on authorization.  Do creds currently come from cmd line tools and local keys?
 
### pkg/registry

#### add /disk
 
* add rest.go - implement RESTStorage
* add registry.go

#### etcd/etcd.go

* add Disk registry methods

### pkg/client

* add volumes.go and client volume methods