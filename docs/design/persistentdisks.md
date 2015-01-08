# Persistent Storage


## Abstract

This document defines persistent, cluster-scoped storage for applications requiring long lived data.  The term "PersistentStorage" was used to avoid overloading the term "Volume."  Storage is exposed to a pod as a volume.  Developers and pod authors request storage by defining their performance needs and are matched with an appropriate object from a pool maintained by administrators.

Kubernetes makes no guarantees at runtime that these volumes exist or are available. High availability is left to the storage provider.


## Goals

* Allow administrators to create and manage pools of persistent storage
* Allow developers and pod authors to request storage from the pool
* Implement dynamic storage for cloud providers (using GCE/AWS APIs to create disks/volumes)
* Provide working examples for each type of storage and cloud provider.
* Ensure developers and pod authors can rely on storage being available, without being closely bound to a particular disk, server, network topology, or storage technology.
* Implement  [ #1515 - Eric Tune's durable data](https://github.com/GoogleCloudPlatform/kubernetes/pull/1515) - by maintaining LastMount on PersistentStorage.Status for the pod scheduler to inspect.
* Use or don't impede [ #2598 - Tim Hockin's volumes framework](https://github.com/GoogleCloudPlatform/kubernetes/pull/2598)

## New Types

_**Detailed Structs in types.go coming next **_

`PersistentStorage` -- new long lived entity that represents a storage request by pod authors.  Contains Spec and Status.

`PersistentStorageSpec` -- a description of the storage being requested

`PersistentStorageStatus` -- represents information about the status of a storage device

`BoundStorage` -- a binding between PooledStorage and a PersistentStorage request.

`PooledStorage` -- an available storage device created by an administrator.  Contains Spec and Status.

`PooledStorageSpec` -- a description of the storage in this storage device in terms of the underlying provider

`PooledStorageStatus` -- status and recent mount history of the storage device 

`PooledStorageSource` -- the specific underlying provider.  Patterned after Volume->VolumeSource and the many types of volumes available.   GCEPersistentStorage, AWSPersistentStorage, NFSPersistentStorage, etc.
    
## New Controllers

New controller processes running on master will facilitate the creation and attaching of disks/volumes to a host.

### `StorageBindingController`

Watches etcd for new volume requests from pod authors.  Volumes are POSTed to the API server.  The VolumeBindingController satisfies requests with Volumes from the pool.  Volumes are "removed" from the pool by marking them used and creating a VolumeBinding with the PooledVolume.ID and Volume.ID of the request.

### `CloudStorageAttachingController`

As pods are scheduled onto hosts, the VolumeAttachingController performs the task of attaching a block device (GCE PD or AWS EBS volume) to a host.  Kubelet mounts the volume for the pod, and the volume will already be attached by the time Kubelet performs this task.

## Security

Storage is secured by restricting it to the same namespace as the pod requesting access.  [ACLs](../authorization.md) can restrict users to specific namespaces to prevent inadvertent access to a disk. Legacy disks are supported by administrative action by creating the requisite data objects in etcd.

## Constraints and Assumptions

* File storage on hosts should be limited to prevent malicious or careless clients from using all available storage.
    * A filesystem like XFS provides project quotas.  Useful for an NFS example with storage limits.
 * All persistent volumes are considered "new"
 	* Migrating legacy data is not in scope
 	* Legacy volumes can be manually created by administrative action when manual solutions exist

## Concepts & High Level Flow


1. Fill a pool with available storage by creating PooledStorage instances in etcd
    1.  Description (size, speed) and ID (from AWS/GCE) required
    2.  Storage must already be available with a list of unique identifiers.
    3.  Tools will be made to help with this task.  Manual until then.
2. Request storage from the pool by posting PersistentStorage to the API
    1.  Size and speed required in request.  ID of existing volume is optional.  
    2.  Namespace required for security
    3.  PersistentStorage is referenced by a Volume in a Pod by name.
3. StorageBindingController
    1. Watches etcd for new PersistentStorage 
    2. Matches storage request to available storage in the pool, creating a StorageBinding
        1.  Create StorageBinding with PooledStorage.ID and PersistentStorage.ID
        2.  The presence of StorageBinding with that PooledStorage.ID means it is in use and unavailable for any future requests.
4. Scheduler places pod on node with constraints
    1. New predicates are required for volumes (e.g, only 16 GCEPDs attached to a host)
    2. Requires a way to validate without having the scheduler know the types of storage devices being used.
5. CloudStorageAttachingController watches pod changes and performs the task of attaching the GCEPD/AWSEBS to the host
    1.  Kubelet continues to mount but the volume is expected to be attached by this point
    2.  Future refactor -- remove attaching from GCEPersistentDisk, allow StorageAttachingController to attach the PD


# Detailed Use Cases and System Effects

The following use cases intend to show the order of operations for creating and using persistent volumes as well as the components of the system that are responsible for each part of the work flow.  Several use cases are provided to show how persistent volumes are requested, provisioned, secured, and potentially deleted.

**Manual steps are required by the system administrator to create persistent volumes for consumption.  Tools and scripts will be created to automate these tasks.  The first use case describes the actions taken by the admin.**



## Creating Volumes for use
 User Story | System Action
 -----|-----
Admin Ed creates an EC2 instance, creates, attaches, and formats 20 EBS volumes (list below).  All volumes are subsequently detached from the instance and the instance shutdown, but the volumes remain in AWS.  Admin Ed maintains a list of all EBS volume IDs.	 | System does nothing.  Automation scripts can help future Admin Ed handle the task of creating volumes and tracking their IDs

Admin Ed creates 20 pooledvolume.json files, each containing a description and unique ID of the EBS volume created earlier.  He posts each json file to the api server.  Each are validated and Admin Ed queries the cluster for a list of persistent volumes after he is done posting them.  All 20 are shown. 								 | The api server validates each POST and stores each volume as a new `PooledVolume` object in etcd.

Admin Ed takes a break because those previous two steps were tedious.  | Manual tasks like these are easy automation targets.

### Volumes created by Admin Ed

Quantity | Size (gb) | Type
---- | ---- | ----
 5 | 5 | General Purpose SSD (3000 max burst IOPS)
 5 | 10 | General Purpose SSD (3000 max burst IOPS)
 5 | 10 | Provisioned IOPS (4000 IOPS)
 5 | 25 | Provisioned IOPS (4000 IOPS)




### In all User Stories, assume the following:

> Admin Ed sets up Kubernetes with an ACL list for two teams using two namespaces:  "Team Kramden" and "Team Norton".

> Ralph and Alice are on "Team Kramden".  Trixie and Alice are on "Team Norton".  All three users are restricted to their projects' namespaces but retain all other permissions, including creating pods, volumes, services, etc.

>  Admin Ed, naturally, has all permissions across all namespaces.



### The GO PATH -- Everything Just Works

 User Story | System Action
 -----|-----
Trixie wants to create a WordPress blog and needs persistent storage.  She first creates persistentvolume.json (namespace='Team Norton', name='WP Blog', size='1') and POSTs it to the api server.  After receiving a successful response, she queries the cluster for persistent volumes and sees her volume request in phase 'Pending'. | After successful validation, the API server creates a `PersistentVolume` object in etcd with the phase  'Pending'

Trixie waits and queries the cluster again to see if her volume has been created.  She sees it in phase 'Created'.  There was much rejoicing. | `VolumeBindingController` is watching etcd for new volume requests and fulfills them by creating `VolumeBinding` objects. <br><br>  A 1gb volume is available in the pool.  A new `VolumeBinding` object is created using PooledVolume.ID and PersistentVolume.ID and stores it in etcd.  The PooledVolume status is changed to 'Unavailable'.  The PersistentVolume.Phase is changed to 'Created' 

Trixie creates pod.json (or replController.json) with a suitable WordPress image containing the entire LAMP stack.  The `VolumeSource` struct for her pod contains `PersistentVolumeRef` which references her PersistentVolume by name.  The `VolumeMount` object referencing the volume contains a path to the container where MySQL expects to find its data. | API server validates the persistent volume exists by name in etcd along with existing pod validation.  A `Pod` object is created and stored in etcd.

tbd | more details to come

### Security Violation Story

Add details showing how requesting storage across namespaces is enforced

 User Story | System Action
 -----|-----
tbd | tbd 


### Failed Validation Stories

Add details showing how storage requests can fail when a match is not found in the pool or is unavailable at attach/mount time.

 User Story | System Action
 -----|-----
tbd | tbd 


 
## Storage Implementations & Validation Rules

### All *PersistentStorage

1. All *PersistentStorage rules are enforced by the scheduler and predicates.
1. Storage has identity that can outlive a pod.
1. Storage must share the same namespace as a pod.
1. Pods can have many volumes of many types except for cloud provider disks (GCE, AWS).
1. Pods can have only a single type of cloud provider disk (GCE, AWS, not both).

### GCEPersistentStorage

1. A PersistentDisk in GCE must already be created and formatted before it can be used as GCEPersistentStorage.
1. GCEPD must exist in the same availability zone as the VM.
1. GCEPDs can be attached once as read/write or many times as read-only, not both.
1. Hard limit 16 disk attachments per VM.
1. High-availability subject to the GCE Service Level Agreement.

### AWSPersistentStorage 

1. An AWS EBS volume must already be created and formatted before it can be used as AWSPersistentStorage
1. Must exist in the same availability zone as the VM.
1. AWS EBS volumes can be attached once as read/write.
1. AWS Encrypted disks can only be attached to VMs that support EBS encryption
1. Attachments limited by Amazon account, not ES2 instance limit.
1. High-availability subject to the AWS Service Level Agreement.

### NodePersistentStorage

1. Local node directory similar to EmptyDir but is not deleted upon pod exit.
1. Node storage has long lived identity but local storage cannot be relied upon (e.g, disk failure, host crash)
1. The scheduler will attempt to place a pod on the host where the NPS lives.
1. High availability of data is left to the cluster administrator.  It is not guaranteed that the host (and disk) is available.




