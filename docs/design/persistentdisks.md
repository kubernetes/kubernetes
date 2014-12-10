# Persistent Disks

## Abstract

This document defines persistent, cluster-scoped storage for applications that require long lived data.  Developers will request storage
 as an intention ("I'd like 3gb of storage, please") and Kubernetes will use the configured cloud provider implementation to create, manage, and secure
 that storage.  Names for storage will vary by cloud provider , but each shall be known internally to Kubernetes as a disk.  A new entity `PersistentDisk` is introduced
 to provide longevity to data that outlasts pods.
   
PersistentDisks with mounting history provides a means for durable or node-specific data [[see Eric Tune's discussion of this topic]](https://github.com/GoogleCloudPlatform/kubernetes/pull/1515).

Kubernetes makes no guarantees at runtime that these volumes exist or are available. High availability is left to the storage provider.
 
Disks are secured by storing the username of the authorized API user who created the disk.  Any pod (or other domain object) requesting
access to the disk must occur as the same user who created the disk.  Legacy disks are supported by administrative action. An admin can
create the necessary objects in etcd with a named disk and username.
 

## Size and Performance -- Requesting a disk 

A storage request is made using the API.  Only the intent is expressed (size and performance).  Kubernetes handles creating, managing, attaching/mounting, and deleting the disk.

Different cloud providers have different types of disks that vary by performance and price.  Types of disks in cloud providers will be normalized internally to Kubernetes and the appropriate value
used when accessing the provider.

`PerformanceType: fast, faster, fastest`    or   `PerformanceType: Low, Normal, High`.     

AWS EBS mapping to internal types:

    fastest:    'io1' for Provisioned IOPS (SSD) volumes 
    faster:      'gp2' for General Purpose (SSD) volumes
    fast:       'standard' for Magnetic volumes  (default)

GCE disk mapping to internal types:

    fastest:    'pd-ssd' for solid-state drives
    faster:     'pd-standard' for hard disk (defaut)
    fast:       'pd-standard' -- there is no 3rd option for GCE
    
NFS disk mapping to internal types:

    fastest:    'nodeLocal' for node local storage
    faster:     'nfsShare' an NFS share on the network
    fast:       'nfsShare' -- there is no 3rd option for NFS
    
## DiskController

Add a new daemon to master that watches for changes to PersistentDisk in etcd.

New disk requests are posted to the API server and are stored in etcd with "pending" status.  The DiskController sees the
change and creates the disk using the cloud provider under which Kubernetes is running (e.g, if in AWS, EBS Volumes are created, etc).
Each provider knows how to make disks for its infrastructure.
  
After creation, the disk goes from "pending" to "created" state.

## Formatting Options

An attached disk may require formatting before being mounted as a filesystem on the host.
  
`GCEPersistentDisk` and `AWSPersistentDisk`, for example, are created as unformatted block storage.

`NFSPersistentDisk`, on the other hand, would come from a pool of created, formatted, and exported shares to mount.
 
`NodePersistentDisk` would either require creating a partion and formatting it with the filesystem of choice or only allow the filesystem that is currently on the host.
 
> XFS is a filesystem that supports project quotas with regards to size and disk usage.  Without a quota in place, any pod can fill an EmptyDir or NodePersistentDisk.

Formatting must happen on the host.  Kubelet attaches a disk and mounts it.  If the disk is not formatted, Kubelet could perform this task in between attaching and mounting.  

DiskController does not seem like the actor to format disks, unless the disk is attached to an arbitrary host for formatting and then detached before a pod is placed on the host
by the Scheduler.   


## Security and Disk Ownership

It is important to secure disks only to those pods authorized to access them.  Disks and pods must belong to the same namespace.  Securing namespaces to users is required to keep pods from accessing disks
that don't belong to them.  Security is implemented via an access control list whereby an administrator specifies which usernames can access which namespaces.  

## <a name=schema></a>Schema

* Add new top-level PersistentDisk object
    * Reasons:
        1. Disks are 1:N.  Normalizing the schema makes sense for this reuse.
        2. Disks have identity. Disks outlive pods. Their IDs must be persistent for future re-mounting.
        3. Disk identity is paired with username for security.
        4. API required to delete old disks as a separate action from deleting pods (not all pod deletes should delete the disk)
* Separate Spec and Status for disks
    * DiskSpec contains one of the specific disk types (GCE, AWS, NFS, NodeLocal)
    * Status keeps mount history for X days, allows pods to preferentially schedule onto previously used hosts


```go


struct PersistentDisk {

    TypeMeta
    ObjectMeta      //namespace will be required.  access to disks only allowed in same namespace for security
	
	// separating *PersistentDisk into Spec and Status like other API objects
	Spec    DiskSpec
	Status  DiskStatus
}

struct DiskSpec {

    // size of the disk, in gb
    Size        int
    
    // references various performance types in a cloud provider 
    Type        DiskPerformanceType    
}


struct DiskStatus {

    // PodCondition recently became PodPhase - see https://github.com/GoogleCloudPlatform/kubernetes/pull/2522
    Condition   DiskCondition
    
    // a disk can be mounted on many hosts, depending on type
    Mounts []Mount
}

struct Mount struct {
    Host            string
    HostIP          string
    MountedDate     string  //RFC 3339 format (e.g, 1985-04-12T23:20:50.52Z)
    MountCondition  DiskCondition
}

type DiskCondition string

const (
    MountPending    DiskCondition = "Pending"
    Attached        DiskCondition = "Attached"
    Mounted         DiskCondition = "Mounted"
    MountFailed     DiskCondition = "Failed"
)

type DiskPerformanceType string

const (
    Fast            DiskPerformanceType  = "fast"
    Faster          DiskPerformanceType  = "faster"
    Fastest         DiskPerformanceType  = "fastest"
)

type VolumeSource struct {
	HostDir *HostDir
	EmptyDir *EmptyDir 
	
	// changed from specific GCEPersistentDisk.
	// selectors that can find a PersistentDisk to attach and mount.
	// could substitute a named disk for selectors.
	Selector []map[string]
	
	GitRepo *GitRepo `json:"gitRepo"`
	
    // Optional: Defaults to false (read/write). 
    // the ReadOnly setting in VolumeMounts.
    // This allows many GCE VMs to attach a single PersistentDisk many times in read-only mode
	ReadOnly bool
}

```



> ## everything below will be updated once the concepts above are finalized.  Tech analysis and design to follow requirements for disks.

## Phased Approach

Every effort will be made to break this task into discrete pieces of functionality which can be committed as individual pulls/merges.

![alt text](http://media-cache-ec0.pinimg.com/236x/da/a1/7e/daa17e92ba3a1b04e203135043db580b.jpg "How do you eat an elephant?")

### Phase One

See [Tim Hockin's new volumes framework](https://github.com/GoogleCloudPlatform/kubernetes/pull/2598)

* Use existing volumes (and working with Tim's code above) framework  
* Implement AWSPersistentDisk
* Implement NFSPersistentDisk
* Have both new disk types reach parity with GCEPersistentDisk

### Phase Two

* Implement PersistentDisk framework
* Implement NodeLocalDisk

### Phase Three

* Implement creation and management of disks

 
## Disk implementations

### All *PersistentDisks

1. All *PersistentDisks rules are enforced by the scheduler and predicates.
1. Disks have identity that can outlive a pod.
1. Disks must share the same namespace as a pod.
1. Pods can have many volumes of many types except for cloud provider disks (GCE, AWS).
1. Pods can have only a single cloud provider disk (GCE, AWS).

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
 
### NFSPersistentDisk

1. NFSPersistentDisks can be mounted many times as read/write.
1. NFSPD will not support file locking.  This responsibility remains with the application developer. 
1. High availability of data is left to the cluster administrator.

### NodeLocalDisk

1. Local node directory similar to EmptyDir but is not deleted upon pod exit.
1. NodePD has long lived identity but local storage cannot be relied upon (e.g, disk failure, host crash)
1. The scheduler will attempt to place a pod on the host where the NodePD lives.
1. High availability of data is left to the cluster administrator.




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

* add AWSPersistentDisk, NFSPersistentDisk, NodeLocalDisk structs
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


### TODO - describe new PersistentDisk object, storage,... 

PersistentDisk.Status.Mounts must be updated when a Volume is requested or released by a pod. 

