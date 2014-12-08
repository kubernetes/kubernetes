# Persistent Disks

## Abstract

This document defines persistent, cluster-scoped storage for applications that require long lived data.  A new entity
`PersistentDisk` is introduced to provide longevity to data that outlasts pods.  PersistentDisks with mounting history
provides an easy implementation of durable or node-specific data [[see Eric Tune's discussion of this topic]](https://github.com/GoogleCloudPlatform/kubernetes/pull/1515)
 
 Currently, all data volumes which will be exposed to pods must already exist and be formatted with the appropriate filesystem prior 
 to being attached and mounted.  Kubernetes makes no guarantees at runtime that these volumes exist or are available.  
 High availability is left to the storage provider. 
 

    *All Names are TBD - PersistentDisk is a placeholder name
 
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

> ## Optional
> 
> ### EmptyDisk
> 
> 1. Refactor EmptyDir -> EmptyDisk.  All behavior remains identical.
> 1. EmptyDisk does not have long lived identity.
> 1. Is an EmptyDir the same as a NodeLocalDisk with w/ boolean flag for persistence?  

## Security

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

## <a name=schema></a>Schema

* Add new PersistentDisk object
    * Two reasons to create new top-level object
        1. Disks are 1:N.  Normalizing the schema makes sense for this reuse.
        2. Disks have identity. Disks outlive pods. Their IDs must be persistent for future re-mounting.  
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

	// HostDir represents a pre-existing directory on the host machine that is directly
	// exposed to the container. This is generally used for system agents or other privileged
	// things that are allowed to see the host machine. Most containers will NOT need this.
	// TODO(jonesdl) We need to restrict who can use host directory mounts and who can/can not
	// mount host directories as read/write.
	HostDir *HostDir `json:"hostDir" yaml:"hostDir"`
	
	// EmptyDir represents a temporary directory that shares a pod's lifetime.
	EmptyDir *EmptyDir `json:"emptyDir" yaml:"emptyDir"`
	
	// GCEPersistentDisk represents a GCE Disk resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	GCEPersistentDisk *GCEPersistentDisk `json:"gcePersistentDisk" yaml:"gcePersistentDisk"`
	
    // AWSPersistentDisk represents an AWS EBS volume that is attached to a
    // kubelet's host machine and then exposed to the pod.
	AWSPersistentDisk  *AWSPersistentDisk `json:"awsPersistentDisk" yaml:"awsPersistentDisk"`
	
	// GitRepo represents a git repository at a particular revision.
	GitRepo *GitRepo `json:"gitRepo" yaml:"gitRepo"`
	
	// NFSPersistentDisk represents a pre-existing server on the network with ready exports
	NFSPersistentDisk  *NFSPersistentDisk `json:"nfsPersistentDisk" yaml:"nfsPersistentDisk"`
}


// AWSPersistentDisk represents an Elastic Block Store in Amazon Web Services.
//
// A AWS PD must exist and be formatted before mounting to a container.
// The disk must also be in the same AWS project and zone as the kubelet.
// A AWS PD can only be mounted once.
type AWSPersistentDisk struct {

	// Unique name of the PD resource. Used to identify the disk in GCE
	PDName string `yaml:"pdName" json:"pdName"`
	
	// Required: Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs"
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	FSType string `yaml:"fsType,omitempty" json:"fsType,omitempty"`
	
	// Optional: Partition on the disk to mount.
	// If omitted, kubelet will attempt to mount the device name.
	// Ex. For /dev/sda1, this field is "1", for /dev/sda, this field is 0 or empty.
	Partition int `yaml:"partition,omitempty" json:"partition,omitempty"`
	
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	ReadOnly bool `yaml:"readOnly,omitempty" json:"readOnly,omitempty"`
}

// HostDir represents bare host directory volume.
type NFSPersistentDisk struct {
	
	Path string `json:"path" yaml:"path"`
	
    // Required: Constants from sys/mount.h
    MountFlags  uint64
    
    // Options understood by the file system type provided by FSTYPE, see nfs(5) for details
    // nfs: hard,rsize=xxx,wsize=yyy,noac
    Options     string
    
    // From mount(2)
    // Required: Server hosting NFS server
    // NFS: host name or IP address
    Server      string
    
    // Required: File system to be attached
    // NFS: path
    Source      string
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

type VolumeSource struct {
	HostDir *HostDir
	EmptyDir *EmptyDir 
	
	// changed from specific GCEPersistentDisk.
	// selectors that can find a PersistentDisk to attach and mount
	Selector []map[string]
	
    // Optional: Defaults to false (read/write). ReadOnly here will force
    // the ReadOnly setting in VolumeMounts.
	ReadOnly bool
}

```



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

