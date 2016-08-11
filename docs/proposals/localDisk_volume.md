# localDisk volume

## Abstract

This is a proposal for implementing a new volume type, localDisk, which will support,

 1. the Node has multiple physical disks and pod requires 1 ~ N disks as its storage volumes.
 2. the Node has lvm VG and pods allocate specific size LVs from this VG.
 3. When pods are deleted, the data in the volume could be either retained or deleted.
 
As localDisk volume life cycle would be different from pod, we need volume APIs to do the management. 
The existing PersistentVolume/PersistentVolumeClaim(PV/PVC) API is for network storage in cluster, we propose new LocalVolume/LocalVolumeClaim(LV/LVC) API to manage localDisk volume.                      

## Motivation

* This new volume type is for application that requires high I/O performance, one pod need a dedicated disk.
* The bigdata application, e.g. Kafka, Cassandra, Hadoop can deal with a bunch of distinct volumes, one pod requires multiple disks.  
* The emptyDir and hostPath volume doesn't have disk size limitation, the localDisk volume is able to provide this feature by using lvm.

## Constraints and Assumptions

1. The physical disks or logial volumes will be mounted to pre-defined mount-points in Nodes as below path format, and containers will use this host directory as data volume.
  * disks, `/var/vol-pool/disks/<uuid>`
  * logical volumes, `/var/vol-pool/lvm/<vg-name>/<lv-name>`

2. If user requests a volume with type of `disk`, the whole disk will be assigned to container.


## Use Cases

1. The pod need N local disks, e.g. 2 disks.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ld-test
spec:
  containers:
  - image: busybox
    command:
      - sleep
      - "3600"
    imagePullPolicy: IfNotPresent
    name: ld-test
    volumeMounts:
    - mountPath: /mnt/disk1
      name: test-vol1
    - mountPath: /mnt/disk2
      name: test-vol2
  volumes:
  - localDisk:
        type: disk
    name: test-vol1
  - localDisk:
        type: disk
    name: test-vol2
  restartPolicy: Always
```

2. The pod need a N GigaBytes lvm volume, e.g. 500 GB

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ld-test
spec:
  containers:
  - image: busybox
    command:
      - sleep
      - "3600"
    imagePullPolicy: IfNotPresent
    name: ld-test
    volumeMounts:
    - mountPath: /mnt/vol1
      name: test-vol
  volumes:
  - localDisk:
        volumesize: 500000
        type: lvm
    name: test-vol
  restartPolicy: Always
```

3. Create a LVC, then create a pod by using this LVC

```yaml
kind: LocalVolumeClaim
apiVersion: extensions/v1beta1
metadata:
  name: myclaim
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  volumetype: disk    
```

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ld-test
spec:
  containers:
  - image: busybox
    command:
      - sleep
      - "3600"
    imagePullPolicy: IfNotPresent
    name: ld-test
    volumeMounts:
    - mountPath: /mnt/vol
      name: test-vol
  volumes:
  - localVolumeClaim:
        claimName: myclaim
    name: test-vol
  restartPolicy: Always
```

## Proposed Design

### localDisk volume

New volume type localDisk will be created and registered as volume plugin in kubelet, `pkg/volume/local_disk/local_disk.go`

```go
type localDiskPlugin struct {
	host volume.VolumeHost
	// decouple creating Recyclers/Deleters/Provisioners by deferring to a function.  Allows for easier testing.
	newRecyclerFunc    func(pvName string, spec *volume.Spec, host volume.VolumeHost, volumeConfig volume.VolumeConfig) (volume.Recycler, error)
	newDeleterFunc     func(spec *volume.Spec, host volume.VolumeHost) (volume.Deleter, error)
	newProvisionerFunc func(options volume.VolumeOptions, host volume.VolumeHost) (volume.Provisioner, error)
	config             volume.VolumeConfig
}

const (
	localDiskPluginName = "kubernetes.io/local-disk"
)

```

In PodSpec, we shall add a new type of VolumeSource, `LocalDiskVolumeSource`

```go
type LocalDiskVolumeSource struct {
	// Node and Path here are for localDisk volume recycle pod only
	Node string `json: node, omitempty`
	Path string `json:"path, omitempty"`
	//The size of local disk, unit is MB
	VolumeSize int64 `json:"volumeszie, omitempty"`
	// The type of local disk, it could be "disk", "lvm" now.
	Type LocalDiskType `json:"type"`
	// The filesystem type of local disk, the default is ext4
	FSType string `json:"fsType,omitempty"`
}
```

### LocalVolume and LocalVolumeClaim (LV/LVC)

The LocalVolume and LocalVolumeClaim (LV/LVC) are new volume management objects in this proposal.

1. It's for localDisk volume only now.
2. LVs will be created by kubelet, when localDisk volume is registered in kubelet, admin can't create LVs by kube clinet api. And LVs will be removed when their Node is deleted.
3. LVCs will be created either by kubelet or by user, e.g. using kubectl,
    * When Pod with a localDisk volume is created, localDisk volume plugin in kubelet will create a LVC and bind this LVC to an allocated LV. This LVC will be deleted when Pod with a localDisk volume is deleted.
    * User can create a LVC, and then create a pod by using LVC as a volume. If user delete the pod, the LVC will not be deleted and the data in the volume retained. The data in volume will be removed when LVC is deleted by user.
4. A localvolume controller will be added in the kube-controller-manager, this controller shall watch LVCs and bind new added LVCs to available LV. 
5. kubectl will support all LV/LVC operations. 

#### Flow of LV/LVC lifecycle
* Kubelet creates LVs by exporting configured disks on the the Node when kubelet starts
* User creates LVC
* Localvolume controller binds LVC to one available LV
* User creates a pod by using LVC
* User deletes the pod.
* The LVC is there still, and the data on the LV is there too.
* User creates a new pod by using the same LVC.
* User deletes the pod
* User deletes the lVC
* Localvolume controller detects LVC's deletion, unbinds it from LV and removes all data on the LV by recycle pod.
* When Node is deleted.
* Localvolume controller deletes all LVs on that Node.


```go
type LocalVolume struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta           `json:"metadata,omitempty"`
	//Spec defines a local volume owned by the cluster
	Spec LocalVolumeSpec `json:"spec,omitempty"`
	// Status represents the current information about local volume.
	Status LocalVolumeStatus `json:"status,omitempty"`
}

type LocalVolumeSpec struct {
	// Resources represents the actual resources of the volume
	LocalVolumeSource  `json:",inline"`
	// Reference to LV claim
	ClaimRef []*api.ObjectReference `json:"claimRef,omitempty"`
	// AccessModes contains all ways the volume can be mounted
	AccessModes []LocalVolumeAccessMode `json:"accessModes,omitempty"`
	// NodeName represent node where the LV is
	NodeName string `json:"nodeName,omitempty"`
}

type LocalVolumeSource struct {
	LocalDisk *api.LocalDiskVolumeSource `json:"localDisk,omitempty"`
}

type LocalVolumeStatus struct {
	AvailSize int64 `json:"availsize"`
}

type LocalVolumeList struct {
	unversioned.TypeMeta `json:",inline"`
	unversioned.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items []LocalVolume `json:"items" protobuf:"bytes,2,rep,name=items"`
}

```

```go
type LocalVolumeClaim struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta `json:"metadata,omitempty"`
	// Spec defines the volume requested by a pod author
	Spec LocalVolumeClaimSpec `json:"spec,omitempty"`
	// Status represents the current information about a claim
	Status LocalVolumeClaimStatus `json:"status,omitempty"`
}


type LocalVolumeClaimSpec struct {
	// Contains the types of access modes required
	AccessModes []LocalVolumeAccessMode `json:"accessModes,omitempty"`
	// Resources represents the minimum resources required
	Resources api.ResourceRequirements `json:"resources,omitempty"`
	// VolumeType here to inidicate disks or lvm
	VolumeType string `json:"volumetype.omitempty"`
	// VolumeName is the binding reference to the PersistentVolume backing this
	// claim. When set to non-empty value Selector is not evaluated
	VolumeName string `json:"volumeName,omitempty"`
	// If Recycle is true, the data on the localDisk will be removed when pod is deleted.
	// Else the data will be kept after pod is deleted.
	Recycle bool `json:"recycle,omitempty"`
}

type LocalVolumeClaimStatus struct {
	// Phase represents the current phase of LocalVolumeClaim
	Phase LocalVolumeClaimPhase `json:"phase,omitempty"`
	// AccessModes contains all ways the volume backing the PVC can be mounted
	AccessModes []LocalVolumeAccessMode `json:"accessModes,omitempty"`
	// Represents the actual resources of the underlying volume
	Capacity api.ResourceList `json:"capacity,omitempty"`
}

type LocalVolumeClaimList struct {
	unversioned.TypeMeta `json:",inline"`
	unversioned.ListMeta `json:"metadata,omitempty"`
	Items                []LocalVolumeClaim `json:"items"`
}

```

### kubelet enhancement

kubelet will have below changes to support localDisk volume,

* Read config file which has local disks configuration information, format the disks and mount the disks to specified path, e.g. /var/vol-pool/disks/<uuid>, also create LV instances during the kubelet volume plugin initialization.

```
[disks]
/dev/vdc
/dev/vdd

[lvm]
/dev/ldvg

```

* If the `VolumeSource` is localDisk
    * When pod is scheduled to the Node, kubelet will check the `LocalDiskType`,
	    * if the type is `disk`, find an unused LV and get the corret path for plugin.
	    * if the type is `lvm`, allocate required size logical volume, mount it to specified path, e.g. /var/vol-pool/lvm/<vg-name>/<lv-name>. Get the correct path for plugin and update available size of VG in its LV .
	    * No matter what type it is, a LVC instance is allocated and bind the LVC to corresponding LV.

    * When pod is deleted, kubelet will check the `LocalDiskType`,
	    * if the type is `disk`, remove all files under disks.
	    * if the type is `lvm`, umount the logical volume and delete this logical volume, update available size of VG in its LV.
	    * No matter what type it is, unbind the LVC and LV and remove its LVC instance.
	   
* If the `VolumeSource` is LVC, 
    * When pod is scheduled to Node, get path for the bound LV for plugin.
    * When pod is deleted, do nothing.  
`

### kube-scheduler enhancement

New predicates function will be required for LV resource calculation during pod scheduling, and one cache is created for `localvolumes`.  When the function is called for every Node, it will check items in `localvolumes`

* if request's localDisk volume type is `disk`, it will check available disks in specified Node.
* if request's localDisk volume type is `lvm`, it will check available VG size in specified Node.
* if pod's volume is LVC, get the Node from its LV, and bind pod to Node. 
