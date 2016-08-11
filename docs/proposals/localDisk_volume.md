# localDisk volume

## Abstract

This is a proposal for implementing a new volume type, localDisk, which will support,

 1. the Node has multiple physical disks and pod requires 1 ~ N disks as its storage volumes.
 2. the Node has Linux LVM(Logical Volume Manager) partition and pods allocate specific size LV(Logical Volume) from this VG(Volume Group).
 3. When pods are deleted, the data in the volume could be either retained or deleted.


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

3. Create a PVC and stroage-class is required in PVC annotations, then create a pod by using this PVC.

```yaml
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: myclaim
  annotations:
    # using either ld-disk or ld-lvm as storage-class
    volume.alpha.kubernetes.io/storage-class: ld-disk
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi   
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
  - PersistentVolumeClaim:
        claimName: myclaim
    name: test-vol
  restartPolicy: Always
```

## Proposed Design

### localDisk volume plugin

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
	Node string `json: node,omitempty`
	Path string `json:"path,omitempty"`
	//The size of local disk, unit is MB
	VolumeSize int64 `json:"volumesize, omitempty"`
	// The type of local disk, it could be "disk", "lvm" now.
	Type LocalDiskType `json:"type"`
	// The filesystem type of local disk, the default is ext4
	FSType string `json:"fsType,omitempty"`
}
```

### New PersistentVolumeSource in PersistentVolume

New PersistentVolumeSource will be added for localDisk, so localDisk volume lifecycle could be independent from pod, and localDisk volume could be managed by PV/PVC too.

```go
type PersistentVolumeSource struct {
    ... 
    //localDisk Volume Source
   	LocalDisk *LocalDiskVolumeSource `json:"localDisk,omitempty"`
    
    ... 
}
```

#### Disk
If the localDisk type is `disk`

* The localDisk PVs will be created by kubelet when localDisk volume plugin is initialized.
* PVC will be created either by kubelet or by user, e.g. using kubectl,
    * When Pod with a localDisk volume is created, localDisk volume plugin in kubelet will create a PVC and volume controller bind this PVC to an allocated PV. This PVC will be deleted when Pod with a localDisk volume is deleted.
    * User can create a PVC, and then create a pod by using PVC as a volume. If user delete the pod, the PVC will not be deleted and the data in the volume retained. The data in volume will be removed when PVC is deleted by user.
* `volume.alpha.kubernetes.io/storage-class: ld-disk` must be in the annotations of PVC, and will be changed to `PersistentVolumeClaim.Spec.Class` if this Class attribute is implemented later.
* When user creates a pod by using this type PVC, kube-scheduler will schedule pod to the node where the bound PV is. 
* To support multi-disk in a pod, put disk number in annotation `volume.alpha.kubernetes.io/replica`.

```yaml
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: myclaim
  annotations:
    volume.alpha.kubernetes.io/storage-class: ld-disk
    volume.alpha.kubernetes.io/replica: 4
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

#### LVM
If the localDisk type is `lvm`

* A new annotations member `volume.alpha.kubernetes.io/vgallocatable` is added in `Node` object.
* Kubelet on each Node will update VGs' available size in this`Node` annotations `volume.alpha.kubernetes.io/vgallocatable`
* Admin will not need to create PV of lvm, user can create PVC directly and volume dynamic provision controller will create PV for the PVC.
* When user creates a PVC, `volume.alpha.kubernetes.io/storage-class: ld-lvm` must be in the annotations. The volume provision controller will call plugin's provision function to create a PV of lvm and bind it to PVC.
* When user creates a pod by using this type PVC, kube-scheduler will schedule pod to the node where the bound PV is.


```yaml
apiVersion: v1
kind: Node
metadata:
  annotations:    
    volume.alpha.kubernetes.io/vgallocatable: [{"/dev/vg1": "15G" , "/dev/vg2": "20G"}]
    ... 
```


### Kubelet enhancement

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
	    * if the type is `disk`, find an unused PV, allocate a PVC, wait until PVC and PV are bound, then put the corret path for plugin.
	    * if the type is `lvm`, allocate required size logical volume, mount it to specified path, e.g. /var/vol-pool/lvm/<vg-name>/<lv-name>. put the correct path for plugin.

    * When pod is deleted, kubelet will check the `LocalDiskType`,
	    * if the type is `disk`, remove all files under disks, and delete the PVC.
	    * if the type is `lvm`, umount the logical volume and delete this logical volume, update available size of VG in `NodeStatus`.

* If the `VolumeSource` is PVC,
    * When pod is scheduled to Node, get path for the bound PV for plugin.
    * When pod is deleted, do nothing.
`

### Kube-scheduler enhancement

New predicates function will be required for PV of localDisk resource calculation during pod scheduling. When the function is called, it will check items in `persistentvolumes`

* if request's localDisk volume type is `disk`, it will check available disks in specified Node.
* if request's localDisk volume type is `lvm`, it will check available VG size in specified Node.
* if pod's volume is PVC, get the Node from its PV, and bind pod to Node.
