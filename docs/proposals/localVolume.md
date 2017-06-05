# localVolume volume

## Abstract

This is a proposal for implementing a new volume type, localVolume, which will support,

 1. Exposing a set of local disks from each node as first class PVs that can be claimed by the end user and individual disks.
 2. Allow a group of such disks to be claimed on a single node.
 3. Exposing the storage capacity on a node (a collection of disks) and use a volume manager (lvm) to allocate volumes of specific size.
 4. The lifecycle of the volumes are independent of the lifecycle of the pods (unlike emptydir). A deletion of a pod does not cause the deletion of the disk automatically, it is handled as a Claim.

## Motivation

* This new volume type is for applications that requires high I/O performance, by isolating access to disk to a single pod (disk mode).
* Big Data applications, e.g. Kafka, Cassandra, Hadoop has inbuilt data replication and can deal with a bunch of distinct disks (JBOD) and we would like to leverage the native replication features while getting the benefits of the fast local SSDs.
* The emptyDir and hostPath volume doesn't have disk size limitation. Empty dir volume does not treat the disk as first class entity with life cycles separated from that of a pod using the data. HostPath is not a good alternative as it ties clients to a layout of a node which needs to be avoided.
* The FlexVolume allows user to develop his own driver to mount volume, e.g. lvm, but it doesn't support PV dynamic provisioning and no pod scheduling support for local volume.

## Constraints and Assumptions

1. The physical disks or logical volumes will be mounted to pre-defined mount-points in Nodes as the following example shows, and are bind mounted to containers by the local disk plugin.
  * disks, `/var/vol-pool/disks/<uuid>`
  * logical volumes, `/var/vol-pool/lvm/<vg-name>/<lv-name>`
2. If user requests a volume with type of `disk`, the whole disk will be assigned to container.
3. LVM VolumeGroup(VG) is statically created at the initialization time (first time kubelet sees the volumes).


## Use Cases

* The pod need N local disks, e.g. 2 disks.

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
  - localVolume:
        type: disk
    name: test-vol1
  - localVolume:
        type: disk
    name: test-vol2
  restartPolicy: Always
```

* The pod need a N GigaBytes lvm volume, e.g. 500 GB

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: mylvm01
spec:
  containers:
  - image: busybox
    command:
      - sleep
      - "3600"
    imagePullPolicy: IfNotPresent
    name: mylvm01
    volumeMounts:
    - mountPath: /mnt
      name: test-lvm
  volumes:
  - localVolume:
        type: "lvm"
        volumesize: 500Gi
        fsType: "ext4"
    name: test-lvm
  restartPolicy: Always
```

* Create a PVC and stroage-class is required in PVC annotations, then create a pod to use this PVC.

```yaml
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: myclaim
  annotations:
    volume.beta.kubernetes.io/storage-class: "local-ssd"
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

* Create two PVCs of local-ssd storage-class, then create a pod to use these two PVCs.
```yaml
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: mydisk2-1
  annotations:
    volume.beta.kubernetes.io/storage-class: "local-ssd"
    volume.beta.kubernetes.io/group: '{"name":"group1", "count":"2"}'
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

```yaml
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: mydisk2-2
  annotations:
    volume.beta.kubernetes.io/storage-class: "local-ssd"
    volume.beta.kubernetes.io/group: '{"name":"group1", "count":"2"}'
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
    name: mybb03
    volumeMounts:
    - mountPath: /mnt/disk1
      name: test-vol1
    - mountPath: /mnt/disk2
      name: test-vol2
  volumes:
  - persistentVolumeClaim:
        claimName: mydisk2-2
    name: test-vol2
  - persistentVolumeClaim:
        claimName: mydisk2-1
    name: test-vol1
  restartPolicy: Always
```
## Proposed Design

### localVolume volume plugin

New volume type localVolume will be created and registered as volume plugin in kubelet, `pkg/volume/local_volume/local_volume.go`

```go
type localVolumePlugin struct {
	host volume.VolumeHost
	// decouple creating Recyclers/Deleters/Provisioners by deferring to a function.  Allows for easier testing.
	newRecyclerFunc    func(pvName string, spec *volume.Spec, eventRecorder volume.RecycleEventRecorder, host volume.VolumeHost, volumeConfig volume.VolumeConfig) (volume.Recycler, error)
	newDeleterFunc     func(spec *volume.Spec, host volume.VolumeHost) (volume.Deleter, error)
	newProvisionerFunc func(options volume.VolumeOptions, host volume.VolumeHost) (volume.Provisioner, error)
	config             volume.VolumeConfig
	ldUtil             LVUInterface
}

const (
	localVolumePluginName = "kubernetes.io/local-volume"
)

```

In PodSpec, we shall add a new type of VolumeSource, `LocalVolumeSource`

```go
type LocalVolumeSource struct {
	// The node to host the localVolume volume
	NodeName string `json:"nodename,omitempty"`
	// This the path on which disk or lvm is mounted.
	Path string `json:"path,omitempty"`
	// The size of local disk
	VolumeSize resource.Quantity `json:"volumesize,omitempty"`
	// The type of local disk, it could be "disk", "lvm" now.
	Type LocalVolumeType `json:"type"`
	// The filesystem type of local disk, the default is ext4
	FSType string `json:"fsType,omitempty"`
}
```

### PV/PVC support for the new localVolume

New PersistentVolumeSource will be added for localVolume, so localVolume volume lifecycle could be independent from pod, and localVolume volume could be managed by PV/PVC too.

```go
type PersistentVolumeSource struct {
    ...
	// LocalVolume represents the volume likes disk and lvm
	LocalVolume *LocalVolumeSource `json:"localVolume,omitempty"`
    ...
}
```

The localVolume PV/PVC will have two storage classes,

#### local-ssd storage class
The local-ssd storage class will support the case that application pod need dedicated disks.

* The local-ssd PVs will be created by kubelet when localVolume volume plugin is initialized.
     * The volume plugin read a config file which has available local disk, then format disks and mount the disks to specified path, e.g. /var/vol-pool/disks/<uuid>.
     * If kubelet or node is restarted, the local-ssd PV will not be re-created if the PV for the same disk uuid has been there.
```
[Global]
disk="/dev/vdc /dev/vdd /dev/vde"
```
* PVC will be created either by kubelet or by user, e.g. using kubectl,
    * When Pod with a localVolume volume is created, localVolume volume plugin in kubelet will create a PVC and volume controller bind this PVC to an allocated PV. This PVC will be deleted when Pod with a localVolume volume is deleted.
    * User can create a PVC, and then create a pod by using PVC as a volume. If user delete the pod, the PVC will not be deleted and the data in the volume retained. The data in volume will be removed when PVC is deleted by user.
* `volume.alpha.kubernetes.io/storage-class: local-ssd` must be in the annotations of PVC, and will be changed to `PersistentVolumeClaim.Spec.Class` if this Class attribute is implemented later.
* When user creates a pod by using this type PVC, kube-scheduler will schedule pod to the node where the bound PV is. 
* To support multi-disk in a pod, put a group name and disk number in annotation `volume.beta.kubernetes.io/group`.
* When user deletes the pod, PVC/PV will be remained.
* When user deletes the PVC, the volume plugin's `Recycle()` will be called and all data on the disk of PV will be removed, but PV instance will be kept and be `Available`.
```yaml
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: mydisk2-1
  annotations:
    volume.beta.kubernetes.io/storage-class: "local-ssd"
    volume.beta.kubernetes.io/group: '{"name":"group1", "count":"2"}'
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

#### local-volume storage class
The local-volume storage class will provide the local storage volume which is backed by lvm.

* When localVolume plugin is initialized, it reads available lvm vg name from the config file
```
[Global]
lvm="/dev/atomicos"
```

* A new annotations member `volume.alpha.kubernetes.io/vgallocatable` is added in `Node` object.
* Kubelet on each Node will update VGs' available size in this`Node` annotations `volume.alpha.kubernetes.io/vgallocatable`
* Admin will not need to create PV of lvm, user can create PVC directly and volume dynamic provision controller will create PV for the PVC.
* When user creates a PVC, `volume.alpha.kubernetes.io/storage-class: local-volume` must be in the annotations. The volume provision controller will call plugin's `Provision()`, and will do
    * find a Node with enough lvm VG available size.
    * create a PV of lvm and bind it to PVC.
    * The PV's recyle policy is `Delete`
* When user creates a pod by using this type PVC, kube-scheduler will schedule pod to the node where the bound PV is.
* When user deletes the pod, PVC/PV will be remained and logical volume on the Node will be kept too.
* When user deletes the PVC, the volume plugin's `Delete()` will be called, and it will create a deletion pod to remove logical volume on the Node. Then PV controller will delete PV instnace.


```yaml
apiVersion: v1
kind: Node
metadata:
  annotations:
    volume.alpha.kubernetes.io/vgallocatable: '{"/dev/vg1":"12180"}'
    ...
```


### Scheduling changes

#### Pod scheduling

New predicates function will be required for localVolume resource calculation during pod scheduling. When the function is called, it will check the following

* if request's localVolume volume type is `disk`, it will check available disks in specified Node.
* if request's localVolume volume type is `lvm`, it will check available VolumeGroup(VG) size in specified Node.
* if pod's volume is PVC, return true for the Node which has the localVolume bound by the PVC.

#### PV allocation

If a pod is using localVolume PV, pod’s target host node is selected during PVC creation instead of pod scheduling, so some pod scheduling features need be applied before PV is bound to a PVC,
the features include,

* PVC anti-affinity, we need make sure that host nodes for the allocated localVolumes are in different fault domain, e.g. to make sure no any two localVolume PVs are not in the same node.
* Guarantee the node that pod with localVolume PV is assigned to has enough cpu and memory resource
