# Add local disk space as a resource for scheudling

## 1. Background

Local disk is useful for containers to store data. It is faster to write data to local disk(mounted into container) than to container. And a container crashing does not remove a pod from a node, so data in local disks is safe across container crashing.

Now `emptyDir` supports local disk, however it does not take disk space into consideration. Then there might be some out of disks errors if containers use too mush disk space. PR [Sticky emptydir proposal](https://github.com/kubernetes/kubernetes/pull/30044) is discussing it. PR [LocalDisk Volume Proposal](https://github.com/kubernetes/kubernetes/pull/30499) make another proposal to support local disk. PV/PVC and dynamic provision suggeted in both PRs.

In this proposal, we'd like to provide another way: add local disk space as a resource for scheduling. Kubelet reports its local disks information, and scheduler takes disk space into consideration if user requests local disk space in spec. There are mainly two reasons for us to add local disk space for scheduling. First, if we treat local disk as a resource like CPU/Memory, it will be more convinient for user to request local disk. Second, if we let kubelet to allocate local disk, scheduler could not know whether there is enough disk space for kubelet to allocate, then kublete might fail to allocate when it tries to launch pod.

# 2. Implementation

## 2.1 Local disk request

At present, we suppose that user request localdisk like [Sticky emptydir proposal](https://github.com/kubernetes/kubernetes/pull/30044) and [LocalDisk Volume Proposal](https://github.com/kubernetes/kubernetes/pull/30499):

```
apiVersion: v1
kind: Pod
metadata:
  name: test-local-disk
spec:
  containers:
  - image: nginx
    name: test-container
    volumeMounts:
    - mountPath: /test  // mount local disk to container
      name: test-local-disk
  volumes:
  - name: test-local-disk
    localDisk:  // request local disk 4Gi
      diskSize: 4
```

The main data struct is:

```
// Represents local disk claim for a pod
type LocalDiskSource struct {
	// Requested local disk size, the unit is Gi
	DiskSize uint32 `json:"diskSize" protobuf:"bytes,1,opt,name=diskSize"`
	// LocalPath represents the local disk path that pod will use. It will be
	// backfilled by scheduler when scheduler find a fit local path on a
	// kubelet node. If user specifies it, it might not be satisfied.
	LocalPath string `json:"localPath,omitempty" protobuf:"bytes,2,opt,name=localPath"`
	// TODO: Add labels to select local disk, e.g. kind=SSD.
}
```

## 2.2 kubelet

Kubelet reports its local disks infomation. We could specify local disks and spaces that could be used by kubelet, e.g. `--local-disks /home/jungong/test/test1,/home/jungong/test/test2 --reserved-local-disk-capacity=2`.

The main data struct is:

```
type LocalDisk struct {
	// Local directory
	LocalDir string `json:"localDir,omitempty"`
	// Total capacity, unit is Gi
	Capacity uint32 `json:"capacity,omitempty"`
	// Allocatable capacity represents the disk capacity that are available for scheduling, Unit is Gi
	Allocatable uint32 `json:"allocatable,omitempty"`
	// TODO: Add labels for local disk, e.g. kind=SSD. They will be helpful for scheduling.
	// Labels map[string]string
}
```

## 2.3 kube-scheduler

Add predicate for local disk space to kube-scheduler to find better kubelet node for pod. At last, if scheduler find a best node for the pod, scheduler will allocate a local path on it for the pod, and backfill it in pod's spec(or backfill it in pod's status). Then kubelet could find out the local path allocated for the pod when running the pod.
