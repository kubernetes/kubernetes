/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package stats

type CoreInfo struct {
	// MachineID reported by the node. For unique machine identification
	// in the cluster this field is prefered. Learn more from man(5)
	// machine-id: http://man7.org/linux/man-pages/man5/machine-id.5.html
	MachineID string `json:"machineID" protobuf:"bytes,1,opt,name=machineID"`
	// SystemUUID reported by the node. For unique machine identification
	// MachineID is prefered. This field is specific to Red Hat hosts
	// https://access.redhat.com/documentation/en-US/Red_Hat_Subscription_Management/1/html/RHSM/getting-system-uuid.html
	SystemUUID string `json:"systemUUID" protobuf:"bytes,2,opt,name=systemUUID"`
	// Boot ID reported by the node.
	BootID string `json:"bootID" protobuf:"bytes,3,opt,name=bootID"`
	// Kernel Version reported by the node from 'uname -r' (e.g. 3.16.0-0.bpo.4-amd64).
	KernelVersion string `json:"kernelVersion" protobuf:"bytes,4,opt,name=kernelVersion"`
	// OS Image reported by the node from /etc/os-release (e.g. Debian GNU/Linux 7 (wheezy)).
	OSImage string `json:"osImage" protobuf:"bytes,5,opt,name=osImage"`
	// Capacity represents the total resources of a node.
	// More info: http://kubernetes.io/docs/user-guide/persistent-volumes#capacity for more details.
	// +optional
	// ContainerRuntime Version reported by the node through runtime remote API (e.g. docker://1.5.0).
	ContainerRuntimeVersion string `json:"containerRuntimeVersion" protobuf:"bytes,6,opt,name=containerRuntimeVersion"`
	// Cloud provider the machine belongs to.
	CloudProvider CloudProvider `json:"cloud_provider"`
	// ID of cloud instance (e.g. instance-1) given to it by the cloud provider.
	InstanceID InstanceID `json:"instance_id"`
}

type CloudProvider string

const (
	GCE             CloudProvider = "GCE"
	AWS                           = "AWS"
	Azure                         = "Azure"
	Baremetal                     = "Baremetal"
	UnknownProvider               = "Unknown"
)

// CoreStats is a top-level container for holding NodeStats and PodStats.
type CoreStats struct {
	// Overall node resource stats.
	Node NodeResources `json:"node"`
	// Per-pod usage stats.
	Pods []PodUsage `json:"pods"`
}

// NodeStats holds node-level stats.  NodeStats contains capacity and availibility for Node Resources.
type NodeResources struct {
	// The filesystem device used by node k8s components.
	// +optional
	KubeletFsDevice string `json:"kubeletfs"`
	// The filesystem device used by node runtime components.
	// +optional
	RuntimeFsDevice string `json:"runtimefs"`
	// Stats pertaining to cpu resources.
	// +optional
	CPU *CpuResources `json:"cpu,omitempty"`
	// Stats pertaining to memory (RAM) resources.
	// +optional
	Memory *MemoryResources `json:"memory,omitempty"`
	// Stats pertaining to node filesystem resources.
	// +optional
	Filesystems []DiskResources `json:"filesystems, omitempty" patchStrategy:"merge" patchMergeKey:"device"`
}

// CpuResources containes data about cpu resource usage
type CpuResources struct {
	// The number of cores in this machine.
	NumCores int `json:"numcores"`
	// The current Usage of CPU resources
	TotalUsage *CpuUsage `json:"cpuusage,omitempty"`
}

// MemoryResources contains data about memory resource usage.
type MemoryResources struct {
	// The time at which these stats were updated.
	Time metav1.Time `json:"time"`
	// The memory capacity, in bytes
	CapacityBytes *uint64 `json:"capacitybytes,omitempty"`
	// The available memory, in bytes
	AvailableBytes *uint64 `json:"availablebytes,omitempty"`
}

// DiskResources contains data about filesystem disk resources.
type DiskResources struct {
	// The time at which these stats were updated.
	Time metav1.Time `json:"time"`
	// The device that this filesystem is on
	Device string `json:"device"`
	// AvailableBytes represents the storage space available (bytes) for the filesystem.
	// +optional
	AvailableBytes *uint64 `json:"availableBytes,omitempty"`
	// CapacityBytes represents the total capacity (bytes) of the filesystems underlying storage.
	// +optional
	CapacityBytes *uint64 `json:"capacityBytes,omitempty"`
	// InodesFree represents the free inodes in the filesystem.
	// +optional
	InodesFree *uint64 `json:"inodesFree,omitempty"`
	// Inodes represents the total inodes in the filesystem.
	// +optional
	Inodes *uint64 `json:"inodes,omitempty"`
}

// PodUsage holds pod-level unprocessed sample stats.
type PodUsage struct {
	// UID of the pod
	PodUID string `json:"uid"`
	// Stats pertaining to pod total usage of cpu
	// This may include additional overhead not included in container usage statistics.
	// +optional
	CPU *CpuUsage `json:"cpu,omitempty"`
	// Stats pertaining to pod total usage of system memory
	// This may include additional overhead not included in container usage statistics.
	// +optional
	Memory *MemoryUsage `json:"memory,omitempty"`
	// Stats of containers in the pod.
	Containers []ContainerUsage `json:"containers" patchStrategy:"merge" patchMergeKey:"uid"`
	// Stats pertaining to volume usage of filesystem resources.
	// +optional
	Volumes []VolumeUsage `json:"volume,omitempty" patchStrategy:"merge" patchMergeKey:"name"`
}

// ContainerUsage holds container-level usage stats.
type ContainerUsage struct {
	// UID of the container
	ContainerUID string `json:"uid"`
	// Stats pertaining to container usage of cpu
	// +optional
	CPU *CpuUsage `json:"memory,omitempty"`
	// Stats pertaining to container usage of system memory
	// +optional
	Memory *MemoryUsage `json:"memory,omitempty"`
	// Stats pertaining to container rootfs usage of disk.
	// Rootfs.UsedBytes is the number of bytes used for the container write layer.
	// +optional
	Rootfs *DiskUsage `json:"rootfs,omitempty"`
	// Stats pertaining to container logs usage of Disk.
	// +optional
	Logs *DiskUsage `json:"logs,omitempty"`
}

// CpuUsage holds statistics about the amount of cpu time consumed
type CpuUsage struct {
	// The time at which these stats were updated.
	Time metav1.Time `json:"time"`
	// Total CPU usage (sum of all cores) averaged over the sample window.
	// The "core" unit can be interpreted as CPU core-nanoseconds per second.
	// +optional
	UsageNanoCores *uint64 `json:"usageNanoCores,omitempty"`
	// Cumulative CPU usage (sum of all cores) since object creation.
	// +optional
	UsageCoreNanoSeconds *uint64 `json:"usageCoreNanoSeconds,omitempty"`
}

// MemoryUsage holds statistics about the quantity of memory consumed
type MemoryUsage struct {
	// The time at which these stats were updated.
	Time metav1.Time `json:"time"`
	// The amount of working set memory. This includes recently accessed memory,
	// dirty memory, and kernel memory.
	// +optional
	WorkingSetBytes *uint64 `json:"workingSetBytes,omitempty"`
}

// VolumeUsage holds statistics about the quantity of disk resources consumed for a volume
type VolumeUsage struct {
	// Embedded DiskUsage
	DiskUsage
	// Name is the name given to the Volume
	// +optional
	Name string `json:"name,omitempty"`
}

// DiskUsage holds statistics about the quantity of disk resources consumed
type DiskUsage struct {
	// The time at which these stats were updated.
	Time metav1.Time `json:"time"`
	// The device on which resources are consumed
	Device string `json:"device"`
	// UsedBytes represents the disk space consumed on the device, in bytes.
	// +optional
	UsedBytes *uint64 `json:"usedBytes,omitempty"`
	// InodesUsed represents the inodes consumed on the device
	// +optional
	InodesUsed *uint64 `json:"inodesUsed,omitempty"`
}
