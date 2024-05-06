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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Summary is a top-level container for holding NodeStats and PodStats.
type Summary struct {
	// Overall node stats.
	Node NodeStats `json:"node"`
	// Per-pod stats.
	Pods []PodStats `json:"pods"`
}

// NodeStats holds node-level unprocessed sample stats.
type NodeStats struct {
	// Reference to the measured Node.
	NodeName string `json:"nodeName"`
	// Stats of system daemons tracked as raw containers.
	// The system containers are named according to the SystemContainer* constants.
	// +optional
	// +patchMergeKey=name
	// +patchStrategy=merge
	SystemContainers []ContainerStats `json:"systemContainers,omitempty" patchStrategy:"merge" patchMergeKey:"name"`
	// The time at which data collection for the node-scoped (i.e. aggregate) stats was (re)started.
	StartTime metav1.Time `json:"startTime"`
	// Stats pertaining to CPU resources.
	// +optional
	CPU *CPUStats `json:"cpu,omitempty"`
	// Stats pertaining to memory (RAM) resources.
	// +optional
	Memory *MemoryStats `json:"memory,omitempty"`
	// Stats pertaining to network resources.
	// +optional
	Network *NetworkStats `json:"network,omitempty"`
	// Stats pertaining to total usage of filesystem resources on the rootfs used by node k8s components.
	// NodeFs.Used is the total bytes used on the filesystem.
	// +optional
	Fs *FsStats `json:"fs,omitempty"`
	// Stats about the underlying container runtime.
	// +optional
	Runtime *RuntimeStats `json:"runtime,omitempty"`
	// Stats about the rlimit of system.
	// +optional
	Rlimit *RlimitStats `json:"rlimit,omitempty"`
	// Stats pertaining to swap resources. This is reported to non-windows systems only.
	// +optional
	Swap *SwapStats `json:"swap,omitempty"`
}

// RlimitStats are stats rlimit of OS.
type RlimitStats struct {
	Time metav1.Time `json:"time"`

	// The max number of extant process (threads, precisely on Linux) of OS. See RLIMIT_NPROC in getrlimit(2).
	// The operating system ceiling on the number of process IDs that can be assigned.
	// On Linux, tasks (either processes or threads) consume 1 PID each.
	MaxPID *int64 `json:"maxpid,omitempty"`
	// The number of running process (threads, precisely on Linux) in the OS.
	NumOfRunningProcesses *int64 `json:"curproc,omitempty"`
}

// RuntimeStats are stats pertaining to the underlying container runtime.
type RuntimeStats struct {
	// Stats about the underlying filesystem where container images are stored.
	// This filesystem could be the same as the primary (root) filesystem.
	// Usage here refers to the total number of bytes occupied by images on the filesystem.
	// +optional
	ImageFs *FsStats `json:"imageFs,omitempty"`
	// Stats about the underlying filesystem where container's writeable layer is stored.
	// This filesystem could be the same as the primary (root) filesystem or the ImageFS.
	// Usage here refers to the total number of bytes occupied by the writeable layer on the filesystem.
	// +optional
	ContainerFs *FsStats `json:"containerFs,omitempty"`
}

const (
	// SystemContainerKubelet is the container name for the system container tracking Kubelet usage.
	SystemContainerKubelet = "kubelet"
	// SystemContainerRuntime is the container name for the system container tracking the runtime (e.g. docker) usage.
	SystemContainerRuntime = "runtime"
	// SystemContainerMisc is the container name for the system container tracking non-kubernetes processes.
	SystemContainerMisc = "misc"
	// SystemContainerPods is the container name for the system container tracking user pods.
	SystemContainerPods = "pods"
)

// ProcessStats are stats pertaining to processes.
type ProcessStats struct {
	// Number of processes
	// +optional
	ProcessCount *uint64 `json:"process_count,omitempty"`
}

// PodStats holds pod-level unprocessed sample stats.
type PodStats struct {
	// Reference to the measured Pod.
	PodRef PodReference `json:"podRef"`
	// The time at which data collection for the pod-scoped (e.g. network) stats was (re)started.
	StartTime metav1.Time `json:"startTime"`
	// Stats of containers in the measured pod.
	// +patchMergeKey=name
	// +patchStrategy=merge
	Containers []ContainerStats `json:"containers" patchStrategy:"merge" patchMergeKey:"name"`
	// Stats pertaining to CPU resources consumed by pod cgroup (which includes all containers' resource usage and pod overhead).
	// +optional
	CPU *CPUStats `json:"cpu,omitempty"`
	// Stats pertaining to memory (RAM) resources consumed by pod cgroup (which includes all containers' resource usage and pod overhead).
	// +optional
	Memory *MemoryStats `json:"memory,omitempty"`
	// Stats pertaining to network resources.
	// +optional
	Network *NetworkStats `json:"network,omitempty"`
	// Stats pertaining to volume usage of filesystem resources.
	// VolumeStats.UsedBytes is the number of bytes used by the Volume
	// +optional
	// +patchMergeKey=name
	// +patchStrategy=merge
	VolumeStats []VolumeStats `json:"volume,omitempty" patchStrategy:"merge" patchMergeKey:"name"`
	// EphemeralStorage reports the total filesystem usage for the containers and emptyDir-backed volumes in the measured Pod.
	// +optional
	EphemeralStorage *FsStats `json:"ephemeral-storage,omitempty"`
	// ProcessStats pertaining to processes.
	// +optional
	ProcessStats *ProcessStats `json:"process_stats,omitempty"`
	// Stats pertaining to swap resources. This is reported to non-windows systems only.
	// +optional
	Swap *SwapStats `json:"swap,omitempty"`
}

// ContainerStats holds container-level unprocessed sample stats.
type ContainerStats struct {
	// Reference to the measured container.
	Name string `json:"name"`
	// The time at which data collection for this container was (re)started.
	StartTime metav1.Time `json:"startTime"`
	// Stats pertaining to CPU resources.
	// +optional
	CPU *CPUStats `json:"cpu,omitempty"`
	// Stats pertaining to memory (RAM) resources.
	// +optional
	Memory *MemoryStats `json:"memory,omitempty"`
	// Metrics for Accelerators. Each Accelerator corresponds to one element in the array.
	Accelerators []AcceleratorStats `json:"accelerators,omitempty"`
	// Stats pertaining to container rootfs usage of filesystem resources.
	// Rootfs.UsedBytes is the number of bytes used for the container write layer.
	// +optional
	Rootfs *FsStats `json:"rootfs,omitempty"`
	// Stats pertaining to container logs usage of filesystem resources.
	// Logs.UsedBytes is the number of bytes used for the container logs.
	// +optional
	Logs *FsStats `json:"logs,omitempty"`
	// User defined metrics that are exposed by containers in the pod. Typically, we expect only one container in the pod to be exposing user defined metrics. In the event of multiple containers exposing metrics, they will be combined here.
	// +patchMergeKey=name
	// +patchStrategy=merge
	UserDefinedMetrics []UserDefinedMetric `json:"userDefinedMetrics,omitempty" patchStrategy:"merge" patchMergeKey:"name"`
	// Stats pertaining to swap resources. This is reported to non-windows systems only.
	// +optional
	Swap *SwapStats `json:"swap,omitempty"`
}

// PodReference contains enough information to locate the referenced pod.
type PodReference struct {
	Name      string `json:"name"`
	Namespace string `json:"namespace"`
	UID       string `json:"uid"`
}

// InterfaceStats contains resource value data about interface.
type InterfaceStats struct {
	// The name of the interface
	Name string `json:"name"`
	// Cumulative count of bytes received.
	// +optional
	RxBytes *uint64 `json:"rxBytes,omitempty"`
	// Cumulative count of receive errors encountered.
	// +optional
	RxErrors *uint64 `json:"rxErrors,omitempty"`
	// Cumulative count of bytes transmitted.
	// +optional
	TxBytes *uint64 `json:"txBytes,omitempty"`
	// Cumulative count of transmit errors encountered.
	// +optional
	TxErrors *uint64 `json:"txErrors,omitempty"`
}

// NetworkStats contains data about network resources.
type NetworkStats struct {
	// The time at which these stats were updated.
	Time metav1.Time `json:"time"`

	// Stats for the default interface, if found
	InterfaceStats `json:",inline"`

	Interfaces []InterfaceStats `json:"interfaces,omitempty"`
}

// CPUStats contains data about CPU usage.
type CPUStats struct {
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

// MemoryStats contains data about memory usage.
type MemoryStats struct {
	// The time at which these stats were updated.
	Time metav1.Time `json:"time"`
	// Available memory for use.  This is defined as the memory limit - workingSetBytes.
	// If memory limit is undefined, the available bytes is omitted.
	// +optional
	AvailableBytes *uint64 `json:"availableBytes,omitempty"`
	// Total memory in use. This includes all memory regardless of when it was accessed.
	// +optional
	UsageBytes *uint64 `json:"usageBytes,omitempty"`
	// The amount of working set memory. This includes recently accessed memory,
	// dirty memory, and kernel memory. WorkingSetBytes is <= UsageBytes
	// +optional
	WorkingSetBytes *uint64 `json:"workingSetBytes,omitempty"`
	// The amount of anonymous and swap cache memory (includes transparent
	// hugepages).
	// +optional
	RSSBytes *uint64 `json:"rssBytes,omitempty"`
	// Cumulative number of minor page faults.
	// +optional
	PageFaults *uint64 `json:"pageFaults,omitempty"`
	// Cumulative number of major page faults.
	// +optional
	MajorPageFaults *uint64 `json:"majorPageFaults,omitempty"`
}

// SwapStats contains data about memory usage
type SwapStats struct {
	// The time at which these stats were updated.
	Time metav1.Time `json:"time"`
	// Available swap memory for use.  This is defined as the <swap-limit> - <current-swap-usage>.
	// If swap limit is undefined, this value is omitted.
	// +optional
	SwapAvailableBytes *uint64 `json:"swapAvailableBytes,omitempty"`
	// Total swap memory in use.
	// +optional
	SwapUsageBytes *uint64 `json:"swapUsageBytes,omitempty"`
}

// AcceleratorStats contains stats for accelerators attached to the container.
type AcceleratorStats struct {
	// Make of the accelerator (nvidia, amd, google etc.)
	Make string `json:"make"`

	// Model of the accelerator (tesla-p100, tesla-k80 etc.)
	Model string `json:"model"`

	// ID of the accelerator.
	ID string `json:"id"`

	// Total accelerator memory.
	// unit: bytes
	MemoryTotal uint64 `json:"memoryTotal"`

	// Total accelerator memory allocated.
	// unit: bytes
	MemoryUsed uint64 `json:"memoryUsed"`

	// Percent of time over the past sample period (10s) during which
	// the accelerator was actively processing.
	DutyCycle uint64 `json:"dutyCycle"`
}

// VolumeStats contains data about Volume filesystem usage.
type VolumeStats struct {
	// Embedded FsStats
	FsStats `json:",inline"`
	// Name is the name given to the Volume
	// +optional
	Name string `json:"name,omitempty"`
	// Reference to the PVC, if one exists
	// +optional
	PVCRef *PVCReference `json:"pvcRef,omitempty"`

	// VolumeHealthStats contains data about volume health
	// +optional
	VolumeHealthStats *VolumeHealthStats `json:"volumeHealthStats,omitempty"`
}

// VolumeHealthStats contains data about volume health.
type VolumeHealthStats struct {
	// Normal volumes are available for use and operating optimally.
	// An abnormal volume does not meet these criteria.
	Abnormal bool `json:"abnormal"`
}

// PVCReference contains enough information to describe the referenced PVC.
type PVCReference struct {
	Name      string `json:"name"`
	Namespace string `json:"namespace"`
}

// FsStats contains data about filesystem usage.
type FsStats struct {
	// The time at which these stats were updated.
	Time metav1.Time `json:"time"`
	// AvailableBytes represents the storage space available (bytes) for the filesystem.
	// +optional
	AvailableBytes *uint64 `json:"availableBytes,omitempty"`
	// CapacityBytes represents the total capacity (bytes) of the filesystems underlying storage.
	// +optional
	CapacityBytes *uint64 `json:"capacityBytes,omitempty"`
	// UsedBytes represents the bytes used for a specific task on the filesystem.
	// This may differ from the total bytes used on the filesystem and may not equal CapacityBytes - AvailableBytes.
	// e.g. For ContainerStats.Rootfs this is the bytes used by the container rootfs on the filesystem.
	// +optional
	UsedBytes *uint64 `json:"usedBytes,omitempty"`
	// InodesFree represents the free inodes in the filesystem.
	// +optional
	InodesFree *uint64 `json:"inodesFree,omitempty"`
	// Inodes represents the total inodes in the filesystem.
	// +optional
	Inodes *uint64 `json:"inodes,omitempty"`
	// InodesUsed represents the inodes used by the filesystem
	// This may not equal Inodes - InodesFree because this filesystem may share inodes with other "filesystems"
	// e.g. For ContainerStats.Rootfs, this is the inodes used only by that container, and does not count inodes used by other containers.
	InodesUsed *uint64 `json:"inodesUsed,omitempty"`
}

// UserDefinedMetricType defines how the metric should be interpreted by the user.
type UserDefinedMetricType string

const (
	// MetricGauge is an instantaneous value. May increase or decrease.
	MetricGauge UserDefinedMetricType = "gauge"

	// MetricCumulative is a counter-like value that is only expected to increase.
	MetricCumulative UserDefinedMetricType = "cumulative"

	// MetricDelta is a rate over a time period.
	MetricDelta UserDefinedMetricType = "delta"
)

// UserDefinedMetricDescriptor contains metadata that describes a user defined metric.
type UserDefinedMetricDescriptor struct {
	// The name of the metric.
	Name string `json:"name"`

	// Type of the metric.
	Type UserDefinedMetricType `json:"type"`

	// Display Units for the stats.
	Units string `json:"units"`

	// Metadata labels associated with this metric.
	// +optional
	Labels map[string]string `json:"labels,omitempty"`
}

// UserDefinedMetric represents a metric defined and generated by users.
type UserDefinedMetric struct {
	UserDefinedMetricDescriptor `json:",inline"`
	// The time at which these stats were updated.
	Time metav1.Time `json:"time"`
	// Value of the metric. Float64s have 53 bit precision.
	// We do not foresee any metrics exceeding that value.
	Value float64 `json:"value"`
}
