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

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
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
	SystemContainers []ContainerStats `json:"systemContainers,omitempty" patchStrategy:"merge" patchMergeKey:"name"`
	// The time at which data collection for the node-scoped (i.e. aggregate) stats was (re)started.
	StartTime unversioned.Time `json:"startTime"`
	// Stats pertaining to CPU resources.
	CPU *CPUStats `json:"cpu,omitempty"`
	// Stats pertaining to memory (RAM) resources.
	Memory *MemoryStats `json:"memory,omitempty"`
	// Stats pertaining to network resources.
	Network *NetworkStats `json:"network,omitempty"`
	// Stats pertaining to total usage of filesystem resources on the rootfs used by node k8s components.
	// NodeFs.Used is the total bytes used on the filesystem.
	Fs *FsStats `json:"fs,omitempty"`
	// Stats about the underlying container runtime.
	Runtime *RuntimeStats `json:"runtime,omitempty"`
}

// Stats pertaining to the underlying container runtime.
type RuntimeStats struct {
	// Stats about the underlying filesystem where container images are stored.
	// This filesystem could be the same as the primary (root) filesystem.
	// Usage here refers to the total number of bytes occupied by images on the filesystem.
	ImageFs *FsStats `json:"imageFs,omitempty"`
}

const (
	// Container name for the system container tracking Kubelet usage.
	SystemContainerKubelet = "kubelet"
	// Container name for the system container tracking the runtime (e.g. docker or rkt) usage.
	SystemContainerRuntime = "runtime"
	// Container name for the system container tracking non-kubernetes processes.
	SystemContainerMisc = "misc"
)

// PodStats holds pod-level unprocessed sample stats.
type PodStats struct {
	// Reference to the measured Pod.
	PodRef PodReference `json:"podRef"`
	// The time at which data collection for the pod-scoped (e.g. network) stats was (re)started.
	StartTime unversioned.Time `json:"startTime"`
	// Stats of containers in the measured pod.
	Containers []ContainerStats `json:"containers" patchStrategy:"merge" patchMergeKey:"name"`
	// Stats pertaining to network resources.
	Network *NetworkStats `json:"network,omitempty"`
	// Stats pertaining to volume usage of filesystem resources.
	// VolumeStats.UsedBytes is the number of bytes used by the Volume
	VolumeStats []VolumeStats `json:"volume,omitempty" patchStrategy:"merge" patchMergeKey:"name"`
}

// ContainerStats holds container-level unprocessed sample stats.
type ContainerStats struct {
	// Reference to the measured container.
	Name string `json:"name"`
	// The time at which data collection for this container was (re)started.
	StartTime unversioned.Time `json:"startTime"`
	// Stats pertaining to CPU resources.
	CPU *CPUStats `json:"cpu,omitempty"`
	// Stats pertaining to memory (RAM) resources.
	Memory *MemoryStats `json:"memory,omitempty"`
	// Stats pertaining to container rootfs usage of filesystem resources.
	// Rootfs.UsedBytes is the number of bytes used for the container write layer.
	Rootfs *FsStats `json:"rootfs,omitempty"`
	// Stats pertaining to container logs usage of filesystem resources.
	// Logs.UsedBytes is the number of bytes used for the container logs.
	Logs *FsStats `json:"logs,omitempty"`
	// User defined metrics that are exposed by containers in the pod. Typically, we expect only one container in the pod to be exposing user defined metrics. In the event of multiple containers exposing metrics, they will be combined here.
	UserDefinedMetrics []UserDefinedMetric `json:"userDefinedMetrics,omitmepty" patchStrategy:"merge" patchMergeKey:"name"`
}

// PodReference contains enough information to locate the referenced pod.
type PodReference struct {
	Name      string `json:"name"`
	Namespace string `json:"namespace"`
	UID       string `json:"uid"`
}

// NetworkStats contains data about network resources.
type NetworkStats struct {
	// The time at which these stats were updated.
	Time unversioned.Time `json:"time"`
	// Cumulative count of bytes received.
	RxBytes *uint64 `json:"rxBytes,omitempty"`
	// Cumulative count of receive errors encountered.
	RxErrors *uint64 `json:"rxErrors,omitempty"`
	// Cumulative count of bytes transmitted.
	TxBytes *uint64 `json:"txBytes,omitempty"`
	// Cumulative count of transmit errors encountered.
	TxErrors *uint64 `json:"txErrors,omitempty"`
}

// CPUStats contains data about CPU usage.
type CPUStats struct {
	// The time at which these stats were updated.
	Time unversioned.Time `json:"time"`
	// Total CPU usage (sum of all cores) averaged over the sample window.
	// The "core" unit can be interpreted as CPU core-nanoseconds per second.
	UsageNanoCores *uint64 `json:"usageNanoCores,omitempty"`
	// Cumulative CPU usage (sum of all cores) since object creation.
	UsageCoreNanoSeconds *uint64 `json:"usageCoreNanoSeconds,omitempty"`
}

// MemoryStats contains data about memory usage.
type MemoryStats struct {
	// The time at which these stats were updated.
	Time unversioned.Time `json:"time"`
	// Available memory for use.  This is defined as the memory limit - workingSetBytes.
	// If memory limit is undefined, the available bytes is omitted.
	AvailableBytes *uint64 `json:"availableBytes,omitempty"`
	// Total memory in use. This includes all memory regardless of when it was accessed.
	UsageBytes *uint64 `json:"usageBytes,omitempty"`
	// The amount of working set memory. This includes recently accessed memory,
	// dirty memory, and kernel memory. WorkingSetBytes is <= UsageBytes
	WorkingSetBytes *uint64 `json:"workingSetBytes,omitempty"`
	// The amount of anonymous and swap cache memory (includes transparent
	// hugepages).
	RSSBytes *uint64 `json:"rssBytes,omitempty"`
	// Cumulative number of minor page faults.
	PageFaults *uint64 `json:"pageFaults,omitempty"`
	// Cumulative number of major page faults.
	MajorPageFaults *uint64 `json:"majorPageFaults,omitempty"`
}

// VolumeStats contains data about Volume filesystem usage.
type VolumeStats struct {
	// Embedded FsStats
	FsStats
	// Name is the name given to the Volume
	Name string `json:"name,omitempty"`
}

// FsStats contains data about filesystem usage.
type FsStats struct {
	// AvailableBytes represents the storage space available (bytes) for the filesystem.
	AvailableBytes *uint64 `json:"availableBytes,omitempty"`
	// CapacityBytes represents the total capacity (bytes) of the filesystems underlying storage.
	CapacityBytes *uint64 `json:"capacityBytes,omitempty"`
	// UsedBytes represents the bytes used for a specific task on the filesystem.
	// This may differ from the total bytes used on the filesystem and may not equal CapacityBytes - AvailableBytes.
	// e.g. For ContainerStats.Rootfs this is the bytes used by the container rootfs on the filesystem.
	UsedBytes *uint64 `json:"usedBytes,omitempty"`
	// InodesFree represents the free inodes in the filesystem.
	InodesFree *uint64 `json:"inodesFree,omitempty"`
	// Inodes represents the total inodes in the filesystem.
	Inodes *uint64 `json:"inodes,omitempty"`
}

// UserDefinedMetricType defines how the metric should be interpreted by the user.
type UserDefinedMetricType string

const (
	// Instantaneous value. May increase or decrease.
	MetricGauge UserDefinedMetricType = "gauge"

	// A counter-like value that is only expected to increase.
	MetricCumulative UserDefinedMetricType = "cumulative"

	// Rate over a time period.
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
	Labels map[string]string `json:"labels,omitempty"`
}

// UserDefinedMetric represents a metric defined and generate by users.
type UserDefinedMetric struct {
	UserDefinedMetricDescriptor `json:",inline"`
	// The time at which these stats were updated.
	Time unversioned.Time `json:"time"`
	// Value of the metric. Float64s have 53 bit precision.
	// We do not foresee any metrics exceeding that value.
	Value float64 `json:"value"`
}
