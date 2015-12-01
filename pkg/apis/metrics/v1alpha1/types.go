/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/types"
)

// MetricsMeta describes metadata that top-level metrics resources must have.
// Metrics data is a synthetic collection of Samples, and thus does not use ObjectMeta.
// All metrics data is read-only.
type MetricsMeta struct {
	// SelfLink is a URL representing this object.
	// Populated by the system.
	SelfLink string `json:"selfLink,omitempty"`
}

// RawNode holds node-level unprocessed sample metrics.
type RawNodeMetrics struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata, since this is a synthetic resource.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	unversioned.ListMeta `json:"metadata"`
	// Reference to the measured Node.
	NodeName string `json:"nodeName"`
	// Overall node metrics.
	Total []AggregateSample `json:"total,omitempty" patchStrategy:"merge" patchMergeKey:"sampleTime"`
	// Metrics of system daemons tracked as raw containers, which may include:
	//   "/kubelet", "/docker-daemon", "kube-proxy" - Tracks respective component metrics
	//   "/system" - Tracks metrics of non-kubernetes and non-kernel processes (grouped together)
	SystemContainers []RawContainerMetrics `json:"systemContainers,omitempty" patchStrategy:"merge" patchMergeKey:"name"`
}

// RawNodeMetricsList holds a list of RawNodeMetrics.
// Endpoints which return this type respect unversioned.ListOptions
type RawNodeMetricsList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	unversioned.ListMeta `json:"metadata,omitempty"`
	// List of raw node metrics.
	Items []RawNodeMetrics `json:"items"`
}

// RawPod holds pod-level unprocessed sample metrics.
type RawPodMetrics struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata, since this is a synthetic resource.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	unversioned.ListMeta `json:"metadata"`
	// Reference to the measured Pod.
	PodRef NonLocalObjectReference `json:"podRef"`
	// Metrics of containers in the measured pod.
	Containers []RawContainerMetrics `json:"containers" patchStrategy:"merge" patchMergeKey:"name"`
	// Historical metric samples of pod-level resources.
	Samples []PodSample `json:"samples" patchStrategy:"merge" patchMergeKey:"sampleTime"`
}

// RawPodMetricsList holds a list of RawPodMetrics.
// Endpoints which return this type respect unversioned.ListOptions
type RawPodMetricsList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	unversioned.ListMeta `json:"metadata,omitempty"`
	// List of raw pod metrics.
	Items []RawPodMetrics `json:"items"`
}

// RawContainerMetrics holds container-level unprocessed sample metrics.
type RawContainerMetrics struct {
	// Reference to the measured container.
	Name string `json:"name"`
	// Metadata labels associated with this container (not Kubernetes labels).
	// For example, docker labels.
	Labels map[string]string `json:"labels,omitempty"`
	// Historical metric samples gathered from the container.
	Samples []ContainerSample `json:"samples" patchStrategy:"merge" patchMergeKey:"sampleTime"`
}

// NonLocalObjectReference contains enough information to locate the referenced object.
type NonLocalObjectReference struct {
	Name      string    `json:"name"`
	Namespace string    `json:"namespace"`
	UID       types.UID `json:"uid,omitempty"`
}

// Sample defines metadata common to all sample types.
// Samples may not be nested within other samples.
type Sample struct {
	// The time this data point was collected at.
	SampleTime unversioned.Time `json:"sampleTime"`
}

// AggregateSample contains a metric sample point of data aggregated across containers.
type AggregateSample struct {
	Sample `json:",inline"`
	// Metrics pertaining to CPU resources.
	CPU *CPUMetrics `json:"cpu,omitempty"`
	// Metrics pertaining to memory (RAM) resources.
	Memory *MemoryMetrics `json:"memory,omitempty"`
	// Metrics pertaining to network resources.
	Network *NetworkMetrics `json:"network,omitempty"`
	// Metrics pertaining to filesystem resources. Reported per-device.
	Filesystem []FilesystemMetrics `json:"filesystem,omitempty" patchStrategy:"merge" patchMergeKey:"device"`
}

// PodSample contains a metric sample point of pod-level resources.
type PodSample struct {
	Sample `json:",inline"`
	// Metrics pertaining to network resources.
	Network *NetworkMetrics `json:"network,omitempty"`
}

// ContainerSample contains a metric sample point of container-level resources.
type ContainerSample struct {
	Sample `json:",inline"`
	// Metrics pertaining to CPU resources.
	CPU *CPUMetrics `json:"cpu,omitempty"`
	// Metrics pertaining to memory (RAM) resources.
	Memory *MemoryMetrics `json:"memory,omitempty"`
	// Metrics pertaining to filesystem resources. Reported per-device.
	Filesystem []FilesystemMetrics `json:"filesystem,omitempty" patchStrategy:"merge" patchMergeKey:"device"`
}

// NetworkMetrics contains data about network resources.
type NetworkMetrics struct {
	// Cumulative count of bytes received.
	RxBytes *resource.Quantity `json:"rxBytes,omitempty"`
	// Cumulative count of receive errors encountered.
	RxErrors *int64 `json:"rxErrors,omitempty"`
	// Cumulative count of bytes transmitted.
	TxBytes *resource.Quantity `json:"txBytes,omitempty"`
	// Cumulative count of transmit errors encountered.
	TxErrors *int64 `json:"txErrors,omitempty"`
}

// CPUMetrics contains data about CPU usage.
type CPUMetrics struct {
	// Total CPU usage (sum of all cores) averaged over the sample window.
	// The "core" unit can be interpreted as CPU core-seconds per second.
	TotalCores *resource.Quantity `json:"totalCores,omitempty"`
}

// MemoryMetrics contains data about memory usage.
type MemoryMetrics struct {
	// Total memory in use. This includes all memory regardless of when it was accessed.
	TotalBytes *resource.Quantity `json:"totalBytes,omitempty"`
	// The amount of working set memory. This includes recently accessed memory,
	// dirty memory, and kernel memory. UsageBytes is <= TotalBytes.
	UsageBytes *resource.Quantity `json:"usageBytes,omitempty"`
	// Cumulative number of minor page faults.
	PageFaults *int64 `json:"pageFaults,omitempty"`
	// Cumulative number of major page faults.
	MajorPageFaults *int64 `json:"majorPageFaults,omitempty"`
}

// FilesystemMetrics contains data about filesystem usage.
type FilesystemMetrics struct {
	// The block device name associated with the filesystem.
	Device string `json:"device"`
	// Number of bytes that is consumed by the container on this filesystem.
	UsageBytes *resource.Quantity `json:"usageBytes,omitempty"`
	// Number of bytes that can be consumed by the container on this filesystem.
	LimitBytes *resource.Quantity `json:"limitBytes,omitempty"`
}

// RawMetricsOptions are the query options for raw metrics endpoints.
type RawMetricsOptions struct {
	// Specifies the maximum number of elements in any list of samples.
	// When the total number of samples exceeds the maximum the most recent samples are returned.
	// Defaults to unlimited. Minimum value 1.
	MaxSamples int `json:"maxSamples,omitempty"`
}
