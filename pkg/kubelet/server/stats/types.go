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

package stats

import (
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

// Summary is a top-level container for holding NodeStats and PodStats.
type Summary struct {
	// The time the most recent data included in this summary was collect at, rounded to the nearest
	// second.
	Time unversioned.Time `json:"time"`
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
	// Stats pertaining to CPU resources.
	CPU *CPUStats `json:"cpu,omitempty"`
	// Stats pertaining to memory (RAM) resources.
	Memory *MemoryStats `json:"memory,omitempty"`
	// Stats pertaining to network resources.
	Network *NetworkStats `json:"network,omitempty"`
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
	PodRef NonLocalObjectReference `json:"podRef"`
	// Stats of containers in the measured pod.
	Containers []ContainerStats `json:"containers" patchStrategy:"merge" patchMergeKey:"name"`
	// Stats pertaining to network resources.
	Network *NetworkStats `json:"network,omitempty"`
}

// ContainerStats holds container-level unprocessed sample stats.
type ContainerStats struct {
	// Reference to the measured container.
	Name string `json:"name"`
	// Stats pertaining to CPU resources.
	CPU *CPUStats `json:"cpu,omitempty"`
	// Stats pertaining to memory (RAM) resources.
	Memory *MemoryStats `json:"memory,omitempty"`
}

// NonLocalObjectReference contains enough information to locate the referenced object.
type NonLocalObjectReference struct {
	Name      string `json:"name"`
	Namespace string `json:"namespace"`
}

// NetworkStats contains data about network resources.
type NetworkStats struct {
	// Cumulative count of bytes received.
	RxBytes *resource.Quantity `json:"rxBytes,omitempty"`
	// Cumulative count of receive errors encountered.
	RxErrors *int64 `json:"rxErrors,omitempty"`
	// Cumulative count of bytes transmitted.
	TxBytes *resource.Quantity `json:"txBytes,omitempty"`
	// Cumulative count of transmit errors encountered.
	TxErrors *int64 `json:"txErrors,omitempty"`
}

// CPUStats contains data about CPU usage.
type CPUStats struct {
	// Total CPU usage (sum of all cores) averaged over the sample window.
	// The "core" unit can be interpreted as CPU core-seconds per second.
	UsageCores *resource.Quantity `json:"usageCores,omitempty"`
	// Cumulative CPU usage (sum of all cores) since object creation.
	UsageCoreSeconds *resource.Quantity `json:"usageCoreSeconds,omitempty"`
}

// MemoryStats contains data about memory usage.
type MemoryStats struct {
	// Total memory in use. This includes all memory regardless of when it was accessed.
	UsageBytes *resource.Quantity `json:"usageBytes,omitempty"`
	// The amount of working set memory. This includes recently accessed memory,
	// dirty memory, and kernel memory. UsageBytes is <= TotalBytes.
	WorkingSetBytes *resource.Quantity `json:"workingSetBytes,omitempty"`
	// Cumulative number of minor page faults.
	PageFaults *int64 `json:"pageFaults,omitempty"`
	// Cumulative number of major page faults.
	MajorPageFaults *int64 `json:"majorPageFaults,omitempty"`
}
