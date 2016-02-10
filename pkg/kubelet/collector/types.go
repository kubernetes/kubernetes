/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package collector

import (
	"time"
)

// TODO: These structs are mostly taken directly from cadvisor to minimize the
// amount of changes we need to do in this iteration, but these eventually need
// to be cleaned up further to be less cadvisor-specific and more generic to kubernetes

// Basic node information
type NodeInfo struct {
	// The number of cores on this node
	NumCores int32 `json:"numCores"`

	// The amount of memory on this node
	MemoryCapacityBytes int64 `json:"memoryCapacityBytes"`

	// Machine ID that uniquely identifies a node across reboots or network changes
	MachineID string `json:"machineId"`

	// System UUID reported by a node
	SystemUUID string `json:"systemUuid"`

	// The boot ID of a node, which changes upon reboots
	BootID string `json:"bootId"`
}

// Software version of a node
type VersionInfo struct {
	// Host machine kernel version
	KernelVersion string `json:"kernelVersion"`

	// OS image being used for collector container, or host image if running on host directly
	ContainerOsVersion string `json:"containerOsVersion"`

	// Version of a specific container runtime, e.g. Docker/Rkt
	ContainerRuntimeVersion string `json:"containerRuntimeVersion"`

	// Version of a collector instance
	CollectorVersion string `json:"collectorVersion"`
}

const (
	// Label of the mount point where container top layer file systems are placed
	LabelSystemRoot = "root"

	// Label of the mount point where container images are placed
	LabelImages = "images"
)

// Basic file system information
type FsInfo struct {
	// The block device name associated with the filesystem.
	Device string `json:"device"`

	// Path where the filesystem is mounted.
	MountPoint string `json:"mountPoint"`

	// Total filesystem capacity
	CapacityBytes int64 `json:"capacityBytes"`

	// Bytes available for non-root use.
	AvailableBytes int64 `json:"availableBytes"`

	// Number of bytes used on this filesystem.
	UsedBytes int64 `json:"usedBytes"`

	// Labels associated with this filesystem.
	Labels []string `json:"labels,omitempty"`
}

type EventType string

// Event monitored and returned by collector
type Event struct {
	// The absolute container name for which the event occurred
	ContainerName string `json:"containerName"`

	// The time at which the event occurred
	Timestamp time.Time `json:"timestamp"`

	// The type of event. EventType is an enumerated type
	EventType EventType `json:"eventType"`
}

const (
	EventOom               EventType = "Oom"
	EventOomKill           EventType = "OomKill"
	EventContainerCreation EventType = "ContainerCreation"
	EventContainerDeletion EventType = "ContainerDeletion"
)

// Request sent to the collector to monitor for various event types
type EventRequest struct {
	// EventType is a list of event types to be watched
	EventType []EventType `json:"eventType"`

	// The absolute container name for which the event occurred
	ContainerName string `json:"containerName"`
}
