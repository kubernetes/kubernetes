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
	NumCores uint `json:"num_cores"`

	// The amount of memory (in bytes) on this node
	MemoryCapacity uint64 `json:"memory_capacity"`

	// Machine ID that uniquely identifies a node across reboots or network changes
	MachineID string `json:"machine_id"`

	// System UUID reported by a node
	SystemUUID string `json:"system_uuid"`

	// The boot ID of a node, which changes upon reboots
	BootID string `json:"boot_id"`
}

// Software version of a node
type VersionInfo struct {
	// Kernel version
	KernelVersion string `json:"kernel_version"`

	// OS image being used for collector container, or host image if running on host directly
	ContainerOsVersion string `json:"container_os_version"`

	// Container runtime version, e.g., Docker/Rkt version
	ContainerRuntimeVersion string `json:"container_runtime_version"`

	// Collector version
	CollectorVersion string `json:"collector_version"`
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
	Mountpoint string `json:"mountpoint"`

	// Filesystem usage in bytes.
	Capacity uint64 `json:"capacity"`

	// Bytes available for non-root use.
	Available uint64 `json:"available"`

	// Number of bytes used on this filesystem.
	Usage uint64 `json:"usage"`

	// Labels associated with this filesystem.
	Labels []string `json:"labels"`
}

type EventType string

// Event monitored and returned by collector
type Event struct {
	// The absolute container name for which the event occurred
	ContainerName string `json:"container_name"`

	// The time at which the event occurred
	Timestamp time.Time `json:"timestamp"`

	// The type of event. EventType is an enumerated type
	EventType EventType `json:"event_type"`
}

const (
	EventOom               EventType = "oom"
	EventOomKill           EventType = "oomKill"
	EventContainerCreation EventType = "containerCreation"
	EventContainerDeletion EventType = "containerDeletion"
)

type EventRequest struct {
	// EventType is a list of event types wanted
	EventType map[EventType]bool

	// The absolute container name for which the event occurred
	ContainerName string

	// If IncludeSubcontainers is false, only events occurring in the specific
	// container, and not the subcontainers, will be returned
	IncludeSubcontainers bool
}
