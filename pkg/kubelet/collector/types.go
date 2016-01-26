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

package collector

import (
	"time"
)

// These types are mostly taken directly from cadvisor to minimize the amount of changes
// we need to do in this iteration, but these need to be cleaned up further to
// be less cadvisor-specific and more generic to kubernetes

// Simplified version of cadvisor.MachineInfo
type MachineInfo struct {
	// The number of cores in this machine.
	NumCores int

	// The amount of memory (in bytes) in this machine
	MemoryCapacity int64

	// The machine id
	MachineID string

	// The system uuid
	SystemUUID string

	// The boot id
	BootID string
}

// Simplified version of cadvisor.VersionInfo
type VersionInfo struct {
	// Kernel version.
	KernelVersion string

	// OS image being used for cadvisor container, or host image if running on host directly.
	ContainerOsVersion string

	// Docker version.
	DockerVersion string

	// Collector version.
	CollectorVersion string
}

const (
	LabelSystemRoot   = "root"
	LabelDockerImages = "docker-images"
)

// Same as cadvisor.FsInfo
type FsInfo struct {
	// The block device name associated with the filesystem.
	Device string

	// Path where the filesystem is mounted.
	Mountpoint string

	// Filesystem usage in bytes.
	Capacity uint64

	// Bytes available for non-root use.
	Available uint64

	// Number of bytes used on this filesystem.
	Usage uint64

	// Labels associated with this filesystem.
	Labels []string
}

// Simplified version of cadvisor.Event
type Event struct {
	// the absolute container name for which the event occurred
	ContainerName string

	// the time at which the event occurred
	Timestamp time.Time

	// the type of event. EventType is an enumerated type
	EventType EventType
}

type EventType string

const (
	EventOom               EventType = "oom"
	EventOomKill                     = "oomKill"
	EventContainerCreation           = "containerCreation"
	EventContainerDeletion           = "containerDeletion"
)

type Request struct {
	// EventType is a list of event types wanted
	EventType map[EventType]bool

	// the absolute container name for which the event occurred
	ContainerName string

	// if IncludeSubcontainers is false, only events occurring in the specific
	// container, and not the subcontainers, will be returned
	IncludeSubcontainers bool
}

// ContainerInfoRequest is used when users check a container info from the REST API.
// It specifies how much data users want to get about a container
type ContainerInfoRequest struct {
	// Max number of stats to return. Specify -1 for all stats currently available.
	// Default: 60
	NumStats int

	// Start time for which to query information.
	// If ommitted, the beginning of time is assumed.
	Start time.Time

	// End time for which to query information.
	// If ommitted, current time is assumed.
	End time.Time
}
