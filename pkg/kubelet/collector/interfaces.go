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

type Interface interface {
	// Starts the collector process if necessary, e.g., the builtin cadvisor
	// Otherwise, does nothing and just returns
	Start() error

	// Returns the basic machine spec
	// Currently used to report to k8s master the basic information about a node
	MachineInfo() (*MachineInfo, error)

	// Returns the software version of various components of a node, e.g., kernel, container runtime, etc.
	// Currently used to report back to k8s master the versions of relevant software on a node
	VersionInfo() (*VersionInfo, error)

	// Returns the file system information of a given label or mount point
	// Currently used by kubelet to check if there's enough disk space to schedule new pods and
	// by the image manager to decide if garbage collection should start
	FsInfo(fslabel string) (*FsInfo, error)

	// Returns a channel for watching requested event types
	// Currently used by oom watcher to detect oom events.
	WatchEvents(request *Request) (chan *Event, error)

	// Get detailed metrics information (cpu, memory, disk, network, etc.) about one or more containers
	// When isRawContainer=true, containerName refers to a raw Linux container; otherwise it refers to a Docker container
	// When isRawContainer=true and subcontainer=true, returns not only the metrics information of this container, but
	// also the child containers
	// Returned value is a map of container name to its metrics data, and this data is in a collector-specific format
	// Currently used by kubelet to handle /stats/... REST calls
	ContainerInfo(containerName string, req *ContainerInfoRequest, subcontainers bool, isRawContainer bool) (map[string]interface{}, error)

	// ContainerInfo() returns collector-specific raw data, which should be parsed to retrieve its various components,
	// e.g., cpu, memory, cpu load, file system, etc., thus the raw data should be cached locally to avoid frequent polls.
	// These parsing interface functions, e.g., GetContainerCPUInfo(), GetContainerMemoryInfo(), GetContainerNetworkInfo(),
	// etc., should be defined to retrieve such data in a format defined by Kubernetes. However, as Kubernetes is currently
	// not using any of these metrics data, these interface functions (along with the corresponding structs) should be
	// defined and tuned at the time when such code is added.

}
