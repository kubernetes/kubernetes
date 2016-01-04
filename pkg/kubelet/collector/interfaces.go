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
	// connectionURL can be in the following format:
	// - <port>
	// - [<hostname>|<ip>]:<port>
	// If only the port is specified, starts the collector locally
	// Otherwise, uses the connectionURL to connect to a running collector
	Start() error

	// Returns the machine spec including number of cores, amount of memory,
	// file systems, disks, network devices, etc.
	MachineInfo() (*MachineInfo, error)

	// Returns the machine's version info such as kernel and docker version
	VersionInfo() (*VersionInfo, error)

	// Returns the file system information of a given label
	FsInfo(fslabel string) (*FsInfo, error)

	// Returns a channel for watching requested event types
	WatchEvents(request *Request) (chan *Event, error)

	// Get detailed stats (cpu, memory, disk, network) about one or more containers
	// Combines DockerContainer(), ContainerInfo(), and SubContainerinfo() into one interface function
	ContainerInfo(containerName string, req *ContainerInfoRequest, subcontainers bool, isRawContainer bool) (map[string]interface{}, error)
}
