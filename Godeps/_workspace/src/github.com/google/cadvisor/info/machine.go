// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package info

type MachineInfo struct {
	// The number of cores in this machine.
	NumCores int `json:"num_cores"`

	// The amount of memory (in bytes) in this machine
	MemoryCapacity int64 `json:"memory_capacity"`
}

type VersionInfo struct {
	// Kernel version.
	KernelVersion string `json:"kernel_version"`

	// OS image being used for cadvisor container, or host image if running on host directly.
	ContainerOsVersion string `json:"container_os_version"`

	// Docker version.
	DockerVersion string `json:"docker_version"`

	// cAdvisor version.
	CadvisorVersion string `json:"cadvisor_version"`
}

type MachineInfoFactory interface {
	GetMachineInfo() (*MachineInfo, error)
	GetVersionInfo() (*VersionInfo, error)
}
