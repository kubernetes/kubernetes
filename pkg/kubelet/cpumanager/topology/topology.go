/*
Copyright 2017 The Kubernetes Authors.

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

package topology

import (
	"fmt"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/kubernetes/pkg/kubelet/cpuset"
)

type cpuDetails map[int]CPUInfo

// CPUTopology contains details of node cpu, where :
// CPU  - logical CPU, cadvisor - thread
// Core - physical CPU, cadvisor - Core
// Socket - socket, cadvisor - Node
type CPUTopology struct {
	NumCPUs          int
	NumCores         int
	HyperThreading   bool
	NumSockets       int
	CPUtopoDetails   cpuDetails
	NumReservedCores int
}

// CPUInfo holds information on where cpu is
type CPUInfo struct {
	SocketID int
	CoreID   int
}

// Returns a new cpuDetails object with only `cpus` remaining.
func (d cpuDetails) KeepOnly(cpus cpuset.CPUSet) cpuDetails {
	result := cpuDetails{}
	for cpu, info := range d {
		if cpus.Contains(cpu) {
			result[cpu] = info
		}
	}
	return result
}

func (d cpuDetails) Sockets() cpuset.CPUSet {
	result := cpuset.NewCPUSet()
	for _, info := range d {
		result.Add(info.SocketID)
	}
	return result
}

func (d cpuDetails) CPUsInSocket(id int) cpuset.CPUSet {
	result := cpuset.NewCPUSet()
	for cpu, info := range d {
		if info.SocketID == id {
			result.Add(cpu)
		}
	}
	return result
}

func (d cpuDetails) CoresInSocket(id int) cpuset.CPUSet {
	result := cpuset.NewCPUSet()
	for _, info := range d {
		if info.SocketID == id {
			result.Add(info.CoreID)
		}
	}
	return result
}

func (d cpuDetails) CPUsInCore(id int) cpuset.CPUSet {
	result := cpuset.NewCPUSet()
	for cpu, info := range d {
		if info.CoreID == id {
			result.Add(cpu)
		}
	}
	return result
}

// Discover returns CPUTopology based on cadvisor node info
func Discover(machineInfo *cadvisorapi.MachineInfo) (*CPUTopology, error) {

	if machineInfo.NumCores == 0 {
		return nil, fmt.Errorf("could not detect number of cpus")
	}

	CPUtopoDetails := cpuDetails{}

	numCPUs := machineInfo.NumCores
	htEnabled := false
	numPhysicalCores := 0
	for _, socket := range machineInfo.Topology {
		numPhysicalCores += len(socket.Cores)
		for _, core := range socket.Cores {
			for _, cpu := range core.Threads {
				CPUtopoDetails[cpu] = CPUInfo{
					CoreID:   core.Id,
					SocketID: socket.Id,
				}
				// a little bit naive
				if !htEnabled && len(core.Threads) != 1 {
					htEnabled = true
				}
			}
		}
	}

	return &CPUTopology{
		NumCPUs:        numCPUs,
		NumSockets:     len(machineInfo.Topology),
		NumCores:       numPhysicalCores,
		HyperThreading: htEnabled,
		CPUtopoDetails: CPUtopoDetails,
	}, nil
}
