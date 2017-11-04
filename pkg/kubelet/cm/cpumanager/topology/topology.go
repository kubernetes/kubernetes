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
	"sort"

	"github.com/golang/glog"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
)

// CPUDetails is a map from CPU ID to Core ID and Socket ID.
type CPUDetails map[int]CPUInfo

// CPUTopology contains details of node cpu, where :
// CPU  - logical CPU, cadvisor - thread
// Core - physical CPU, cadvisor - Core
// Socket - socket, cadvisor - Node
type CPUTopology struct {
	NumCPUs    int
	NumCores   int
	NumSockets int
	CPUDetails CPUDetails
}

// CPUsPerCore returns the number of logical CPUs are associated with
// each core.
func (topo *CPUTopology) CPUsPerCore() int {
	if topo.NumCores == 0 {
		return 0
	}
	return topo.NumCPUs / topo.NumCores
}

// CPUsPerSocket returns the number of logical CPUs are associated with
// each socket.
func (topo *CPUTopology) CPUsPerSocket() int {
	if topo.NumSockets == 0 {
		return 0
	}
	return topo.NumCPUs / topo.NumSockets
}

// CPUInfo contains the socket and core IDs associated with a CPU.
type CPUInfo struct {
	SocketID int
	CoreID   int
}

// KeepOnly returns a new CPUDetails object with only the supplied cpus.
func (d CPUDetails) KeepOnly(cpus cpuset.CPUSet) CPUDetails {
	result := CPUDetails{}
	for cpu, info := range d {
		if cpus.Contains(cpu) {
			result[cpu] = info
		}
	}
	return result
}

// Sockets returns all of the socket IDs associated with the CPUs in this
// CPUDetails.
func (d CPUDetails) Sockets() cpuset.CPUSet {
	b := cpuset.NewBuilder()
	for _, info := range d {
		b.Add(info.SocketID)
	}
	return b.Result()
}

// CPUsInSocket returns all of the logical CPU IDs associated with the
// given socket ID in this CPUDetails.
func (d CPUDetails) CPUsInSocket(id int) cpuset.CPUSet {
	b := cpuset.NewBuilder()
	for cpu, info := range d {
		if info.SocketID == id {
			b.Add(cpu)
		}
	}
	return b.Result()
}

// Cores returns all of the core IDs associated with the CPUs in this
// CPUDetails.
func (d CPUDetails) Cores() cpuset.CPUSet {
	b := cpuset.NewBuilder()
	for _, info := range d {
		b.Add(info.CoreID)
	}
	return b.Result()
}

// CoresInSocket returns all of the core IDs associated with the given
// socket ID in this CPUDetails.
func (d CPUDetails) CoresInSocket(id int) cpuset.CPUSet {
	b := cpuset.NewBuilder()
	for _, info := range d {
		if info.SocketID == id {
			b.Add(info.CoreID)
		}
	}
	return b.Result()
}

// CPUs returns all of the logical CPU IDs in this CPUDetails.
func (d CPUDetails) CPUs() cpuset.CPUSet {
	b := cpuset.NewBuilder()
	for cpuID := range d {
		b.Add(cpuID)
	}
	return b.Result()
}

// CPUsInCore returns all of the logical CPU IDs associated with the
// given core ID in this CPUDetails.
func (d CPUDetails) CPUsInCore(id int) cpuset.CPUSet {
	b := cpuset.NewBuilder()
	for cpu, info := range d {
		if info.CoreID == id {
			b.Add(cpu)
		}
	}
	return b.Result()
}

// Discover returns CPUTopology based on cadvisor node info
func Discover(machineInfo *cadvisorapi.MachineInfo) (*CPUTopology, error) {

	if machineInfo.NumCores == 0 {
		return nil, fmt.Errorf("could not detect number of cpus")
	}

	CPUDetails := CPUDetails{}
	numCPUs := machineInfo.NumCores
	numPhysicalCores := 0
	var coreID int
	var err error

	for _, socket := range machineInfo.Topology {
		numPhysicalCores += len(socket.Cores)
		for _, core := range socket.Cores {
			if coreID, err = getUniqueCoreID(core.Threads); err != nil {
				glog.Errorf("could not get unique coreID for socket: %d core %d threads: %v",
					socket.Id, core.Id, core.Threads)
				return nil, err
			}
			for _, cpu := range core.Threads {
				CPUDetails[cpu] = CPUInfo{
					CoreID:   coreID,
					SocketID: socket.Id,
				}
			}
		}
	}

	return &CPUTopology{
		NumCPUs:    numCPUs,
		NumSockets: len(machineInfo.Topology),
		NumCores:   numPhysicalCores,
		CPUDetails: CPUDetails,
	}, nil
}

// getUniqueCoreID computes coreId as the lowest cpuID
// for a given Threads []int slice. This will assure that coreID's are
// platform unique (opposite to what cAdvisor reports - socket unique)
func getUniqueCoreID(threads []int) (coreID int, err error) {
	err = nil
	if len(threads) == 0 {
		return 0, fmt.Errorf("no cpus provided")
	}

	if len(threads) != cpuset.NewCPUSet(threads...).Size() {
		return 0, fmt.Errorf("cpus provided are not unique")
	}

	tmpThreads := make([]int, len(threads))
	copy(tmpThreads, threads)
	sort.Ints(tmpThreads)
	return tmpThreads[0], err
}
