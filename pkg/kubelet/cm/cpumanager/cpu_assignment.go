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

package cpumanager

import (
	"fmt"
	"sort"

	"k8s.io/klog/v2"

	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
)

type cpuAccumulator struct {
	topo          *topology.CPUTopology
	details       topology.CPUDetails
	numCPUsNeeded int
	result        cpuset.CPUSet
}

func newCPUAccumulator(topo *topology.CPUTopology, availableCPUs cpuset.CPUSet, numCPUs int) *cpuAccumulator {
	return &cpuAccumulator{
		topo:          topo,
		details:       topo.CPUDetails.KeepOnly(availableCPUs),
		numCPUsNeeded: numCPUs,
		result:        cpuset.NewCPUSet(),
	}
}

// Returns true if the supplied socket is fully available in `topoDetails`.
func (a *cpuAccumulator) isSocketFree(socketID int) bool {
	return a.details.CPUsInSockets(socketID).Size() == a.topo.CPUsPerSocket()
}

// Returns true if the supplied core is fully available in `topoDetails`.
func (a *cpuAccumulator) isCoreFree(coreID int) bool {
	return a.details.CPUsInCores(coreID).Size() == a.topo.CPUsPerCore()
}

// Returns free socket IDs as a slice sorted by sortAvailableSockets().
func (a *cpuAccumulator) freeSockets() []int {
	free := []int{}
	for _, socket := range a.sortAvailableSockets() {
		if a.isSocketFree(socket) {
			free = append(free, socket)
		}
	}
	return free
}

// Returns free core IDs as a slice sorted by sortAvailableCores().
func (a *cpuAccumulator) freeCores() []int {
	free := []int{}
	for _, core := range a.sortAvailableCores() {
		if a.isCoreFree(core) {
			free = append(free, core)
		}
	}
	return free
}

// Returns free CPU IDs as a slice sorted by sortAvailableCPUs().
func (a *cpuAccumulator) freeCPUs() []int {
	return a.sortAvailableCPUs()
}

// Sorts the provided list of sockets/cores/cpus referenced in 'ids' by the
// number of available CPUs contained within them (smallest to largest). The
// 'getCPU()' paramater defines the function that should be called to retrieve
// the list of available CPUs for the type of socket/core/cpu being referenced.
// If two sockets/cores/cpus have the same number of available CPUs, they are
// sorted in ascending order by their id.
func (a *cpuAccumulator) sort(ids []int, getCPUs func(ids ...int) cpuset.CPUSet) {
	sort.Slice(ids,
		func(i, j int) bool {
			iCPUs := getCPUs(ids[i])
			jCPUs := getCPUs(ids[j])
			if iCPUs.Size() < jCPUs.Size() {
				return true
			}
			if iCPUs.Size() > jCPUs.Size() {
				return false
			}
			return ids[i] < ids[j]
		})
}

// Sort all sockets with free CPUs using the sort() algorithm defined above.
func (a *cpuAccumulator) sortAvailableSockets() []int {
	sockets := a.details.Sockets().ToSliceNoSort()
	a.sort(sockets, a.details.CPUsInSockets)
	return sockets
}

// Sort all cores with free CPUs:
// - First by socket using sortAvailableSockets().
// - Then within each socket, using the sort() algorithm defined above.
func (a *cpuAccumulator) sortAvailableCores() []int {
	var result []int
	for _, socket := range a.sortAvailableSockets() {
		cores := a.details.CoresInSockets(socket).ToSliceNoSort()
		a.sort(cores, a.details.CPUsInCores)
		result = append(result, cores...)
	}
	return result
}

// Sort all available CPUs:
// - First by core using sortAvailableCores().
// - Then within each core, using the sort() algorithm defined above.
func (a *cpuAccumulator) sortAvailableCPUs() []int {
	var result []int
	for _, core := range a.sortAvailableCores() {
		cpus := a.details.CPUsInCores(core).ToSliceNoSort()
		sort.Ints(cpus)
		result = append(result, cpus...)
	}
	return result
}

func (a *cpuAccumulator) take(cpus cpuset.CPUSet) {
	a.result = a.result.Union(cpus)
	a.details = a.details.KeepOnly(a.details.CPUs().Difference(a.result))
	a.numCPUsNeeded -= cpus.Size()
}

func (a *cpuAccumulator) takeFullSockets() {
	for _, socket := range a.freeSockets() {
		cpusInSocket := a.topo.CPUDetails.CPUsInSockets(socket)
		if !a.needs(cpusInSocket.Size()) {
			continue
		}
		klog.V(4).InfoS("takeFullSockets: claiming socket", "socket", socket)
		a.take(cpusInSocket)
	}
}

func (a *cpuAccumulator) takeFullCores() {
	for _, core := range a.freeCores() {
		cpusInCore := a.topo.CPUDetails.CPUsInCores(core)
		if !a.needs(cpusInCore.Size()) {
			continue
		}
		klog.V(4).InfoS("takeFullCores: claiming core", "core", core)
		a.take(cpusInCore)
	}
}

func (a *cpuAccumulator) takeRemainingCPUs() {
	for _, cpu := range a.sortAvailableCPUs() {
		klog.V(4).InfoS("takeRemainingCPUs: claiming CPU", "cpu", cpu)
		a.take(cpuset.NewCPUSet(cpu))
		if a.isSatisfied() {
			return
		}
	}
}

func (a *cpuAccumulator) needs(n int) bool {
	return a.numCPUsNeeded >= n
}

func (a *cpuAccumulator) isSatisfied() bool {
	return a.numCPUsNeeded < 1
}

func (a *cpuAccumulator) isFailed() bool {
	return a.numCPUsNeeded > a.details.CPUs().Size()
}

func takeByTopology(topo *topology.CPUTopology, availableCPUs cpuset.CPUSet, numCPUs int) (cpuset.CPUSet, error) {
	acc := newCPUAccumulator(topo, availableCPUs, numCPUs)
	if acc.isSatisfied() {
		return acc.result, nil
	}
	if acc.isFailed() {
		return cpuset.NewCPUSet(), fmt.Errorf("not enough cpus available to satisfy request")
	}

	// Algorithm: topology-aware best-fit
	// 1. Acquire whole sockets, if available and the container requires at
	//    least a socket's-worth of CPUs.
	acc.takeFullSockets()
	if acc.isSatisfied() {
		return acc.result, nil
	}

	// 2. Acquire whole cores, if available and the container requires at least
	//    a core's-worth of CPUs.
	acc.takeFullCores()
	if acc.isSatisfied() {
		return acc.result, nil
	}

	// 3. Acquire single threads, preferring to fill partially-allocated cores
	//    on the same sockets as the whole cores we have already taken in this
	//    allocation.
	acc.takeRemainingCPUs()
	if acc.isSatisfied() {
		return acc.result, nil
	}

	return cpuset.NewCPUSet(), fmt.Errorf("failed to allocate cpus")
}
