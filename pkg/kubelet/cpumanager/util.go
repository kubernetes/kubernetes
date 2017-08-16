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

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/kubelet/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cpuset"
)

func takeByTopology(topo *topology.CPUTopology, availableCPUs cpuset.CPUSet, numCPUs int) (cpuset.CPUSet, error) {
	result := cpuset.NewCPUSet()

	if numCPUs == 0 {
		return result, nil
	}
	if numCPUs > availableCPUs.Size() {
		return nil, fmt.Errorf("not enough cpus available to satisfy request")
	}

	// Algorithm: topology-aware best-fit
	CPUsPerCore := topo.NumCPUs / topo.NumCores
	CPUsPerSocket := topo.NumCPUs / topo.NumSockets
	topoDetails := topo.CPUtopoDetails.KeepOnly(availableCPUs)

	// Auxilliary closure to update intermediate results:
	// - Adds cpus to result
	// - Recalculates availableCPUs
	// - Prunes topoDetails
	// - decrements numCPUs
	take := func(cpus cpuset.CPUSet) {
		result = result.Union(cpus)
		availableCPUs = availableCPUs.Difference(result)
		topoDetails = topoDetails.KeepOnly(availableCPUs)
		numCPUs -= cpus.Size()
	}

	// Returns true if the supplied socket is fully available in
	// `topoDetails`.
	isFullSocket := func(socketID int) bool {
		return topoDetails.CPUsInSocket(socketID).Size() == CPUsPerSocket
	}

	// Returns true if the supplied core is fully available in
	// `topoDetails`.
	isFullCore := func(coreID int) bool {
		return topoDetails.CPUsInCore(coreID).Size() == CPUsPerCore
	}

	// 1. Acquire whole sockets, if available and the container requires
	//    at least a socket's-worth of CPUs.
	for _, s := range topoDetails.Sockets().Filter(isFullSocket).AsSlice() {
		if numCPUs >= CPUsPerSocket {
			glog.V(4).Infof("[cpumanager] takeByTopology: claiming socket [%d]", s)
			take(topoDetails.CPUsInSocket(s))
			if numCPUs < 1 {
				return result, nil
			}
		}
	}

	// 2. Acquire whole cores, if available and the container requires
	//    at least a core's-worth of CPUs.

	// `socketIDs` are sorted by:
	// - the number of whole available cores, ascending.
	socketIDs := topoDetails.Sockets().AsSlice()
	sort.Slice(socketIDs,
		func(i, j int) bool {
			iCores := topoDetails.CoresInSocket(socketIDs[i]).Filter(isFullCore)
			jCores := topoDetails.CoresInSocket(socketIDs[j]).Filter(isFullCore)
			return iCores.Size() < jCores.Size()
		})

	for _, s := range socketIDs {
		sCores := topoDetails.CoresInSocket(s).Filter(isFullCore)
		for _, core := range sCores.AsSlice() {
			if numCPUs >= CPUsPerCore {
				glog.V(4).Infof("[cpumanager] takeByTopology: claiming core [%d]", core)
				take(topoDetails.CPUsInCore(core))
				if numCPUs < 1 {
					return result, nil
				}
			}
		}
	}

	// 3. Acquire single threads, preferring to fill partially-allocated cores
	//    on the same sockets as the whole cores we have already taken.

	// `cpuIDs` are sorted by:
	// - the number of available CPUs on the same core, ascending
	// - the number of already assigned CPUs for this allocation on the
	//   same socket, descending
	cpuIDs := availableCPUs.AsSlice()
	sort.Slice(cpuIDs,
		func(i, j int) bool {
			// Compute the number of CPUs on the same socket as i and j in
			// the result.
			iSocket := topo.CPUtopoDetails[cpuIDs[i]].SocketID
			jSocket := topo.CPUtopoDetails[cpuIDs[j]].SocketID
			iSocketScore := topo.CPUtopoDetails.CPUsInSocket(iSocket).Intersection(result).Size()
			jSocketScore := topo.CPUtopoDetails.CPUsInSocket(jSocket).Intersection(result).Size()

			// Compute the number of available CPUs on the same core as i and j.
			iCoreScore := topoDetails.CPUsInCore(topoDetails[cpuIDs[i]].CoreID).Size()
			jCoreScore := topoDetails.CPUsInCore(topoDetails[cpuIDs[j]].CoreID).Size()

			return iSocketScore > jSocketScore || iCoreScore < jCoreScore || i < j
		})

	for _, cpu := range cpuIDs {
		glog.V(4).Infof("[cpumanager] takeByTopology: claiming CPU [%d]", cpu)
		if numCPUs > 0 {
			take(cpuset.NewCPUSet(cpu))
		}
		if numCPUs < 1 {
			return result, nil
		}
	}

	return nil, fmt.Errorf("failed to allocate cpus")
}
