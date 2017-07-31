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
	"k8s.io/api/core/v1"
	v1qos "k8s.io/kubernetes/pkg/api/v1/helper/qos"
	"k8s.io/kubernetes/pkg/kubelet/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cpuset"
)

// PolicyStatic name of static policy
const PolicyStatic policyName = "static"

var _ Policy = &staticPolicy{}

type staticPolicy struct {
	topology *topology.CPUTopology
}

// NewStaticPolicy returns a cupset manager policy that does not change
// CPU assignments for exclusively pinned guaranteed containers after
// the main container process starts.
func NewStaticPolicy(topology *topology.CPUTopology) Policy {
	return &staticPolicy{
		topology: topology,
	}
}

func (p *staticPolicy) Name() string {
	return string(PolicyStatic)
}

func (p *staticPolicy) Start(s state.State) {
	fullCpuset := cpuset.NewCPUSet()
	for cpuid := 0; cpuid < p.topology.NumCPUs; cpuid++ {
		fullCpuset.Add(cpuid)
	}
	// takeByTopology will filter out fullCpuset returning low-number cores
	// i.e. NumReservedCores=2, then reserved={0,5} (HT enabled Case)
	reserved, _ := takeByTopology(p.topology, fullCpuset, p.topology.NumReservedCores)
	s.SetDefaultCPUSet(fullCpuset.Difference(reserved))
}

func (p *staticPolicy) RegisterContainer(s state.State, pod *v1.Pod, container *v1.Container, containerID string) error {
	glog.Infof("[cpumanager] static policy: RegisterContainer (pod: %s, container: %s, container id: %s)", pod.Name, container.Name, containerID)
	if numCPUs := guaranteedCPUs(pod, container); numCPUs != 0 {
		// container belongs in an exclusively allocated pool
		cpuset, err := p.allocateCPUs(s, numCPUs)
		if err != nil {
			glog.Errorf("[cpumanager] unable to allocate %d CPUs (container id: %s, error: %v)", numCPUs, containerID, err)
			return err
		}
		s.SetCPUSet(containerID, cpuset)
	}
	// container belongs in the shared pool (nothing to do; use default cpuset)
	return nil
}

func (p *staticPolicy) UnregisterContainer(s state.State, containerID string) error {
	glog.Infof("[cpumanager] static policy: UnregisterContainer (container id: %s)", containerID)
	if toRelease, ok := s.GetCPUSet(containerID); ok {
		s.Delete(containerID)
		p.releaseCPUs(s, toRelease)
	}
	return nil
}

func (p *staticPolicy) allocateCPUs(s state.State, numCPUs int) (cpuset.CPUSet, error) {
	glog.Infof("[cpumanager] allocateCpus: (numCPUs: %d)", numCPUs)
	if numCPUs > s.GetDefaultCPUSet().Size() {
		return cpuset.NewCPUSet(), fmt.Errorf("not enough cpus available to satisfy request")
	}
	result, err := takeByTopology(p.topology, s.GetDefaultCPUSet(), numCPUs)
	if err != nil {
		return nil, err
	}
	// Remove allocated CPUs from the shared CPUSet.
	s.SetDefaultCPUSet(s.GetDefaultCPUSet().Difference(result))

	glog.Infof("[cpumanager] allocateCPUs: returning \"%v\"", result)
	return result, nil
}

func takeByTopology(topo *topology.CPUTopology, availableCPUs cpuset.CPUSet, numCPUs int) (cpuset.CPUSet, error) {
	if numCPUs > availableCPUs.Size() {
		return cpuset.NewCPUSet(), fmt.Errorf("not enough cpus available to satisfy request")
	}

	// Algorithm: topology-aware best-fit
	result := cpuset.NewCPUSet()
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
	for s := range topoDetails.Sockets().Filter(isFullSocket) {
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
		for core := range sCores {
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
			iSocketScore := topo.CPUtopoDetails.CPUsInSocket(i).Intersection(result).Size()
			jSocketScore := topo.CPUtopoDetails.CPUsInSocket(j).Intersection(result).Size()

			// Compute the number of available CPUs on the same core as i and j.
			iCoreScore := topoDetails.CPUsInCore(topoDetails[i].CoreID).Size()
			jCoreScore := topoDetails.CPUsInCore(topoDetails[j].CoreID).Size()

			return iSocketScore > jSocketScore || iCoreScore < jCoreScore
		})

	for _, cpu := range cpuIDs {
		glog.V(4).Infof("[cpumanager] takeByTopology: claiming CPU [%d]", cpu)
		take(cpuset.NewCPUSet(cpu))
		if numCPUs < 1 {
			return result, nil
		}
	}

	return nil, fmt.Errorf("failed to allocate cpus")
}

func (p *staticPolicy) releaseCPUs(s state.State, release cpuset.CPUSet) {
	// mutate the shared pool, adding supplied cpus
	s.SetDefaultCPUSet(s.GetDefaultCPUSet().Union(release))
}

func guaranteedCPUs(pod *v1.Pod, container *v1.Container) int {
	if v1qos.GetPodQOS(pod) != v1.PodQOSGuaranteed {
		return 0
	}
	cpuQuantity := container.Resources.Requests[v1.ResourceCPU]
	glog.Infof("[cpumanager] guaranteedCpus (container: %s, cpu request: %v)", container.Name, cpuQuantity.MilliValue())
	if cpuQuantity.Value()*1000 != cpuQuantity.MilliValue() {
		return 0
	}
	// Safe downcast to do for all systems with < 2.1 billion CPUs.
	// Per the language spec, `int` is guaranteed to be at least 32 bits wide.
	// https://golang.org/ref/spec#Numeric_types
	return int(cpuQuantity.Value())
}
