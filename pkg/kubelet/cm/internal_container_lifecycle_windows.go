//go:build windows
// +build windows

/*
Copyright 2020 The Kubernetes Authors.

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

package cm

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/winstats"
	"k8s.io/utils/cpuset"
)

func (i *internalContainerLifecycleImpl) PreCreateContainer(logger klog.Logger, pod *v1.Pod, container *v1.Container, containerConfig *runtimeapi.ContainerConfig) error {
	if !utilfeature.DefaultFeatureGate.Enabled(kubefeatures.WindowsCPUAndMemoryAffinity) {
		return nil
	}

	logger.V(4).Info("PreCreateContainer for Windows")

	// retrieve CPU and NUMA affinity from CPU Manager and Memory Manager (if enabled)
	var allocatedCPUs cpuset.CPUSet
	if i.cpuManager != nil {
		allocatedCPUs = i.cpuManager.GetCPUAffinity(string(pod.UID), container.Name)
	}

	var numaNodes sets.Set[int]
	if i.memoryManager != nil {
		numaNodes = i.memoryManager.GetMemoryNUMANodes(logger, pod, container)
	}

	// Gather all CPUs associated with the selected NUMA nodes
	var allNumaNodeCPUs []winstats.GroupAffinity
	for _, numaNode := range sets.List(numaNodes) {
		affinity, err := winstats.GetCPUsforNUMANode(uint16(numaNode))
		if err != nil {
			return fmt.Errorf("failed to get CPUs for NUMA node %d: %v", numaNode, err)
		}
		allNumaNodeCPUs = append(allNumaNodeCPUs, *affinity)
	}

	var finalCPUSet = computeFinalCpuSet(allocatedCPUs, allNumaNodeCPUs)

	logger.V(4).Info("Setting CPU affinity", "affinity", finalCPUSet, "container", container.Name, "pod", pod.UID)

	// Set CPU group affinities in the container config
	if finalCPUSet != nil {
		var cpusToGroupAffinities []*runtimeapi.WindowsCpuGroupAffinity
		for group, mask := range groupMasks(finalCPUSet) {

			cpusToGroupAffinities = append(cpusToGroupAffinities, &runtimeapi.WindowsCpuGroupAffinity{
				CpuGroup: uint32(group),
				CpuMask:  uint64(mask),
			})
		}
		containerConfig.Windows.Resources.AffinityCpus = cpusToGroupAffinities
	}

	// return nil if no CPUs were selected
	return nil
}

// computeFinalCpuSet determines the final set of CPUs to use based on the CPU and memory managers
// and is extracted so that it can be tested
func computeFinalCpuSet(allocatedCPUs cpuset.CPUSet, allNumaNodeCPUs []winstats.GroupAffinity) sets.Set[int] {
	if !allocatedCPUs.IsEmpty() && len(allNumaNodeCPUs) > 0 {
		// Both CPU and memory managers are enabled

		numaNodeAffinityCPUSet := computeCPUSet(allNumaNodeCPUs)
		cpuManagerAffinityCPUSet := sets.New[int](allocatedCPUs.List()...)

		// Determine which set of CPUs to use using the following logic outlined in the KEP:
		// Case 1: CPU manager selects more CPUs than those available in the NUMA nodes selected by the memory manager
		// Case 2: CPU manager selects fewer CPUs, and they all fall within the CPUs available in the NUMA nodes selected by the memory manager
		// Case 3: CPU manager selects fewer CPUs, but some are outside of the CPUs available in the NUMA nodes selected by the memory manager

		if cpuManagerAffinityCPUSet.Len() > numaNodeAffinityCPUSet.Len() {
			// Case 1, use CPU manager selected CPUs
			return cpuManagerAffinityCPUSet
		} else if numaNodeAffinityCPUSet.IsSuperset(cpuManagerAffinityCPUSet) {
			// case 2, use CPU manager selected CPUstry
			return cpuManagerAffinityCPUSet
		} else {
			// Case 3, merge CPU manager and memory manager selected CPUs
			return cpuManagerAffinityCPUSet.Union(numaNodeAffinityCPUSet)
		}
	} else if !allocatedCPUs.IsEmpty() {
		// Only CPU manager is enabled, use CPU manager selected CPUs
		return sets.New[int](allocatedCPUs.List()...)
	} else if len(allNumaNodeCPUs) > 0 {
		// Only memory manager is enabled, use CPUs associated with selected NUMA nodes
		return computeCPUSet(allNumaNodeCPUs)
	}
	return nil
}

// computeCPUSet converts a list of GroupAffinity to a set of CPU IDs
func computeCPUSet(affinities []winstats.GroupAffinity) sets.Set[int] {
	cpuSet := sets.New[int]()
	for _, affinity := range affinities {
		for i := 0; i < 64; i++ {
			if (affinity.Mask>>i)&1 == 1 {
				cpuID := int(affinity.Group)*64 + i
				cpuSet.Insert(cpuID)
			}
		}
	}
	return cpuSet
}

// groupMasks converts a set of CPU IDs into group and mask representations
func groupMasks(cpuSet sets.Set[int]) map[int]uint64 {
	groupMasks := make(map[int]uint64)
	for cpu := range cpuSet {
		group := cpu / 64
		mask := uint64(1) << (cpu % 64)
		groupMasks[group] |= mask
	}
	return groupMasks
}
