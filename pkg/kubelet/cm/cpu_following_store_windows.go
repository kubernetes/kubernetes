//go:build windows

/*
Copyright The Kubernetes Authors.

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
	cadvisorapi "github.com/google/cadvisor/info/v1"

	"k8s.io/utils/cpuset"

	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
)

// This file implements the Windows-only "memory follows CPU" mechanism for NUMA
// placement.
//
// On Windows there is no cpuset.mems: memory cannot be pinned to a NUMA node, and
// the kernel serves a thread's pages from the NUMA node of the CPU it runs on. CPU
// affinity is therefore the only NUMA lever that is actually enforced, and it is
// owned by the CPU Manager. To keep the Memory Manager's per-NUMA bookkeeping
// consistent with what the OS actually does — and to avoid the CPU-affinity union
// in computeFinalCpuSet — the Memory Manager must mirror the CPU Manager's NUMA
// decision rather than choose its own nodes independently.
//
// cpuFollowingStore (below) is that mechanism: a Topology Manager Store wrapper,
// injected as the Memory Manager's affinity store in container_manager_windows.go.
// It makes the Memory Manager read back the NUMA nodes of a container's exclusive
// CPUs, and reports (via HasExclusiveCPUs) whether there is a CPU decision to
// follow at all — when there is not (CPU Manager policy "none", or a shared /
// non-Guaranteed container), the Memory Manager falls back to its own calculation.

// Ordering guarantee: the CPU Manager is registered as a hint provider before
// the Memory Manager (see NewContainerManager), so the CPU Manager's Allocate has
// already run and committed its exclusive CPUs by the time the Memory Manager
// calls GetAffinity.
type cpuFollowingStore struct {
	// Store is the wrapped Topology Manager; GetPolicy and Name are promoted from
	// it, and GetAffinity falls back to it when there is nothing to follow.
	topologymanager.Store
	cpuManager cpumanager.Manager
	// cpuToNode maps a logical CPU id to the NUMA node id that contains it, built
	// once from the machine topology (the same mapping the CPU Manager uses).
	cpuToNode map[int]int
}

var _ topologymanager.Store = &cpuFollowingStore{}

// newCPUFollowingStore builds the wrapper from the base Topology Manager store,
// the CPU Manager, and the machine topology.
func newCPUFollowingStore(base topologymanager.Store, cpuManager cpumanager.Manager, machineInfo *cadvisorapi.MachineInfo) *cpuFollowingStore {
	cpuToNode := make(map[int]int)
	for _, node := range machineInfo.Topology {
		for _, core := range node.Cores {
			for _, cpu := range core.Threads {
				cpuToNode[cpu] = node.Id
			}
		}
	}
	return &cpuFollowingStore{
		Store:      base,
		cpuManager: cpuManager,
		cpuToNode:  cpuToNode,
	}
}

// GetAffinity returns the NUMA affinity the Memory Manager should use for the
// given container: the set of NUMA nodes owning the container's exclusive CPUs.
// If the container has no exclusive CPUs (shared pool / non-Guaranteed) or the
// CPUs cannot be mapped to NUMA nodes, it defers to the wrapped store.
func (s *cpuFollowingStore) GetAffinity(podUID string, containerName string) topologymanager.TopologyHint {
	base := s.Store.GetAffinity(podUID, containerName)

	exclusiveCPUs := s.cpuManager.GetExclusiveCPUs(podUID, containerName)
	if exclusiveCPUs.IsEmpty() {
		// Nothing to follow: this container did not get exclusive CPUs, so let the
		// Topology Manager's own hint stand.
		return base
	}

	mask := s.numaMaskForCPUs(exclusiveCPUs)
	if mask == nil || mask.IsEmpty() {
		return base
	}

	return topologymanager.TopologyHint{
		NUMANodeAffinity: mask,
		Preferred:        base.Preferred,
	}
}

// HasExclusiveCPUs reports whether the CPU manager assigned the given container
// any exclusive CPUs. The memory manager uses this to decide whether to follow
// the CPU manager's NUMA decision (exclusive CPUs present → stay synced, do not
// extend) or to do its own calculation (none present — e.g. CPU manager policy
// "none", or a shared/non-Guaranteed container).
func (s *cpuFollowingStore) HasExclusiveCPUs(podUID, containerName string) bool {
	return !s.cpuManager.GetExclusiveCPUs(podUID, containerName).IsEmpty()
}

// numaMaskForCPUs returns the set of NUMA nodes that contain the given CPUs.
func (s *cpuFollowingStore) numaMaskForCPUs(cpus cpuset.CPUSet) bitmask.BitMask {
	nodeSet := make(map[int]struct{})
	for _, cpu := range cpus.List() {
		if node, ok := s.cpuToNode[cpu]; ok {
			nodeSet[node] = struct{}{}
		}
	}
	if len(nodeSet) == 0 {
		return nil
	}
	nodes := make([]int, 0, len(nodeSet))
	for node := range nodeSet {
		nodes = append(nodes, node)
	}
	mask, err := bitmask.NewBitMask(nodes...)
	if err != nil {
		return nil
	}
	return mask
}
