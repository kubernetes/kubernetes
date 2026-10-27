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
	"math"

	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/utils/cpuset"
)

// ## Resize Operations
//
// For resize operations, the goal is to allocate new CPUs as close as possible
// to the retained CPUs in the topology hierarchy (same NUMA node, same socket, and same core).
// When sorting topology elements (NUMA nodes, sockets, or cores), prioritize elements that already
// have CPUs allocated to this container (retained CPUs), followed by remaining available elements.
// This minimizes topology changes and maintains performance characteristics.
//
// Pod add can be viewed as a special case of Pod resize where the retained CPUs are empty.
// However, the resize functions are kept separate from the Pod add flow to
// eliminate any impact of the resize changes on it.
//
// TODO: For beta, refactor and unify the add/resize code paths. (Tracking issue: https://github.com/kubernetes/kubernetes/issues/143218)

type numaFirstForResize struct{ acc *cpuAccumulatorForResize }
type socketsFirstForResize struct{ acc *cpuAccumulatorForResize }

var _ numaOrSocketsFirstFuncs = (*numaFirstForResize)(nil)
var _ numaOrSocketsFirstFuncs = (*socketsFirstForResize)(nil)

// Sort the UncoreCaches within NUMA nodes for resize.
// For resize operation, this function sorts UncoreCaches in a specific order to maintain topology continuity.
// Within each NUMA node (already sorted by sortAvailableNUMANodes):
// 1. First, UncoreCaches with allocated CPUs for this container (retained CPUs)
// 2. Then, other available UncoreCaches
func (a *cpuAccumulatorForResize) sortAvailableUncoreCaches() []int {
	var result []int
	added := cpuset.New()
	allocatedUncoreCachesSet := a.resultDetails.UncoreCaches()
	for _, numa := range a.sortAvailableNUMANodes() {
		availableUncoreCachesSet := a.details.UncoreInNUMANodes(numa)
		// UncoreCache can span NUMA nodes, so skip already added
		availableUncoreCachesSet = availableUncoreCachesSet.Difference(added)
		added = added.Union(availableUncoreCachesSet)

		// Sort UncoreCaches that have allocated CPUs to this container.
		allocatedUncoreCaches := allocatedUncoreCachesSet.Intersection(availableUncoreCachesSet).UnsortedList()
		a.sort(allocatedUncoreCaches, a.details.CPUsInUncoreCaches)
		result = append(result, allocatedUncoreCaches...)

		// Sort other available UncoreCaches that don't have CPUs allocated to this container yet.
		availableUncoreCaches := availableUncoreCachesSet.Difference(allocatedUncoreCachesSet).UnsortedList()
		a.sort(availableUncoreCaches, a.details.CPUsInUncoreCaches)
		result = append(result, availableUncoreCaches...)
	}
	return result
}

// If NUMA nodes are higher in the memory hierarchy than sockets, then we take
// from the set of NUMA Nodes as the first level for resize.
func (n *numaFirstForResize) takeFullFirstLevel() {
	n.acc.takeFullNUMANodes()
}

// If NUMA nodes are higher in the memory hierarchy than sockets, then we take
// from the set of sockets as the second level for resize.
func (n *numaFirstForResize) takeFullSecondLevel() {
	n.acc.takeFullSockets()
}

// Sort available NUMA nodes for resize when NUMA nodes are higher than sockets in the memory hierarchy.
// For resize operation, this function sorts NUMA nodes in a specific order to maintain topology continuity.
// 1. First, NUMA nodes with allocated CPUs for this container (retained CPUs)
// 2. Then, other available NUMA nodes
func (n *numaFirstForResize) sortAvailableNUMANodes() []int {
	var result []int

	allocatedNumaNodesSet := n.acc.resultDetails.NUMANodes()
	availableNumaNodesSet := n.acc.details.NUMANodes()

	// Sort Numa nodes which have allocated CPUs to this container.
	allocatedNumas := allocatedNumaNodesSet.Intersection(availableNumaNodesSet).UnsortedList()
	n.acc.sort(allocatedNumas, n.acc.details.CPUsInNUMANodes)
	result = append(result, allocatedNumas...)

	// Sort other Numa nodes that don't have CPUs allocated to this container yet.
	availableNumas := availableNumaNodesSet.Difference(allocatedNumaNodesSet).UnsortedList()
	n.acc.sort(availableNumas, n.acc.details.CPUsInNUMANodes)
	result = append(result, availableNumas...)
	return result
}

// Sort available sockets for resize when NUMA nodes are higher than sockets in the memory hierarchy.
// For resize operation, this function sorts sockets in a specific order to maintain topology continuity.
// Within each NUMA node (already sorted by sortAvailableNUMANodes):
// 1. First, sockets with allocated CPUs for this container (retained CPUs)
// 2. Then, other available sockets
func (n *numaFirstForResize) sortAvailableSockets() []int {
	var result []int

	for _, numa := range n.sortAvailableNUMANodes() {
		allocatedSocketsSet := n.acc.resultDetails.SocketsInNUMANodes(numa)
		availableSocketsSet := n.acc.details.SocketsInNUMANodes(numa)

		// Sort sockets that have allocated CPUs to this container.
		allocatedSockets := allocatedSocketsSet.Intersection(availableSocketsSet).UnsortedList()
		n.acc.sort(allocatedSockets, n.acc.details.CPUsInSockets)
		result = append(result, allocatedSockets...)

		// Sort other available sockets that don't have CPUs allocated to this container yet.
		availableSockets := availableSocketsSet.Difference(allocatedSocketsSet).UnsortedList()
		n.acc.sort(availableSockets, n.acc.details.CPUsInSockets)
		result = append(result, availableSockets...)
	}
	return result
}

// Sort available cores for resize when NUMA nodes are higher than sockets in the memory hierarchy.
// For resize operation, this function sorts cores in a specific order to maintain topology continuity.
// Within each socket (already sorted by sortAvailableSockets):
// 1. First, cores with allocated CPUs for this container (retained CPUs)
// 2. Then, other available cores
func (n *numaFirstForResize) sortAvailableCores() []int {
	var result []int

	for _, socket := range n.acc.sortAvailableSockets() {
		allocatedCoresSet := n.acc.resultDetails.CoresInSockets(socket)
		availableCoresSet := n.acc.details.CoresInSockets(socket)

		// Sort cores that have allocated CPUs to this container.
		allocatedCores := allocatedCoresSet.Intersection(availableCoresSet).UnsortedList()
		n.acc.sort(allocatedCores, n.acc.details.CPUsInCores)
		result = append(result, allocatedCores...)

		// Sort other available cores that don't have CPUs allocated to this container yet.
		availableCores := availableCoresSet.Difference(allocatedCoresSet).UnsortedList()
		n.acc.sort(availableCores, n.acc.details.CPUsInCores)
		result = append(result, availableCores...)
	}
	return result
}

// If sockets are higher in the memory hierarchy than NUMA nodes, then we take
// from the set of NUMA Nodes as the first level for resize.
func (s *socketsFirstForResize) takeFullFirstLevel() {
	s.acc.takeFullSockets()
}

// If sockets are higher in the memory hierarchy than NUMA nodes, then we take
// from the set of sockets as the second level for resize.
func (s *socketsFirstForResize) takeFullSecondLevel() {
	s.acc.takeFullNUMANodes()
}

// Sort available NUMA nodes for resize when sockets are higher than NUMA nodes in the memory hierarchy.
// For resize operation, this function sorts NUMA nodes in a specific order to maintain topology continuity.
// Within each socket (already sorted by sortAvailableSockets):
// 1. First, NUMA nodes with allocated CPUs for this container (retained CPUs)
// 2. Then, other available NUMA nodes
func (s *socketsFirstForResize) sortAvailableNUMANodes() []int {
	var result []int

	for _, socket := range s.sortAvailableSockets() {
		allocatedNumaNodesSet := s.acc.resultDetails.NUMANodesInSockets(socket)
		availableNumaNodesSet := s.acc.details.NUMANodesInSockets(socket)

		// Sort Numa nodes which have allocated CPUs to this container.
		allocatedNumas := allocatedNumaNodesSet.Intersection(availableNumaNodesSet).UnsortedList()
		s.acc.sort(allocatedNumas, s.acc.details.CPUsInNUMANodes)
		result = append(result, allocatedNumas...)

		// Sort other Numa nodes that don't have CPUs allocated to this container yet.
		availableNumas := availableNumaNodesSet.Difference(allocatedNumaNodesSet).UnsortedList()
		s.acc.sort(availableNumas, s.acc.details.CPUsInNUMANodes)
		result = append(result, availableNumas...)
	}
	return result
}

// Sort available sockets for resize when sockets are higher than NUMA nodes in the memory hierarchy.
// For resize operation, this function sorts sockets in a specific order to maintain topology continuity.
// 1. First, sockets with allocated CPUs for this container (retained CPUs)
// 2. Then, other available sockets
func (s *socketsFirstForResize) sortAvailableSockets() []int {
	var result []int

	allocatedSocketsSet := s.acc.resultDetails.Sockets()
	availableSocketsSet := s.acc.details.Sockets()

	// Sort Sockets which have allocated CPUs to this container.
	allocatedSockets := allocatedSocketsSet.Intersection(availableSocketsSet).UnsortedList()
	s.acc.sort(allocatedSockets, s.acc.details.CPUsInSockets)
	result = append(result, allocatedSockets...)

	// Sort other Sockets that don't have CPUs allocated to this container yet.
	availableSockets := availableSocketsSet.Difference(allocatedSocketsSet).UnsortedList()
	s.acc.sort(availableSockets, s.acc.details.CPUsInSockets)
	result = append(result, availableSockets...)

	return result
}

// Sort available cores for resize when sockets are higher than NUMA nodes in the memory hierarchy.
// For resize operation, this function sorts cores in a specific order to maintain topology continuity.
// Within each NUMA node (already sorted by sortAvailableNUMANodes):
// 1. First, cores with allocated CPUs for this container (retained CPUs)
// 2. Then, other available cores
func (s *socketsFirstForResize) sortAvailableCores() []int {
	var result []int

	for _, numa := range s.acc.sortAvailableNUMANodes() {
		allocatedCoresSet := s.acc.resultDetails.CoresInNUMANodes(numa)
		availableCoresSet := s.acc.details.CoresInNUMANodes(numa)

		// Sort cores that have allocated CPUs to this container.
		allocatedCores := allocatedCoresSet.Intersection(availableCoresSet).UnsortedList()
		s.acc.sort(allocatedCores, s.acc.details.CPUsInCores)
		result = append(result, allocatedCores...)

		// Sort other available cores that don't have CPUs allocated to this container yet.
		availableCores := availableCoresSet.Difference(allocatedCoresSet).UnsortedList()
		s.acc.sort(availableCores, s.acc.details.CPUsInCores)
		result = append(result, availableCores...)
	}
	return result
}

type sortCPUsPackedForResize struct{ acc *cpuAccumulatorForResize }
type sortCPUsSpreadForResize struct{ acc *cpuAccumulatorForResize }

var _ availableCPUSorter = (*sortCPUsPackedForResize)(nil)
var _ availableCPUSorter = (*sortCPUsSpreadForResize)(nil)

func (s sortCPUsPackedForResize) sort() []int {
	return s.acc.sortAvailableCPUsPacked()
}

func (s sortCPUsSpreadForResize) sort() []int {
	return s.acc.sortAvailableCPUsSpread()
}

// cpuAccumulatorForResize shares allocation state with cpuAccumulator while
// keeping retained CPU ordering and taking rules on a separate method set.
type cpuAccumulatorForResize struct {
	*cpuAccumulator
	// `resultDetails` is a map from CPU ID to Core ID, Socket ID, and NUMA ID, for the set of CPUs
	// that have been accumulated so far.
	resultDetails topology.CPUDetails
}

// newCPUAccumulatorForResize creates an accumulator for CPU resize operations.
//
// Resizing operations involve two partitions (see https://github.com/kubernetes/kubernetes/pull/140629#discussion_r3595055804):
// - Partition A: the initial priority seed - all allocations MUST include this
// - Partition B: The grow pool
//
// For scale-up and scale-down, the partitions are determined differently:
// - Partition A (retained CPUs):
//   - For scale-up: currentlyAllocatedCPUs
//   - For scale-down: baselineCPUs
//
// - Partition B (availableCPUs, computed before calling this function):
//   - For scale-up: availableCPUs should be the free CPU pool (CPUs not allocated to any container)
//   - For scale-down: availableCPUs should be the currentlyAllocatedCPUs (CPUs allocated to this container)
//
// Preconditions (enforced by the caller):
//   - baselineCPUs ⊆ currentlyAllocatedCPUs (baselineCPUs must be a subset of currentlyAllocatedCPUs)
//   - baselineCPUs.Size() <= numCPUs (the number of CPUs to retain cannot exceed the requested total)
//
// TODO: In the beta stage, consider refactoring so that Partition A is also determined at the caller level.
// This way, this function would only need to know about Partition A and Partition B, without needing
// to distinguish between scale-up and scale-down scenarios. (Tracking issue: https://github.com/kubernetes/kubernetes/issues/143218)
func newCPUAccumulatorForResize(logger klog.Logger, topo *topology.CPUTopology, availableCPUs cpuset.CPUSet, numCPUs int, cpuSortingStrategy CPUSortingStrategy, currentlyAllocatedCPUs cpuset.CPUSet, baselineCPUs cpuset.CPUSet) *cpuAccumulatorForResize {
	acc := &cpuAccumulatorForResize{cpuAccumulator: &cpuAccumulator{
		logger:        logger,
		topo:          topo,
		details:       topo.CPUDetails.KeepOnly(availableCPUs),
		numCPUsNeeded: numCPUs,
		result:        cpuset.New(),
	}, resultDetails: topology.CPUDetails{}}

	if !currentlyAllocatedCPUs.IsEmpty() {
		// Increase of CPU resources ( scale up )
		// Take existing from currently allocated
		// CPUs
		if numCPUs > currentlyAllocatedCPUs.Size() {
			acc.take(currentlyAllocatedCPUs.Clone())
		}

		// Decrease of CPU resources ( scale down )
		// Take existing container-level CPU assignments recorded
		// at admission time, prior to any resize, if those exist.
		if numCPUs < currentlyAllocatedCPUs.Size() {
			if !baselineCPUs.IsEmpty() {
				// If explicitly CPUs to keep
				// during scale down is given ( this requires
				// addition in container[].resources ... which
				// could be possible to patch ? Esotsal Note This means
				// modifying API code
				acc.take(baselineCPUs.Clone())
			}
		}

		if numCPUs == currentlyAllocatedCPUs.Size() {
			// nothing to do return as is
			acc.take(currentlyAllocatedCPUs.Clone())
			return acc
		}
	}

	if topo.NumSockets >= topo.NumNUMANodes {
		acc.numaOrSocketsFirst = &numaFirstForResize{acc}
	} else {
		acc.numaOrSocketsFirst = &socketsFirstForResize{acc}
	}

	if cpuSortingStrategy == CPUSortingStrategyPacked {
		acc.availableCPUSorter = &sortCPUsPackedForResize{acc}
	} else {
		acc.availableCPUSorter = &sortCPUsSpreadForResize{acc}
	}

	return acc
}

// Returns true if this NUMA node can be fully claimed by this Container (no CPUs in this NUMA node allocated to other containers)
func (a *cpuAccumulatorForResize) isFullNUMANode(numaID int) bool {
	return a.resultDetails.CPUsInNUMANodes(numaID).Size()+a.details.CPUsInNUMANodes(numaID).Size() == a.topo.CPUDetails.CPUsInNUMANodes(numaID).Size()
}

// Returns true if this socket can be fully claimed by this Container (no CPUs in this socket are allocated to other containers).
func (a *cpuAccumulatorForResize) isFullSocket(socketID int) bool {
	return a.resultDetails.CPUsInSockets(socketID).Size()+a.details.CPUsInSockets(socketID).Size() == a.topo.CPUDetails.CPUsInSockets(socketID).Size()
}

// Returns true if this Core can be fully claimed by this Container (no CPUs in this Core are allocated to other containers).
func (a *cpuAccumulatorForResize) isFullCore(coreID int) bool {
	return a.resultDetails.CPUsInCores(coreID).Size()+a.details.CPUsInCores(coreID).Size() == a.topo.CPUDetails.CPUsInCores(coreID).Size()
}

// Returns true if this UncoreCache can be fully claimed by this Container (no CPUs in this UncoreCache are allocated to other containers).
func (a *cpuAccumulatorForResize) isFullUncoreCache(uncoreID int) bool {
	return a.resultDetails.CPUsInUncoreCaches(uncoreID).Size()+a.details.CPUsInUncoreCaches(uncoreID).Size() == a.topo.CPUDetails.CPUsInUncoreCaches(uncoreID).Size()
}

// Sort all NUMA nodes with at least one free CPU for resize.
//
// The sorting follows a nested two-part ordering: elements with retained CPUs first, then other available elements. (different from add operation)
// Within each part, elements are sorted by number of free CPUs they contain (ascending). (Same as add operation)
//
// If NUMA nodes are higher than sockets in the memory hierarchy (each NUMA node contains more than one socket),
// the NUMA nodes are sorted directly: NUMA nodes with retained CPUs first, then other NUMA nodes.
//
// If instead sockets are higher in the memory hierarchy than NUMA nodes (each socket contains more than one NUMA node),
//  1. First, sort sockets: sockets with retained CPUs first, then other sockets.
//  2. Within each socket, sort NUMA nodes: NUMA nodes with retained CPUs first, then other NUMA nodes.
func (a *cpuAccumulatorForResize) sortAvailableNUMANodes() []int {
	return a.numaOrSocketsFirst.sortAvailableNUMANodes()
}

// Sort all sockets with at least one free CPU for resize.
//
// The sorting follows a nested two-part ordering: elements with retained CPUs first, then other available elements. (different from add operation)
// Within each part, elements are sorted by number of free CPUs they contain (ascending). (Same as add operation)
//
// If NUMA nodes are higher in the memory hierarchy than sockets (each NUMA node contains more than one socket),
//  1. First, sort NUMA nodes: NUMA nodes with retained CPUs first, then other NUMA nodes.
//  2. Within each NUMA node, sort sockets: sockets with retained CPUs first, then other sockets.
//
// If instead sockets are higher than NUMA nodes in the memory hierarchy (each socket contains more than one NUMA node),
// the sockets are sorted directly: sockets with retained CPUs first, then other sockets.
func (a *cpuAccumulatorForResize) sortAvailableSockets() []int {
	return a.numaOrSocketsFirst.sortAvailableSockets()
}

// Sort all cores with at least one free CPU for resize.
//
// The sorting follows a nested two-part ordering: elements with retained CPUs first, then other available elements. (different from add operation)
// Within each part, elements are sorted by number of free CPUs they contain (ascending). (Same as add operation)
//
// If NUMA nodes are higher in the memory hierarchy than sockets (each NUMA node contains more than one socket),
//  1. First, sort NUMA nodes: NUMA nodes with retained CPUs first, then other NUMA nodes.
//  2. Within each NUMA node, sort sockets: sockets with retained CPUs first, then other sockets.
//  3. Within each socket, sort cores: cores with retained CPUs first, then other cores.
//
// If instead sockets are higher in the memory hierarchy than NUMA nodes (each socket contains more than one NUMA node),
//  1. First, sort sockets: sockets with retained CPUs first, then other sockets.
//  2. Within each socket, sort NUMA nodes: NUMA nodes with retained CPUs first, then other NUMA nodes.
//  3. Within each NUMA node, sort cores: cores with retained CPUs first, then other cores.
func (a *cpuAccumulatorForResize) sortAvailableCores() []int {
	return a.numaOrSocketsFirst.sortAvailableCores()
}

// Sort all free CPUs for resize.
//
// First, uses sortAvailableCores() to determine the core ordering.
// Then, sorts CPUs within each core by numerical ID. (Same as sortAvailableCPUsPacked)
func (a *cpuAccumulatorForResize) sortAvailableCPUsPacked() []int {
	var result []int
	for _, core := range a.sortAvailableCores() {
		cpus := a.details.CPUsInCores(core).List()
		result = append(result, cpus...)
	}
	return result
}

// Sort all available CPUs to spread across physical cores for resize:
//   - First order sockets using sortAvailableSockets().
//   - Then, within each socket, order cores by free-CPU count and select
//     CPUs round-robin across those cores so sibling threads are only used
//     once every physical core has contributed a CPU.
//
// Same as sortAvailableCPUsSpread(), except sockets are ordered with
// sortAvailableSockets() to keep retained CPUs' sockets first.
func (a *cpuAccumulatorForResize) sortAvailableCPUsSpread() []int {
	var result []int
	for _, socket := range a.sortAvailableSockets() {
		cores := a.details.CoresInSockets(socket).UnsortedList()
		// Deliberately sorted by free-CPU count instead of sortAvailableCores():
		// sortAvailableCores() performs NUMA ranking, but spread mode at
		// this stage only cares about physical-core coverage.
		a.sort(cores, a.details.CPUsInCores)
		cpusPerCore := make([][]int, 0, len(cores))
		maxCPUsPerCore := 0
		for _, core := range cores {
			cpus := a.details.CPUsInCores(core).List()
			cpusPerCore = append(cpusPerCore, cpus)
			if len(cpus) > maxCPUsPerCore {
				maxCPUsPerCore = len(cpus)
			}
		}
		for cpuIndex := 0; cpuIndex < maxCPUsPerCore; cpuIndex++ {
			for _, cpus := range cpusPerCore {
				if cpuIndex < len(cpus) {
					result = append(result, cpus[cpuIndex])
				}
			}
		}
	}
	return result
}

func (a *cpuAccumulatorForResize) take(cpus cpuset.CPUSet) {
	a.cpuAccumulator.take(cpus)
	a.resultDetails = a.topo.CPUDetails.KeepOnly(a.result)
}

// takeFullUncore is the resize-aware variant of cpuAccumulator.takeFullUncore.
// It calls isFullUncoreCache and sortAvailableUncoreCaches on the resize accumulator
// so retained CPUs count toward a full cache.
func (a *cpuAccumulatorForResize) takeFullUncore() {
	for _, uncore := range a.sortAvailableUncoreCaches() {
		if a.isSatisfied() {
			return
		}
		if a.isFullUncoreCache(uncore) {
			cpusInUncore := a.details.CPUsInUncoreCaches(uncore)
			if !a.needsAtLeast(cpusInUncore.Size()) {
				continue
			}
			a.logger.V(4).Info("takeFullUncore ForResize: claiming uncore", "uncore", uncore)
			a.take(cpusInUncore)
		}
	}
}

// takePartialUncore is the resize-aware variant of cpuAccumulator.takePartialUncore.
// Currently calls isFullCore first then CoresNeededInUncoreCache select cores
// without checking the available CPUs number per core, following the same strategy
// as cpuAccumulator.takePartialUncore to allocate remaining CPUs.
// TODO: Use conceptual framework from fix of https://github.com/kubernetes/kubernetes/pull/137583
// and update description.
func (a *cpuAccumulatorForResize) takePartialUncore(uncoreID int) {
	// First to take the cores with allocated CPUs in this uncore when SMT/hyperthread is enabled
	if a.topo.CPUsPerCore() != 1 {
		// find cores in this uncore cache that already have one allocated CPU
		allocatedCoresInUncoreCache := a.resultDetails.CoresInUncoreCache(uncoreID)
		availableCoresInUncoreCache := a.details.CoresInUncoreCache(uncoreID)
		allocatedCores := availableCoresInUncoreCache.Intersection(allocatedCoresInUncoreCache).List()
		// Take full cores that already have allocated CPUs.
		for _, core := range allocatedCores {
			if a.isFullCore(core) {
				cpusInCore := a.details.CPUsInCores(core)
				if !a.needsAtLeast(cpusInCore.Size()) {
					continue
				}
				a.logger.V(4).Info("takePartialUncore ForResize: claiming Core", "core", core, "cpusInCore", cpusInCore)
				a.take(cpusInCore)
			}
		}
	}

	// Second to take the other cores or CPUs in this uncore
	// determine the number of cores needed whether SMT/hyperthread is enabled or disabled
	numCoresNeeded := (a.numCPUsNeeded + a.topo.CPUsPerCore() - 1) / a.topo.CPUsPerCore()

	// determine the N number of free cores (physical cpus) within the UncoreCache, then
	// determine the M number of free cpus (virtual cpus) that correspond with the free cores
	freeCores := a.details.CoresNeededInUncoreCache(numCoresNeeded, uncoreID)
	freeCPUs := a.details.CPUsInCores(freeCores.UnsortedList()...)

	// when SMT/hyperthread is enabled and remaining cpu requirement is an odd integer value:
	// sort the free CPUs that were determined based on the cores that have available cpus.
	// if the amount of free cpus is greather than the cpus needed, we can drop the last cpu
	// since the odd integer request will only require one out of the two free cpus that
	// correspond to the last core
	if a.numCPUsNeeded%2 != 0 && a.topo.CPUsPerCore() > 1 {
		// we sort freeCPUs to ensure we pack virtual cpu allocations, meaning we allocate
		// whole core's worth of cpus as much as possible to reduce smt-misalignment
		sortFreeCPUs := freeCPUs.List()
		if len(sortFreeCPUs) > a.numCPUsNeeded {
			// if we are in takePartialUncore, the accumulator is not satisfied after
			// takeFullUncore, so freeCPUs.Size() can't be < 1
			sortFreeCPUs = sortFreeCPUs[:freeCPUs.Size()-1]
		}
		freeCPUs = cpuset.New(sortFreeCPUs...)
	}

	// claim the cpus if the free cpus within the UncoreCache can satisfy the needed cpus
	claimed := (a.numCPUsNeeded == freeCPUs.Size())
	a.logger.V(4).Info("takePartialUncore: trying to claim partial uncore",
		"uncore", uncoreID,
		"claimed", claimed,
		"needed", a.numCPUsNeeded,
		"cores", freeCores.String(),
		"cpus", freeCPUs.String())
	if !claimed {
		return

	}
	a.take(freeCPUs)
}

// takeUncoreCache is the resize-aware variant of cpuAccumulator.takeUncoreCache.
// It calls takeFullUncore, takePartialUncore, and sortAvailableUncoreCaches on the resize accumulator.
// Optimization: takeFullUncore is called once before the for loop (instead of inside the loop).
func (a *cpuAccumulatorForResize) takeUncoreCache() {
	// take full UncoreCache if the CPUs needed is greater than free UncoreCache size
	a.takeFullUncore()
	if a.isSatisfied() {
		return
	}

	// take partial UncoreCache if the CPUs needed is less than free UncoreCache size
	for _, uncore := range a.sortAvailableUncoreCaches() {
		a.takePartialUncore(uncore)
		if a.isSatisfied() {
			return
		}
	}
}

// takeFullNUMANodes is the resize-aware variant of cpuAccumulator.takeFullNUMANodes.
// Algorithm structure is identical, but calls isFullNUMANode and sortAvailableNUMANodes
// on the resize accumulator.
func (a *cpuAccumulatorForResize) takeFullNUMANodes() {
	for _, numa := range a.sortAvailableNUMANodes() {
		if a.isFullNUMANode(numa) {
			cpusInNUMANode := a.details.CPUsInNUMANodes(numa)
			if !a.needsAtLeast(cpusInNUMANode.Size()) {
				continue
			}
			a.logger.V(4).Info("takeFullNUMANodes ForResize: claiming NUMA node", "numa", numa, "cpusInNUMANode", cpusInNUMANode)
			a.take(cpusInNUMANode)
		}
	}
}

// takeFullSockets is the resize-aware variant of cpuAccumulator.takeFullSockets.
// Algorithm structure is identical, but calls isFullSocket and sortAvailableSockets
// on the resize accumulator.
func (a *cpuAccumulatorForResize) takeFullSockets() {
	for _, socket := range a.sortAvailableSockets() {
		if a.isFullSocket(socket) {
			cpusInSocket := a.details.CPUsInSockets(socket)
			if !a.needsAtLeast(cpusInSocket.Size()) {
				continue
			}
			a.logger.V(4).Info("takeFullSockets ForResize: claiming Socket", "socket", socket, "cpusInSocket", cpusInSocket)
			a.take(cpusInSocket)
		}
	}
}

// takeFullCores is the resize-aware variant of cpuAccumulator.takeFullCores.
// Algorithm structure is identical, but calls isFullCore and sortAvailableCores
// on the resize accumulator.
func (a *cpuAccumulatorForResize) takeFullCores() {
	for _, core := range a.sortAvailableCores() {
		if a.isFullCore(core) {
			cpusInCore := a.details.CPUsInCores(core)
			if !a.needsAtLeast(cpusInCore.Size()) {
				continue
			}
			a.logger.V(4).Info("takeFullCores: claiming Core", "core", core, "cpusInCore", cpusInCore)
			a.take(cpusInCore)
		}
	}
}

// takeRemainingCPUs is the resize-aware variant of cpuAccumulator.takeRemainingCPUs.
// Algorithm structure is identical, but calls the resize sorter selected at construction.
func (a *cpuAccumulatorForResize) takeRemainingCPUs() {
	for _, cpu := range a.availableCPUSorter.sort() {
		a.logger.V(4).Info("takeRemainingCPUs: claiming CPU", "cpu", cpu)
		a.take(cpuset.New(cpu))
		if a.isSatisfied() {
			return
		}
	}
}

// rangeNUMANodesNeededToSatisfy returns minimum and maximum (in this order) number of NUMA nodes
// needed to satisfy the cpuAccumulator's goal of accumulating `a.numCPUsNeeded` CPUs, assuming that
// CPU groups have size given by the `cpuGroupSize` argument.
//
// When minNumNUMAsFromHint is greater than 0, it raises the minimum NUMA node count
// to honor the TopologyManager's topology hint. This ensures CPU allocation alignes with device
// topology (e.g., GPUs, NICs that span multiple NUMA nodes). If the hint exceeds the maximum feasible NUMA count,
// minNumNUMAsFromHint is ignored.
func (a *cpuAccumulatorForResize) rangeNUMANodesNeededToSatisfy(cpuGroupSize int, minNumNUMAsFromHint int, totalCPUsNeeded int) (minNumNUMAs, maxNumNUMAs int) {
	// Get the total number of NUMA nodes in the system.
	numNUMANodes := a.topo.CPUDetails.NUMANodes().Size()

	// Get the total number of NUMA nodes that have CPUs available on them.
	// For resize, also count NUMA nodes that have no available CPUs but hold CPUs retained by this container.
	// For add, resultDetails is unset, so the count is the number of NUMA nodes with available CPUs.
	numNUMANodesAvailable := a.details.NUMANodes().Union(a.resultDetails.NUMANodes()).Size()

	// Get the total number of CPUs in the system.
	numCPUs := a.topo.CPUDetails.CPUs().Size()

	// Get the total number of 'cpuGroups' in the system.
	numCPUGroups := (numCPUs-1)/cpuGroupSize + 1

	// Calculate the number of 'cpuGroups' per NUMA Node in the system (rounding up).
	numCPUGroupsPerNUMANode := (numCPUGroups-1)/numNUMANodes + 1

	// Calculate the number of available 'cpuGroups' across all NUMA nodes as
	// well as the number of 'cpuGroups' that need to be allocated (rounding up).

	// 'totalCPUsNeeded' is the total number of CPUs requested for the container.
	// For resize, it counts retained CPUs plus the remaining CPUs to allocate;
	numCPUGroupsNeeded := (totalCPUsNeeded-1)/cpuGroupSize + 1

	// Calculate the minimum number of numa nodes required to satisfy the
	// allocation (rounding up).
	minNumNUMAs = (numCPUGroupsNeeded-1)/numCPUGroupsPerNUMANode + 1

	// Calculate the maximum number of numa nodes required to satisfy the allocation.
	maxNumNUMAs = min(numCPUGroupsNeeded, numNUMANodesAvailable)

	// If the TopologyManager selected a specific NUMA affinity, honor it by
	// ensuring we distribute across at least that many NUMA nodes. This
	// prevents CPUManager from shrinking a multi-NUMA affinity back to fewer
	// nodes than what TopologyManager intended (e.g., for GPU/NIC alignment).
	if minNumNUMAsFromHint > 0 {
		if minNumNUMAsFromHint > minNumNUMAs && minNumNUMAsFromHint <= maxNumNUMAs {
			minNumNUMAs = minNumNUMAsFromHint
		} else if minNumNUMAsFromHint > maxNumNUMAs {
			a.logger.V(4).Info("NUMA affinity hint exceeds maximum feasible NUMA count, ignoring", "minNumNUMAsFromHint", minNumNUMAsFromHint, "maxNumNUMAs", maxNumNUMAs)
		}
	}
	return
}

// NUMA Distributed Strategy for Resize
//
// When distributing CPUs evenly across NUMA nodes, this strategy considers retained CPUs
// (currentlyAllocatedCPUs for scale-up, baselineCPUs for scale-down).
// This ensures both retained CPUs and newly allocated CPUs are balanced across NUMA nodes
// during resize operations.
//
// Key differences from add operation (takeByTopologyNUMADistributed):
//  1. Retained CPUs preservation: Before allocating new CPUs, the retained CPUs must be preserve
//     first (currentlyAllocatedCPUs for scale-up, baselineCPUs for scale-down).
//  2. NUMA node sorting priority: When sorting NUMA nodes, prioritize nodes that have
//     retained CPUs, ensuring topology continuity during resize operations.
//  3. NUMA node combination constraint: The selected NUMA node combination must include
//     all NUMA nodes with retained CPUs. minNUMAs cannot be less than the number of
//     NUMA nodes with retained CPUs.
//  4. CPU group calculation: When calculating the number of CPU groups, consider both
//     available CPUs (acc.details) and retained CPUs (acc.resultDetails), not just available CPUs.
//  5. Remainder CPU allocation: Each NUMA node's distribution is filled by its retained CPUs first,
//     and the excess beyond the distribution as remainder. The balance score is then computed
//     the same way as the add operation, based on the remaining CPUs of each numa node.
//  6. Final CPU allocation: Call takeByTopologyNUMAPacked with retained CPUs,
//     and only take the difference (newly allocated CPUs) to avoid duplicating retained CPUs.
//
// minNUMAsFromHint is the NUMA node count from the TopologyManager's topology
// hint. When non-zero, it raises the minimum NUMA node count for distribution,
// ensuring CPU allocation stays aligned with device topology (e.g., GPUs, NICs).
// Pass 0 to use the default behavior (minimum NUMAs needed to satisfy the request).
func takeByTopologyNUMADistributedForResize(logger klog.Logger, topo *topology.CPUTopology, availableCPUs cpuset.CPUSet, numCPUs int, cpuGroupSize int, cpuSortingStrategy CPUSortingStrategy, alignBySocket bool, currentlyAllocatedCPUs cpuset.CPUSet, baselineCPUs cpuset.CPUSet, minNUMAsFromHint int) (cpuset.CPUSet, error) {
	// If the number of CPUs requested cannot be handed out in chunks of
	// 'cpuGroupSize', then we just call out the packing algorithm since we
	// can't distribute CPUs in this chunk size.
	// PreferAlignByUncoreCache feature not implemented here yet and set to false.
	// Support for PreferAlignByUncoreCache to be done at beta release.
	if (numCPUs % cpuGroupSize) != 0 {
		return takeByTopologyNUMAPackedForResize(logger, topo, availableCPUs, numCPUs, cpuSortingStrategy, false, currentlyAllocatedCPUs, baselineCPUs)
	}

	// If the number of CPUs requested to be retained is not a subset
	// of currentlyAllocatedCPUs, then we fail early.
	if !currentlyAllocatedCPUs.IsEmpty() && !baselineCPUs.IsEmpty() {
		if !baselineCPUs.IsSubsetOf(currentlyAllocatedCPUs) {
			return cpuset.New(), fmt.Errorf("requested CPUs to be retained %s are not a subset of reusable CPUs %s", baselineCPUs.String(), currentlyAllocatedCPUs.String())
		}
	}

	// Otherwise build an accumulator to start allocating CPUs from.
	// For resize operations, use newCPUAccumulatorForResize to take the
	// retained CPU (currentlyAllocatedCPUs for scale up, baselineCPUs for scale down) first.
	acc := newCPUAccumulatorForResize(logger, topo, availableCPUs, numCPUs, cpuSortingStrategy, currentlyAllocatedCPUs, baselineCPUs)
	if acc.isSatisfied() {
		return acc.result, nil
	}
	if acc.isFailed() {
		return cpuset.New(), fmt.Errorf("not enough cpus available to satisfy request: requested=%d (of which %d already allocated), available=%d", numCPUs, acc.result.Size(), acc.details.CPUs().Size())
	}

	// Get the list of NUMA nodes represented by the set of CPUs in 'availableCPUs'.
	// For resize operations, use sortAvailableNUMANodes to prioritize NUMA nodes with retained CPUs,
	// ensuring topology continuity during resize operations.
	numas := acc.sortAvailableNUMANodes()

	// NUMA nodes with only retained CPUs have no free CPUs and are missing
	// from the sorted list; combos must cover all NUMA nodes with retained
	// CPUs, so prepend them back.
	missingRetainedNUMANodes := acc.resultDetails.NUMANodes().Difference(cpuset.New(numas...))
	numas = append(missingRetainedNUMANodes.List(), numas...)

	// Calculate the minimum and maximum possible number of NUMA nodes that
	// could satisfy this request. This is used to optimize how many iterations
	// of the loop we need to go through below.
	minNUMAs, maxNUMAs := acc.rangeNUMANodesNeededToSatisfy(cpuGroupSize, minNUMAsFromHint, numCPUs)
	// For resize operations, minNUMAs should not be less than the number of
	// NUMA nodes with retained CPUs, ensuring we consider combinations that
	// include all NUMA nodes with retained CPUs.
	minNUMAs = max(minNUMAs, acc.resultDetails.NUMANodes().Size())

	// Try combinations of 1,2,3,... NUMA nodes until we find a combination
	// where we can evenly distribute CPUs across them. To optimize things, we
	// don't always start at 1 and end at len(numas). Instead, we use the
	// values of 'minNUMAs' and 'maxNUMAs' calculated above.
	for k := minNUMAs; k <= maxNUMAs; k++ {
		// Iterate through the various n-choose-k NUMA node combinations,
		// looking for the combination of NUMA nodes that can best have CPUs
		// distributed across them.
		var bestBalance = math.MaxFloat64
		var bestRemainder []int = nil
		var bestCombo []int = nil
		var bestBalanceInOneSocket = false
		var bestAllocatedRemainder = 0
		acc.iterateCombinations(numas, k, func(combo []int) LoopControl {
			// If we've already found a combo with a balance of 0 in a
			// different iteration, then don't bother checking any others.
			if bestBalance == 0 && (!alignBySocket || bestBalanceInOneSocket) {
				return Break
			}

			// For resize operations, ensure the combination includes all NUMA nodes with retained CPUs.
			comboSet := cpuset.New(combo...)
			if !acc.resultDetails.NUMANodes().IsSubsetOf(comboSet) {
				return Continue
			}

			// Check that this combination of NUMA nodes has enough CPUs to
			// satisfy the allocation overall.
			cpus := acc.details.CPUsInNUMANodes(combo...)
			// For resize operations, acc.result contains retained CPUs that should be counted.
			if (cpus.Size() + acc.result.Size()) < numCPUs {
				return Continue
			}

			// Check that CPUs can be handed out in groups of size
			// 'cpuGroupSize' across the NUMA nodes in this combo.
			numCPUGroups := 0
			for _, numa := range combo {
				// For resize operations, count both available CPUs (acc.details) and retained CPUs (acc.resultDetails)
				// to determine how many CPU groups can be formed from this NUMA node.
				numCPUGroups += ((acc.details.CPUsInNUMANodes(numa).Size() + acc.resultDetails.CPUsInNUMANodes(numa).Size()) / cpuGroupSize)
			}
			if (numCPUGroups * cpuGroupSize) < numCPUs {
				return Continue
			}

			// Calculate an even distribution of CPUs in groups of size
			// 'cpuGroupSize'.
			distribution := (numCPUs / len(combo) / cpuGroupSize) * cpuGroupSize
			if alignBySocket {
				for _, numa := range combo {
					// distribution should not be more than available CPUs
					// in each NUMA node in combo if alignBySocket is set.
					// For resize operations, count both available CPUs (acc.details) and retained CPUs (acc.resultDetails)
					// to determine the maximum CPUs that can be assigned from this NUMA node.
					availableCPUsInNUMA := (acc.details.CPUsInNUMANodes(numa).Size() + acc.resultDetails.CPUsInNUMANodes(numa).Size()) / cpuGroupSize * cpuGroupSize
					if distribution > availableCPUsInNUMA {
						distribution = availableCPUsInNUMA
					}
				}
			}
			// Check that each NUMA node in this combination can allocate
			// an even distribution of CPUs in groups of size 'cpuGroupSize'.
			// For resize operations, each NUMA node can have at most
			// (distribution + neededRemainder) CPUs because a single NUMA node could
			// potentially receive all neededRemainder CPUs.
			// allocatedRemainder tracks how many remainder CPUs have already been allocated
			// to NUMA nodes in this combo (beyond the base distribution).
			allocatedRemainder := 0
			// neededRemainder is the total remainder CPUs to distribute after giving each
			// NUMA node in the combo an equal 'distribution' share.
			neededRemainder := numCPUs - (distribution * len(combo))
			for _, numa := range combo {
				cpus := acc.details.CPUsInNUMANodes(numa)
				allocateCpus := acc.resultDetails.CPUsInNUMANodes(numa)
				// For resize operations, skip NUMA early if its total CPUs (available + retained) is less than distribution
				// or has more CPUs than (distribution + neededRemainder).
				if (cpus.Size()+allocateCpus.Size()) < distribution || allocateCpus.Size() > (distribution+neededRemainder) {
					return Continue
				}
				// For resize operations, increase total allocated remainder by remainder CPUs (beyond the base distribution)
				// on this NUMA node.
				if allocateCpus.Size() > distribution {
					allocatedRemainder += allocateCpus.Size() - distribution
				}
			}
			// Calculate how many CPUs will be available on each NUMA node in
			// the system after allocating an even distribution of CPU groups
			// of size 'cpuGroupSize' from each NUMA node in 'combo'. This will
			// be used in the "balance score" calculation to help decide if
			// this combo should ultimately be chosen.
			availableAfterAllocation := make(mapIntInt, len(numas))
			for _, numa := range numas {
				availableAfterAllocation[numa] = acc.details.CPUsInNUMANodes(numa).Size()
			}
			for _, numa := range combo {
				retained := acc.resultDetails.CPUsInNUMANodes(numa).Size()
				if newlyTaken := distribution - retained; newlyTaken > 0 {
					availableAfterAllocation[numa] -= newlyTaken
				}
			}

			// Check if there are any remaining CPUs to distribute across the
			// NUMA nodes once CPUs have been evenly distributed in groups of
			// size 'cpuGroupSize'.
			// For resize operations, subtract allocatedRemainder when calculating remainder
			remainder := numCPUs - (distribution * len(combo)) - allocatedRemainder

			// For resize operations, if remainder is negative, it means the initial allocation is unbalanced across NUMA nodes
			// (more NUMA nodes have more than distribution CPUs but less than distribution + neededRemainder). Skip this combo.
			if remainder < 0 {
				return Continue
			}

			// Get a list of NUMA nodes to consider pulling the remainder CPUs
			// from. This list excludes NUMA nodes that don't have at least
			// 'cpuGroupSize' CPUs available after being allocated
			// 'distribution' number of CPUs.
			var remainderCombo []int
			for _, numa := range combo {
				if availableAfterAllocation[numa] >= cpuGroupSize {
					remainderCombo = append(remainderCombo, numa)
				}
			}

			// Declare a set of local variables to help track the "balance
			// scores" calculated when using different subsets of
			// 'remainderCombo' to allocate remainder CPUs from.
			var bestLocalBalance = math.MaxFloat64
			var bestLocalRemainder []int = nil

			// If there aren't any remainder CPUs to allocate, then calculate
			// the "balance score" of this combo as the standard deviation of
			// the values contained in 'availableAfterAllocation'.
			if remainder == 0 {
				bestLocalBalance = standardDeviation(availableAfterAllocation.Values())
				bestLocalRemainder = nil
			}

			// Otherwise, find the best "balance score" when allocating the
			// remainder CPUs across different subsets of NUMA nodes in 'remainderCombo'.
			// These remainder CPUs are handed out in groups of size 'cpuGroupSize'.
			// We start from k=len(remainderCombo) and walk down to k=1 so that
			// we continue to distribute CPUs as much as possible across
			// multiple NUMA nodes.
			for k := len(remainderCombo); remainder > 0 && k >= 1; k-- {
				acc.iterateCombinations(remainderCombo, k, func(subset []int) LoopControl {
					// Make a local copy of 'remainder'.
					remainder := remainder

					// Make a local copy of 'availableAfterAllocation'.
					availableAfterAllocation := availableAfterAllocation.Clone()

					// If this subset is not capable of allocating all
					// remainder CPUs in whole groups of 'cpuGroupSize',
					// continue to the next one. Leftover CPUs smaller than a
					// group cannot be handed out, so a raw CPU sum overcounts.
					availableGroups := 0
					for _, numa := range subset {
						availableGroups += availableAfterAllocation[numa] / cpuGroupSize
					}
					if availableGroups*cpuGroupSize < remainder {
						return Continue
					}

					// For all NUMA nodes in 'subset', walk through them,
					// removing 'cpuGroupSize' number of CPUs from each
					// until all remainder CPUs have been accounted for.
					for remainder > 0 {
						for _, numa := range subset {
							if remainder == 0 {
								break
							}
							if availableAfterAllocation[numa] < cpuGroupSize {
								continue
							}
							availableAfterAllocation[numa] -= cpuGroupSize
							remainder -= cpuGroupSize
						}
					}

					// Calculate the "balance score" as the standard deviation
					// of the number of CPUs available on all NUMA nodes in the
					// system after the remainder CPUs have been allocated
					// across 'subset' in groups of size 'cpuGroupSize'.
					balance := standardDeviation(availableAfterAllocation.Values())
					if balance < bestLocalBalance {
						bestLocalBalance = balance
						bestLocalRemainder = subset
					}

					return Continue
				})
			}

			// If alignBySocket is enabled, prefer combinations whose NUMA nodes
			// are in one socket over any cross-socket combination. When comparing
			// combinations in the same socket category, pick the one with the
			// lower balance score.
			inSameSocket := false
			if alignBySocket {
				inSameSocket = topo.CPUDetails.AreNUMANodesInSameSocket(combo)
			}
			isBetter := bestLocalBalance < bestBalance
			if alignBySocket && inSameSocket != bestBalanceInOneSocket {
				isBetter = inSameSocket
			}

			if isBetter {
				bestBalance = bestLocalBalance
				bestRemainder = bestLocalRemainder
				bestCombo = combo
				bestAllocatedRemainder = allocatedRemainder
				if alignBySocket {
					bestBalanceInOneSocket = inSameSocket
				}
			}

			return Continue
		})

		// If we made it through all of the iterations above without finding a
		// combination of NUMA nodes that can properly balance CPU allocations,
		// then move on to the next larger set of NUMA node combinations.
		if bestCombo == nil {
			continue
		}

		// Otherwise, start allocating CPUs from the NUMA node combination
		// chosen. First allocate an even distribution of CPUs in groups of
		// size 'cpuGroupSize' from 'bestCombo'.
		distribution := (numCPUs / len(bestCombo) / cpuGroupSize) * cpuGroupSize
		// At this stage we are past NUMA-node selection (so we no longer need to
		// consider alignBySocket); that happened when choosing bestCombo. Here we
		// only ensure we do not ask any selected NUMA node for more CPUs than it can provide.
		for _, numa := range bestCombo {
			// For resize operations, count both available CPUs (acc.details) and retained CPUs (acc.resultDetails)
			// to determine the maximum CPUs that can be assigned from this NUMA node.
			availableCPUsInNUMA := (acc.details.CPUsInNUMANodes(numa).Size() + acc.resultDetails.CPUsInNUMANodes(numa).Size()) / cpuGroupSize * cpuGroupSize
			if distribution > availableCPUsInNUMA {
				distribution = availableCPUsInNUMA
			}
		}
		// For resize operations, allocate 'distribution' CPUs from each NUMA node in bestCombo.
		// Pass retained CPUs (allocatedCPUs) to takeByTopologyNUMAPackedForResize,
		// which will preserve them and only allocate additional CPUs.
		// Skip NUMA nodes that have already reached or exceeded the distribution target.
		for _, numa := range bestCombo {
			allocatedCPUs := acc.resultDetails.CPUsInNUMANodes(numa)
			if allocatedCPUs.Size() >= distribution {
				continue
			}
			cpus, err := takeByTopologyNUMAPackedForResize(logger, acc.topo, acc.details.CPUsInNUMANodes(numa), distribution, cpuSortingStrategy, false, allocatedCPUs, allocatedCPUs)
			if err != nil {
				logger.V(4).Info("CPU distribution allocation from NUMA node returned error", "numa", numa, "err", err)
			}
			acc.take(cpus.Difference(allocatedCPUs))
		}

		// Then allocate any remaining CPUs in groups of size 'cpuGroupSize'
		// from each NUMA node in the remainder set for remainder > 0.
		// For resize operations, subtract bestAllocatedRemainder because those remainder CPUs are already allocated.
		remainder := numCPUs - (distribution * len(bestCombo)) - bestAllocatedRemainder
		for remainder > 0 {
			for _, numa := range bestRemainder {
				if remainder == 0 {
					break
				}
				if acc.details.CPUsInNUMANodes(numa).Size() < cpuGroupSize {
					continue
				}
				// For resize operations, allocate 1 extra cpuGroupSize to this NUMA node as remainder.
				// Pass retained CPUs (allocatedCPUs) to takeByTopologyNUMAPackedForResize,
				// which will preserve them and allocate additional CPUs to reach needCPUsInNuma.
				allocatedCPUs := acc.resultDetails.CPUsInNUMANodes(numa)
				needCPUsInNuma := allocatedCPUs.Size() + cpuGroupSize
				cpus, err := takeByTopologyNUMAPackedForResize(logger, acc.topo, acc.details.CPUsInNUMANodes(numa), needCPUsInNuma, cpuSortingStrategy, false, allocatedCPUs, allocatedCPUs)
				if err != nil {
					logger.V(4).Info("CPU remainder allocation from NUMA node returned error", "numa", numa, "err", err)
				}
				acc.take(cpus.Difference(allocatedCPUs))
				remainder -= cpuGroupSize
			}
		}

		// If we haven't allocated all of our CPUs at this point, then something
		// went wrong in our accounting and we should error out.
		if acc.numCPUsNeeded > 0 {
			return cpuset.New(), fmt.Errorf("accounting error, not enough CPUs allocated, remaining: %v", acc.numCPUsNeeded)
		}

		// Likewise, if we have allocated too many CPUs at this point, then something
		// went wrong in our accounting and we should error out.
		if acc.numCPUsNeeded < 0 {
			return cpuset.New(), fmt.Errorf("accounting error, too many CPUs allocated, remaining: %v", acc.numCPUsNeeded)
		}

		// Otherwise, return the result
		return acc.result, nil
	}

	// If we never found a combination of NUMA nodes that we could properly
	// distribute CPUs across, fall back to the packing algorithm.
	return takeByTopologyNUMAPackedForResize(logger, topo, availableCPUs, numCPUs, cpuSortingStrategy, false, currentlyAllocatedCPUs, baselineCPUs)
}

// takeByTopologyNUMAPackedForResize returns a CPUSet of size 'numCPUs' for resize operations.
//
// Algorithm structure is identical to takeByTopologyNUMAPacked, but uses
// cpuAccumulatorForResize methods that account for retained CPUs.
//
// Key differences from takeByTopologyNUMAPacked:
//
//  1. Retained CPUs are preserved: Before allocating new CPUs, the retained CPUs must be preserve
//     first (currentlyAllocatedCPUs for scale-up, baselineCPUs for scale-down). This to respect
//     KEP-5554 contract.
//
//  2. Priority to retained topology elements: When sorting each topology elements (NUMA nodes,
//     sockets, UncoreCache, cores), elements that have retained CPUs are prioritized, followed by
//     other available elements. This two-part ordering ensures topology continuity and keeps
//     retained and newly allocated CPUs close in the topology hierarchy.
//
//  3. Two-part CPU pool consideration: When determining if a topology element can be fully
//     claimed, both available CPUs (free pool) and retained CPUs (to this container) are
//     considered together.
func takeByTopologyNUMAPackedForResize(logger klog.Logger, topo *topology.CPUTopology, availableCPUs cpuset.CPUSet, numCPUs int, cpuSortingStrategy CPUSortingStrategy, preferAlignByUncoreCache bool, currentlyAllocatedCPUs cpuset.CPUSet, baselineCPUs cpuset.CPUSet) (cpuset.CPUSet, error) {

	// If the number of CPUs requested to be retained is not a subset
	// of currentlyAllocatedCPUs, then we fail early.
	if !currentlyAllocatedCPUs.IsEmpty() && !baselineCPUs.IsEmpty() {
		if !baselineCPUs.IsSubsetOf(currentlyAllocatedCPUs) {
			return cpuset.New(), fmt.Errorf("requested CPUs to be retained %s are not a subset of reusable CPUs %s", baselineCPUs.String(), currentlyAllocatedCPUs.String())
		}
	}

	// Resize Step 1: Preserve retained CPUs (currentlyAllocatedCPUs for scale-up, baselineCPUs for scale-down)
	acc := newCPUAccumulatorForResize(logger, topo, availableCPUs, numCPUs, cpuSortingStrategy, currentlyAllocatedCPUs, baselineCPUs)
	if acc.isSatisfied() {
		return acc.result, nil
	}
	if acc.isFailed() {
		return cpuset.New(), fmt.Errorf("not enough cpus available to satisfy request: requested=%d (of which %d already allocated), available=%d", numCPUs, acc.result.Size(), acc.details.CPUs().Size())
	}

	// Algorithm: topology-aware best-fit
	// 1. Acquire whole NUMA nodes and sockets, if available and the container
	//    requires at least a NUMA node or socket's-worth of CPUs. If NUMA
	//    Nodes map to 1 or more sockets, pull from NUMA nodes first.
	//    Otherwise pull from sockets first.
	// Resize Step 2: Take full NUMA nodes/sockets with retained CPUs first, then other full NUMA nodes/sockets.
	acc.numaOrSocketsFirst.takeFullFirstLevel()
	if acc.isSatisfied() {
		return acc.result, nil
	}

	acc.numaOrSocketsFirst.takeFullSecondLevel()
	if acc.isSatisfied() {
		return acc.result, nil
	}

	// 2. If PreferAlignByUncoreCache is enabled, acquire whole UncoreCaches
	//    if available and the container requires at least a UncoreCache's-worth
	//    of CPUs. Otherwise, acquire CPUs from the least amount of UncoreCaches.
	// Resize Step 3: Within each sorted NUMA node/socket, take full UncoreCaches with retained CPUs first, then other full UncoreCaches.
	if preferAlignByUncoreCache {
		acc.takeUncoreCache()
		if acc.isSatisfied() {
			return acc.result, nil
		}
	}

	// 3. Acquire whole cores, if available and the container requires at least
	//    a core's-worth of CPUs.
	//    If `CPUSortingStrategySpread` is specified, skip taking the whole core.
	// Resize Step 4: Within each sorted NUMA node/socket, take full cores with retained CPUs first, then other full cores.
	if cpuSortingStrategy != CPUSortingStrategySpread {
		acc.takeFullCores()
		if acc.isSatisfied() {
			return acc.result, nil
		}
	}

	// 4. Acquire single threads, preferring to fill partially-allocated cores
	//    on the same sockets as the whole cores we have already taken in this
	//    allocation.
	// Resize Step 5: Within each sorted cores, take partially-allocated cores first, then other CPUs.
	acc.takeRemainingCPUs()
	if acc.isSatisfied() {
		return acc.result, nil
	}

	return cpuset.New(), fmt.Errorf("failed to allocate requested CPUs: requested=%d (of which %d already allocated), available=%d", numCPUs, currentlyAllocatedCPUs.Size(), availableCPUs.Size())
}
