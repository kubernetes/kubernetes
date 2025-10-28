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

package memorymanager

import (
	"fmt"
	"sort"
	"strings"

	"github.com/go-logr/logr"
	cadvisorapi "github.com/google/cadvisor/info/v1"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	corehelper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/admission"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	cmqos "k8s.io/kubernetes/pkg/kubelet/cm/qos"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

const PolicyTypeStatic policyType = "Static"

type systemReservedMemory map[int]map[v1.ResourceName]uint64
type reusableMemory map[string]map[string]map[v1.ResourceName]uint64

// staticPolicy is implementation of the policy interface for the static policy
type staticPolicy struct {
	// machineInfo contains machine memory related information
	machineInfo *cadvisorapi.MachineInfo
	// reserved contains memory that reserved for kube
	systemReserved systemReservedMemory
	// topology manager reference to get container Topology affinity
	affinity topologymanager.Store
	// initContainersReusableMemory contains the memory allocated for init
	// containers that can be reused.
	// Note that the restartable init container memory is not included here,
	// because it is not reusable.
	initContainersReusableMemory reusableMemory
}

var _ Policy = &staticPolicy{}

// NewPolicyStatic returns new static policy instance
func NewPolicyStatic(logger klog.Logger, machineInfo *cadvisorapi.MachineInfo, reserved systemReservedMemory, affinity topologymanager.Store) (Policy, error) {
	var totalSystemReserved uint64
	for _, node := range reserved {
		if _, ok := node[v1.ResourceMemory]; !ok {
			continue
		}
		totalSystemReserved += node[v1.ResourceMemory]
	}

	// check if we have some reserved memory for the system
	if totalSystemReserved <= 0 {
		return nil, fmt.Errorf("[memorymanager] you should specify the system reserved memory")
	}

	return &staticPolicy{
		machineInfo:                  machineInfo,
		systemReserved:               reserved,
		affinity:                     affinity,
		initContainersReusableMemory: reusableMemory{},
	}, nil
}

func (p *staticPolicy) Name() string {
	return string(PolicyTypeStatic)
}

func (p *staticPolicy) Start(logger klog.Logger, s state.State) error {
	if err := p.validateState(logger, s); err != nil {
		logger.Error(err, "Invalid state, please drain node and remove policy state file")
		return err
	}
	return nil
}

// validatePodScopeResources checks for the "empty shared pool" scenario. This
// occurs when the sum of exclusive container memory requests consumes the
// entire pod-level budget, leaving no memory for containers that require a
// shared pool. Such a configuration is invalid because it would lead to
// containers in the shared pool having no memory allocated, causing them to
// fail.
func (p *staticPolicy) validatePodScopeResources(logger logr.Logger, pod *v1.Pod) error {
	podTotalMemory, err := getPodRequestedResources(logger, pod)
	if err != nil {
		return err
	}

	hasSharedLongRunningContainers := false
	// Sum memory requests for all containers that run for the full pod lifetime
	// (main containers and restartable init containers) to determine the
	// total exclusive memory usage that might impact the shared pool.
	sumOfLongRunningExclusiveMemory := make(map[v1.ResourceName]uint64)

	// Check for empty shared pool for standard init containers.
	for _, container := range pod.Spec.InitContainers {
		containerReqs, err := getContainerRequestedResources(logger, pod, &container)
		if err != nil {
			return err
		}

		if !podutil.IsRestartableInitContainer(&container) {
			if len(containerReqs) > 0 {
				// This check ensures that the pool of memory available to standard init containers
				// (the total pod budget minus any sidecars up to that point) is large enough to satisfy the
				// request of the largest standard init container that requires exclusive memory.
				// Since standard init containers run sequentially, we only need to fit the current.
				poolForStandardInits := make(map[v1.ResourceName]uint64)
				for resourceName, podTotalAmount := range podTotalMemory {
					poolForStandardInits[resourceName] = podTotalAmount - sumOfLongRunningExclusiveMemory[resourceName]
				}
				for resourceName, poolSize := range poolForStandardInits {
					if containerReqs[resourceName] > poolSize {
						return fmt.Errorf("pod rejected, largest exclusive init container requests %d of %s, but only %d available", containerReqs[resourceName], resourceName, poolSize)
					}
				}
			} else {
				// This check ensures that if there are any standard init containers that need a
				// shared pool, that pool is not empty. An empty pool would occur if the pod
				// budget is fully consumed by sidecars.
				for resourceName, poolSize := range podTotalMemory {
					if sumOfLongRunningExclusiveMemory[resourceName] >= poolSize {
						return fmt.Errorf("pod rejected, pod has shared init containers but no %s available for them", resourceName)
					}
				}
			}

			continue
		}

		// If there are restartable init containers, keep track for the empty shared pool check.
		if len(containerReqs) == 0 {
			hasSharedLongRunningContainers = true
			continue
		}

		for resourceName, reqAmount := range containerReqs {
			sumOfLongRunningExclusiveMemory[resourceName] += reqAmount
		}
	}

	// Check for empty shared pool for concurrently running containers (app and restartable init containers).
	for _, container := range pod.Spec.Containers {
		containerReqs, err := getContainerRequestedResources(logger, pod, &container)
		if err != nil {
			return err
		}

		if len(containerReqs) == 0 {
			hasSharedLongRunningContainers = true
			continue
		}

		for resourceName, val := range containerReqs {
			sumOfLongRunningExclusiveMemory[resourceName] += val
		}
	}

	// This check ensures that if there are any standard containers that need a shared
	// pool, that pool is not empty. An empty pool would occur if the pod budget is
	// fully consumed by sidecars and main containers with exclusive resources.
	if hasSharedLongRunningContainers {
		for resourceName, totalPodResource := range podTotalMemory {
			if sumOfLongRunningExclusiveMemory[resourceName] >= totalPodResource {
				return fmt.Errorf("pod rejected, sum of exclusive container %s requests equals pod budget, leaving no memory for shared containers", resourceName)
			}
		}
	}

	return nil
}

// AllocatePod is the entry point for pod-level resource allocation.
// It's called once per pod by the Topology Manager's pod-scope admit handler.
// The logic here allocates a single NUMA-aligned "bubble" of memory for the
// entire pod. All containers within the pod will share this NUMA binding.
func (p *staticPolicy) AllocatePod(logger klog.Logger, s state.State, pod *v1.Pod) (rerr error) {
	logger = klog.LoggerWithValues(logger, "pod", klog.KObj(pod))
	logger.V(4).Info("AllocatePod called for pod-level managed pod")

	qos := v1qos.GetPodQOS(pod)
	if qos != v1.PodQOSGuaranteed {
		return nil
	}

	// 1. Calculate memory resources.
	podTotalMemory, err := getPodRequestedResources(logger, pod)
	if err != nil {
		logger.Error(err, "Failed to get pod requested resources", "podUID", pod.UID)
		return nil
	}
	logger.V(4).Info("Calculated total pod memory", "podTotalMemory", podTotalMemory)

	metrics.MemoryManagerPinningRequestTotal.Inc()
	defer func() {
		if rerr != nil {
			metrics.MemoryManagerPinningErrorsTotal.Inc()
			metrics.ResourceManagerAllocationErrorsTotal.WithLabelValues(metrics.ResourceManagerMemory, metrics.ResourceManagerPod).Inc()
		}
	}()

	// 2. Validate for the "empty shared pool" case.
	if err := p.validatePodScopeResources(logger, pod); err != nil {
		return admission.NewEmptyPodSharedPoolError(err)
	}

	// 3. Handle hints and NUMA alignment.
	machineState := s.GetMachineState()
	bestHint := p.affinity.GetAffinity(string(pod.UID), append(pod.Spec.InitContainers, pod.Spec.Containers...)[0].Name)
	if bestHint.NUMANodeAffinity == nil {
		defaultHint, err := p.getDefaultHint(machineState, pod, podTotalMemory)
		if err != nil {
			return err
		}
		if !defaultHint.Preferred && bestHint.Preferred {
			return fmt.Errorf("[memorymanager] failed to find the default preferred hint")
		}
		bestHint = *defaultHint
	}

	if !isAffinitySatisfyRequest(machineState, bestHint.NUMANodeAffinity, podTotalMemory) {
		extendedHint, err := p.extendTopologyManagerHint(machineState, pod, podTotalMemory, bestHint.NUMANodeAffinity)
		if err != nil {
			return err
		}
		if !extendedHint.Preferred && bestHint.Preferred {
			return fmt.Errorf("[memorymanager] failed to find the extended preferred hint")
		}
		bestHint = *extendedHint
	}

	if isAffinityViolatingNUMAAllocations(machineState, bestHint.NUMANodeAffinity) {
		return fmt.Errorf("[memorymanager] preferred hint violates NUMA node allocation")
	}

	maskBits := bestHint.NUMANodeAffinity.GetBits()

	// 4. Partition the pod's memory allocation, handling init container reuse correctly.
	exclusiveMemory := make(map[string][]state.Block)
	sidecarMemory := make(map[v1.ResourceName]uint64)

	// First, iterate through all init containers and allocate their memory.
	for _, c := range pod.Spec.InitContainers {
		reqs, err := getContainerRequestedResources(logger, pod, &c)
		if err != nil {
			return err
		}

		metrics.MemoryManagerPinningRequestTotal.Inc()

		// The pool available for this init container is the entire pod allocation
		// minus what's already taken by sidecars.
		runnablePool := make(map[v1.ResourceName]uint64)
		for resourceName, total := range podTotalMemory {
			runnablePool[resourceName] = total - sidecarMemory[resourceName]
		}

		if len(reqs) == 0 {
			// Restartable init containers will access the main pod shared pool.
			if podutil.IsRestartableInitContainer(&c) {
				continue
			}

			// Non restartable init containers will have access to the whole pod pool
			// amount minus the sidecar exclusive memory up to that point.
			blocks := []state.Block{}
			for resourceName, req := range runnablePool {
				if req > 0 {
					blocks = append(blocks, state.Block{
						NUMAAffinity: maskBits,
						Size:         req,
						Type:         resourceName,
					})
				}
			}
			exclusiveMemory[c.Name] = blocks
			metrics.ResourceManagerContainerAssignments.WithLabelValues(metrics.ResourceManagerMemory, metrics.ResourceManagerSharedPod).Inc()
			continue
		}

		// Allocate blocks
		blocks := []state.Block{}
		for resourceName, req := range reqs {
			if req > 0 {
				blocks = append(blocks, state.Block{
					NUMAAffinity: maskBits,
					Size:         req,
					Type:         resourceName,
				})
			}
		}
		exclusiveMemory[c.Name] = blocks
		metrics.ResourceManagerContainerAssignments.WithLabelValues(metrics.ResourceManagerMemory, metrics.ResourceManagerExclusivePod).Inc()

		// If it's a restartable sidecar, its memory is permanently consumed from the pod's budget.
		if podutil.IsRestartableInitContainer(&c) {
			for resourceName, req := range reqs {
				sidecarMemory[resourceName] += req
			}
		}
	}

	// Recalculate the pool available for app containers.
	// This is the total pod budget minus what all sidecars are using.
	appContainerPool := make(map[v1.ResourceName]uint64)
	for resourceName, total := range podTotalMemory {
		appContainerPool[resourceName] = total - sidecarMemory[resourceName]
	}

	// Second, iterate through regular containers, allocating from the remaining pool.
	for _, c := range pod.Spec.Containers {
		reqs, err := getContainerRequestedResources(logger, pod, &c)
		if err != nil {
			// This should not happen, as we have already validated the pod's resources.
			return err
		}

		// A container with no memory requests will be part of the shared pool.
		if len(reqs) == 0 {
			continue
		}
		metrics.MemoryManagerPinningRequestTotal.Inc()

		// Allocate blocks and consume from the pool
		blocks := []state.Block{}
		for resourceName, req := range reqs {
			if req > 0 {
				blocks = append(blocks, state.Block{
					NUMAAffinity: maskBits,
					Size:         req,
					Type:         resourceName,
				})
				appContainerPool[resourceName] -= req
			}
		}
		exclusiveMemory[c.Name] = blocks
		metrics.ResourceManagerContainerAssignments.WithLabelValues(metrics.ResourceManagerMemory, metrics.ResourceManagerExclusivePod).Inc()
	}

	// The final shared pool is what remains after all exclusive allocations.
	podSharedPoolBlocks := []state.Block{}
	for resourceName, val := range appContainerPool {
		if val > 0 {
			podSharedPoolBlocks = append(podSharedPoolBlocks, state.Block{
				NUMAAffinity: maskBits,
				Size:         val,
				Type:         resourceName,
			})
		}
	}

	podAllocation := []state.Block{}
	for resourceName, requestedSize := range podTotalMemory {
		podAllocation = append(podAllocation, state.Block{
			NUMAAffinity: maskBits,
			Size:         requestedSize,
			Type:         resourceName,
		})
	}
	s.SetPodMemoryBlocks(string(pod.UID), podAllocation)

	// 5. Now that partitioning is successful, update the global machine state.
	for resourceName, requestedSize := range podTotalMemory {
		p.updateMachineState(machineState, maskBits, resourceName, requestedSize)
	}
	logger.V(4).Info("Partitioned pod-level memory allocation", "exclusiveMemory", podMemoryAllocationToString(exclusiveMemory), "podSharedPool", memoryBlocksToString(podSharedPoolBlocks))

	// 6. Save all container assignments to the state.
	podUID := string(pod.UID)
	for _, c := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
		if blocks, isExclusive := exclusiveMemory[c.Name]; isExclusive {
			s.SetMemoryBlocks(podUID, c.Name, blocks)
		} else {
			s.SetMemoryBlocks(podUID, c.Name, podSharedPoolBlocks)
			metrics.ResourceManagerContainerAssignments.WithLabelValues(metrics.ResourceManagerMemory, metrics.ResourceManagerSharedPod).Inc()
		}

		metrics.ResourceManagerAllocationsTotal.WithLabelValues(metrics.ResourceManagerMemory, metrics.ResourceManagerPod).Inc()
	}

	s.SetMachineState(machineState)
	return nil
}

func podMemoryAllocationToString(alloc map[string][]state.Block) string {
	if len(alloc) == 0 {
		return ""
	}
	var sb strings.Builder
	for name, blocks := range alloc {
		fmt.Fprintf(&sb, ",%s=%s", name, memoryBlocksToString(blocks))
	}
	// if we have gone so far, it means we have at least 1 alloc, so the cut is safe
	repr := sb.String()[1:]
	return "<" + repr + ">"
}

func memoryBlocksToString(blocks []state.Block) string {
	if len(blocks) == 0 {
		return "[]"
	}
	var sb strings.Builder
	for _, block := range blocks {
		// Format: {Type:SizeMi @ [NUMA_Nodes]}
		fmt.Fprintf(&sb, ",{%s:%dMi @ %v}", block.Type, block.Size/(1024*1024), block.NUMAAffinity)
	}
	// Cut the leading comma
	repr := sb.String()[1:]
	return "[" + repr + "]"
}

// Allocate call is idempotent
func (p *staticPolicy) Allocate(logger klog.Logger, s state.State, pod *v1.Pod, container *v1.Container) (rerr error) {
	// allocate the memory only for guaranteed pods
	logger = klog.LoggerWithValues(logger, "pod", klog.KObj(pod), "containerName", container.Name)

	podUID := string(pod.UID)
	// Allocate the memory only for guaranteed pods
	qos := v1qos.GetPodQOS(pod)
	if qos != v1.PodQOSGuaranteed {
		logger.V(5).Info("Exclusive memory allocation skipped, pod QoS is not guaranteed", "qos", qos)
		return nil
	}

	if (!utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) || !utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources)) && resourcehelper.IsPodLevelResourcesSet(pod) {
		logger.V(2).Info("Allocation skipped, pod is using pod-level resources which are not supported by the static Memory manager policy", "podUID", podUID)
		return nil
	}

	logger.Info("Allocate")
	// Container belongs in an exclusively allocated pool
	metrics.MemoryManagerPinningRequestTotal.Inc()
	defer func() {
		if rerr != nil {
			metrics.MemoryManagerPinningErrorsTotal.Inc()
			metrics.ResourceManagerAllocationErrorsTotal.WithLabelValues(metrics.ResourceManagerMemory, metrics.ResourceManagerNode).Inc()
		}
	}()
	if blocks := s.GetMemoryBlocks(podUID, container.Name); blocks != nil {
		p.updatePodReusableMemory(pod, container, blocks)

		logger.Info("Container already present in state, skipping")
		return nil
	}

	// Call Topology Manager to get the aligned affinity across all hint providers.
	hint := p.affinity.GetAffinity(podUID, container.Name)
	logger.Info("Got topology affinity", "hint", hint)

	requestedResources, err := getContainerRequestedResources(logger, pod, container)
	if err != nil {
		return err
	}

	machineState := s.GetMachineState()
	bestHint := &hint
	// topology manager returned the hint with NUMA affinity nil
	// we should use the default NUMA affinity calculated the same way as for the topology manager
	if hint.NUMANodeAffinity == nil {
		defaultHint, err := p.getDefaultHint(machineState, pod, requestedResources)
		if err != nil {
			return err
		}

		if !defaultHint.Preferred && bestHint.Preferred {
			return fmt.Errorf("[memorymanager] failed to find the default preferred hint")
		}
		bestHint = defaultHint
	}

	// topology manager returns the hint that does not satisfy completely the container request
	// we should extend this hint to the one who will satisfy the request and include the current hint
	if !isAffinitySatisfyRequest(machineState, bestHint.NUMANodeAffinity, requestedResources) {
		extendedHint, err := p.extendTopologyManagerHint(machineState, pod, requestedResources, bestHint.NUMANodeAffinity)
		if err != nil {
			return err
		}

		if !extendedHint.Preferred && bestHint.Preferred {
			return fmt.Errorf("[memorymanager] failed to find the extended preferred hint")
		}
		bestHint = extendedHint
	}

	// the best hint might violate the NUMA allocation rule on which
	// NUMA node cannot have both single and cross NUMA node allocations
	// https://kubernetes.io/blog/2021/08/11/kubernetes-1-22-feature-memory-manager-moves-to-beta/#single-vs-cross-numa-node-allocation
	if isAffinityViolatingNUMAAllocations(machineState, bestHint.NUMANodeAffinity) {
		return fmt.Errorf("[memorymanager] preferred hint violates NUMA node allocation")
	}

	var containerBlocks []state.Block
	maskBits := bestHint.NUMANodeAffinity.GetBits()
	for resourceName, requestedSize := range requestedResources {
		// update memory blocks
		containerBlocks = append(containerBlocks, state.Block{
			NUMAAffinity: maskBits,
			Size:         requestedSize,
			Type:         resourceName,
		})

		podReusableMemory := p.getPodReusableMemory(pod, bestHint.NUMANodeAffinity, resourceName)
		if podReusableMemory >= requestedSize {
			requestedSize = 0
		} else {
			requestedSize -= podReusableMemory
		}

		// Update nodes memory state
		p.updateMachineState(machineState, maskBits, resourceName, requestedSize)
		metrics.ResourceManagerAllocationsTotal.WithLabelValues(metrics.ResourceManagerMemory, metrics.ResourceManagerNode).Inc()
	}

	p.updatePodReusableMemory(pod, container, containerBlocks)

	s.SetMachineState(machineState)
	s.SetMemoryBlocks(podUID, container.Name, containerBlocks)

	// update init containers memory blocks to reflect the fact that we re-used init containers memory
	// it is possible that the size of the init container memory block will have 0 value, when all memory
	// allocated for it was re-used
	// we only do this so that the sum(memory_for_all_containers) == total amount of allocated memory to the pod, even
	// though the final state here doesn't accurately reflect what was (in reality) allocated to each container
	// TODO: we should refactor our state structs to reflect the amount of the re-used memory
	p.updateInitContainersMemoryBlocks(logger, s, pod, container, containerBlocks)
	metrics.ResourceManagerContainerAssignments.WithLabelValues(metrics.ResourceManagerMemory, metrics.ResourceManagerExclusiveNode).Inc()

	logger.V(4).Info("Allocated exclusive memory")
	return nil
}

func (p *staticPolicy) updateMachineState(machineState state.NUMANodeMap, numaAffinity []int, resourceName v1.ResourceName, requestedSize uint64) {
	for _, nodeID := range numaAffinity {
		machineState[nodeID].NumberOfAssignments++
		machineState[nodeID].Cells = numaAffinity

		// we need to continue to update all affinity mask nodes
		if requestedSize == 0 {
			continue
		}

		// update the node memory state
		nodeResourceMemoryState := machineState[nodeID].MemoryMap[resourceName]
		if nodeResourceMemoryState.Free <= 0 {
			continue
		}

		// the node has enough memory to satisfy the request
		if nodeResourceMemoryState.Free >= requestedSize {
			nodeResourceMemoryState.Reserved += requestedSize
			nodeResourceMemoryState.Free -= requestedSize
			requestedSize = 0
			continue
		}

		// the node does not have enough memory, use the node remaining memory and move to the next node
		requestedSize -= nodeResourceMemoryState.Free
		nodeResourceMemoryState.Reserved += nodeResourceMemoryState.Free
		nodeResourceMemoryState.Free = 0
	}
}

func (p *staticPolicy) getPodReusableMemory(pod *v1.Pod, numaAffinity bitmask.BitMask, resourceName v1.ResourceName) uint64 {
	podReusableMemory, ok := p.initContainersReusableMemory[string(pod.UID)]
	if !ok {
		return 0
	}

	numaReusableMemory, ok := podReusableMemory[numaAffinity.String()]
	if !ok {
		return 0
	}

	return numaReusableMemory[resourceName]
}

// RemoveContainer call is idempotent
func (p *staticPolicy) RemoveContainer(logger klog.Logger, s state.State, podUID string, containerName string) {
	logger = klog.LoggerWithValues(logger, "podUID", podUID, "containerName", containerName)
	logger.Info("RemoveContainer", "podUID", podUID, "containerName", containerName)

	// Check if this pod is managed with pod-level memory.
	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) {
		if podMemory := s.GetPodMemoryBlocks(podUID); podMemory != nil {
			s.Delete(podUID, containerName)
			// If this was the last container for the pod, then release the entire pod-level memory.
			if len(s.GetMemoryAssignments()[podUID]) == 0 {
				p.releaseMemory(s, podMemory)
				s.DeletePod(podUID) // Clean up all state for the pod.
				logger.Info("Released pod-level memory")
			}
			// If other containers still exist, do not release any memory yet.
			// The pod-level memory will be released when the last container is removed.
			return
		}
	}

	blocks := s.GetMemoryBlocks(podUID, containerName)
	if blocks == nil {
		return
	}
	s.Delete(podUID, containerName)

	p.releaseMemory(s, blocks)
	logger.Info("Released container-level memory")
}

func (p *staticPolicy) releaseMemory(s state.State, blocks []state.Block) {
	// Mutate machine memory state to update free and reserved memory
	machineState := s.GetMachineState()
	for _, b := range blocks {
		releasedSize := b.Size
		for _, nodeID := range b.NUMAAffinity {
			machineState[nodeID].NumberOfAssignments--

			// once we do not have any memory allocations on this node, clear node groups
			if machineState[nodeID].NumberOfAssignments == 0 {
				machineState[nodeID].Cells = []int{nodeID}
			}

			// we still need to pass over all NUMA node under the affinity mask to update them
			if releasedSize == 0 {
				continue
			}

			nodeResourceMemoryState := machineState[nodeID].MemoryMap[b.Type]

			// if the node does not have reserved memory to free, continue to the next node
			if nodeResourceMemoryState.Reserved == 0 {
				continue
			}

			// the reserved memory smaller than the amount of the memory that should be released
			// release as much as possible and move to the next node
			if nodeResourceMemoryState.Reserved < releasedSize {
				releasedSize -= nodeResourceMemoryState.Reserved
				nodeResourceMemoryState.Free += nodeResourceMemoryState.Reserved
				nodeResourceMemoryState.Reserved = 0
				continue
			}

			// the reserved memory big enough to satisfy the released memory
			nodeResourceMemoryState.Free += releasedSize
			nodeResourceMemoryState.Reserved -= releasedSize
			releasedSize = 0
		}
	}

	s.SetMachineState(machineState)
}

func regenerateHints(logger klog.Logger, pod *v1.Pod, ctn *v1.Container, ctnBlocks []state.Block, reqRsrc map[v1.ResourceName]uint64) map[string][]topologymanager.TopologyHint {
	hints := map[string][]topologymanager.TopologyHint{}
	for resourceName := range reqRsrc {
		hints[string(resourceName)] = []topologymanager.TopologyHint{}
	}

	if len(ctnBlocks) != len(reqRsrc) {
		logger.Info("The number of requested resources by the container differs from the number of memory blocks", "containerName", ctn.Name)
		return nil
	}

	for _, b := range ctnBlocks {
		if _, ok := reqRsrc[b.Type]; !ok {
			logger.Info("Container requested resources but none available of this type", "containerName", ctn.Name, "type", b.Type)
			return nil
		}

		if b.Size != reqRsrc[b.Type] {
			logger.Info("Memory already allocated with different numbers than requested", "containerName", ctn.Name, "type", b.Type, "requestedResource", reqRsrc[b.Type], "allocatedSize", b.Size)
			return nil
		}

		containerNUMAAffinity, err := bitmask.NewBitMask(b.NUMAAffinity...)
		if err != nil {
			logger.Error(err, "Failed to generate NUMA bitmask", "containerName", ctn.Name, "type", b.Type)
			return nil
		}

		logger.Info("Regenerating TopologyHints, resource was already allocated to pod", "resourceName", b.Type, "podUID", pod.UID, "containerName", ctn.Name)
		hints[string(b.Type)] = append(hints[string(b.Type)], topologymanager.TopologyHint{
			NUMANodeAffinity: containerNUMAAffinity,
			Preferred:        true,
		})
	}
	return hints
}

func getPodRequestedResources(logger logr.Logger, pod *v1.Pod) (map[v1.ResourceName]uint64, error) {
	// If pod-level resources are set, use them directly.
	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) && resourcehelper.IsPodLevelResourcesSet(pod) {
		requestedResources := make(map[v1.ResourceName]uint64)
		for resourceName, quantity := range pod.Spec.Resources.Requests {
			if resourceName != v1.ResourceMemory && !corehelper.IsHugePageResourceName(resourceName) {
				continue
			}
			requestedSize, succeed := quantity.AsInt64()
			if !succeed {
				return nil, fmt.Errorf("[memorymanager] failed to represent quantity as int64")
			}
			if requestedSize <= 0 {
				continue
			}
			requestedResources[resourceName] = uint64(requestedSize)
		}
		return requestedResources, nil
	}

	// Maximun resources requested by init containers at any given time.
	reqRsrcsByInitCtrs := make(map[v1.ResourceName]uint64)
	// Total resources requested by restartable init containers.
	reqRsrcsByRestartableInitCtrs := make(map[v1.ResourceName]uint64)
	for _, ctr := range pod.Spec.InitContainers {
		reqRsrcs, err := getContainerRequestedResources(logger, pod, &ctr)

		if err != nil {
			return nil, err
		}
		for rsrcName, qty := range reqRsrcs {
			if _, ok := reqRsrcsByInitCtrs[rsrcName]; !ok {
				reqRsrcsByInitCtrs[rsrcName] = uint64(0)
			}

			// See https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/753-sidecar-containers#resources-calculation-for-scheduling-and-pod-admission
			// for the detail.
			if podutil.IsRestartableInitContainer(&ctr) {
				reqRsrcsByRestartableInitCtrs[rsrcName] += qty
			} else if reqRsrcsByRestartableInitCtrs[rsrcName]+qty > reqRsrcsByInitCtrs[rsrcName] {
				reqRsrcsByInitCtrs[rsrcName] = reqRsrcsByRestartableInitCtrs[rsrcName] + qty
			}
		}
	}

	reqRsrcsByAppCtrs := make(map[v1.ResourceName]uint64)
	for _, ctr := range pod.Spec.Containers {
		reqRsrcs, err := getContainerRequestedResources(logger, pod, &ctr)

		if err != nil {
			return nil, err
		}
		for rsrcName, qty := range reqRsrcs {
			if _, ok := reqRsrcsByAppCtrs[rsrcName]; !ok {
				reqRsrcsByAppCtrs[rsrcName] = uint64(0)
			}

			reqRsrcsByAppCtrs[rsrcName] += qty
		}
	}

	reqRsrcs := make(map[v1.ResourceName]uint64)
	for rsrcName := range reqRsrcsByAppCtrs {
		// Total resources requested by long-running containers.
		reqRsrcsByLongRunningCtrs := reqRsrcsByAppCtrs[rsrcName] + reqRsrcsByRestartableInitCtrs[rsrcName]
		reqRsrcs[rsrcName] = reqRsrcsByLongRunningCtrs

		if reqRsrcs[rsrcName] < reqRsrcsByInitCtrs[rsrcName] {
			reqRsrcs[rsrcName] = reqRsrcsByInitCtrs[rsrcName]
		}
	}
	return reqRsrcs, nil
}

func (p *staticPolicy) GetPodTopologyHints(logger klog.Logger, s state.State, pod *v1.Pod) map[string][]topologymanager.TopologyHint {
	logger = klog.LoggerWithValues(logger, "pod", klog.KObj(pod))

	if v1qos.GetPodQOS(pod) != v1.PodQOSGuaranteed {
		return nil
	}

	// Get a count of how much memory has been requested by Pod.
	reqRsrcs, err := getPodRequestedResources(logger, pod)
	if err != nil {
		logger.Error(err, "Failed to get pod requested resources", "podUID", pod.UID)
		return nil
	}

	// If the pod has pod-level resources but the feature gate is disabled,
	// log it and return nil hints to admit the pod without alignment.
	if (!utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) || !utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources)) && resourcehelper.IsPodLevelResourcesSet(pod) {
		logger.V(3).Info("Memory Manager pod hint generation skipped, pod is using pod-level resources but the PodLevelResourceManagers feature gate is not enabled", "podUID", pod.UID)
		return nil
	}

	// Validate that if a pod has containers that will be placed in a shared pool,
	// there are actually memory left over for that pool after accounting for all
	// exclusive allocations. If the sum of exclusive memory requests consumes the
	// entire pod-level memory budget, no hints will be generated, causing the pod
	// to be rejected by the Topology Manager.
	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) && resourcehelper.IsPodLevelResourcesSet(pod) {
		if err := p.validatePodScopeResources(logger, pod); err != nil {
			logger.V(2).Info("Pod admission will fail", "error", err)
			hints := make(map[string][]topologymanager.TopologyHint)
			for resourceName := range reqRsrcs {
				hints[string(resourceName)] = []topologymanager.TopologyHint{}
			}
			return hints
		}
	}

	if len(reqRsrcs) == 0 {
		return nil
	}

	for _, ctn := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
		containerBlocks := s.GetMemoryBlocks(string(pod.UID), ctn.Name)
		// Short circuit to regenerate the same hints if there are already
		// memory allocated for the container. This might happen after a
		// kubelet restart, for example.
		if containerBlocks != nil {
			return regenerateHints(logger, pod, &ctn, containerBlocks, reqRsrcs)
		}
	}

	// the pod topology hints calculated only once for all containers, so no need to pass re-usable state
	return p.calculateHints(s.GetMachineState(), pod, reqRsrcs)
}

// GetTopologyHints implements the topologymanager.HintProvider Interface
// and is consulted to achieve NUMA aware resource alignment among this
// and other resource controllers.
func (p *staticPolicy) GetTopologyHints(logger klog.Logger, s state.State, pod *v1.Pod, container *v1.Container) map[string][]topologymanager.TopologyHint {
	logger = klog.LoggerWithValues(logger, "pod", klog.KObj(pod))

	if v1qos.GetPodQOS(pod) != v1.PodQOSGuaranteed {
		return nil
	}

	requestedResources, err := getContainerRequestedResources(logger, pod, container)
	if err != nil {
		logger.Error(err, "Failed to get container requested resources", "podUID", pod.UID, "containerName", container.Name)
		return nil
	}

	// If the pod has pod-level resources but the feature gate is disabled,
	// log it and return nil hints to admit the pod without alignment.
	if (!utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) || !utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources)) && resourcehelper.IsPodLevelResourcesSet(pod) {
		logger.V(3).Info("Memory Manager pod hint generation skipped, pod is using pod-level resources but the PodLevelResourceManagers feature gate is not enabled", "podUID", pod.UID)
		return nil
	}

	containerBlocks := s.GetMemoryBlocks(string(pod.UID), container.Name)
	// Short circuit to regenerate the same hints if there are already
	// memory allocated for the container. This might happen after a
	// kubelet restart, for example.
	if containerBlocks != nil {
		return regenerateHints(logger, pod, container, containerBlocks, requestedResources)
	}

	return p.calculateHints(s.GetMachineState(), pod, requestedResources)
}

func getContainerRequestedResources(logger logr.Logger, pod *v1.Pod, container *v1.Container) (map[v1.ResourceName]uint64, error) {
	requestedResources := map[v1.ResourceName]uint64{}
	// For pod-level resource management, a container is only considered for exclusive
	// memory if its request equals its limit for both the CPU and Memory. This
	// aligns with the Guaranteed QoS requirement for container-level resources.
	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) && resourcehelper.IsPodLevelResourcesSet(pod) && !cmqos.IsContainerEquivalentQOSGuaranteed(container) {
		logger.V(5).Info("Exclusive Memory allocation skipped, container is not eligible, request and limit are not equal", "pod", klog.KObj(pod), "containerName", container.Name)
		return requestedResources, nil
	}

	for resourceName, quantity := range container.Resources.Requests {
		if resourceName != v1.ResourceMemory && !corehelper.IsHugePageResourceName(resourceName) {
			continue
		}

		requestedSize, succeed := quantity.AsInt64()
		if !succeed {
			return nil, fmt.Errorf("[memorymanager] failed to represent quantity as int64")
		}
		requestedResources[resourceName] = uint64(requestedSize)
	}
	return requestedResources, nil
}

func (p *staticPolicy) calculateHints(machineState state.NUMANodeMap, pod *v1.Pod, requestedResources map[v1.ResourceName]uint64) map[string][]topologymanager.TopologyHint {
	var numaNodes []int
	for n := range machineState {
		numaNodes = append(numaNodes, n)
	}
	sort.Ints(numaNodes)

	// Initialize minAffinitySize to include all NUMA Cells.
	minAffinitySize := len(numaNodes)

	hints := map[string][]topologymanager.TopologyHint{}
	bitmask.IterateBitMasks(numaNodes, func(mask bitmask.BitMask) {
		maskBits := mask.GetBits()
		singleNUMAHint := len(maskBits) == 1

		totalFreeSize := map[v1.ResourceName]uint64{}
		totalAllocatableSize := map[v1.ResourceName]uint64{}
		// calculate total free and allocatable memory for the node mask
		for _, nodeID := range maskBits {
			for resourceName := range requestedResources {
				if _, ok := totalFreeSize[resourceName]; !ok {
					totalFreeSize[resourceName] = 0
				}
				totalFreeSize[resourceName] += machineState[nodeID].MemoryMap[resourceName].Free

				if _, ok := totalAllocatableSize[resourceName]; !ok {
					totalAllocatableSize[resourceName] = 0
				}
				totalAllocatableSize[resourceName] += machineState[nodeID].MemoryMap[resourceName].Allocatable
			}
		}

		// verify that for all memory types the node mask has enough allocatable resources
		for resourceName, requestedSize := range requestedResources {
			if totalAllocatableSize[resourceName] < requestedSize {
				return
			}
		}

		// set the minimum amount of NUMA nodes that can satisfy the container resources requests
		if mask.Count() < minAffinitySize {
			minAffinitySize = mask.Count()
		}

		// the node already in group with another node, it can not be used for the single NUMA node allocation
		if singleNUMAHint && len(machineState[maskBits[0]].Cells) > 1 {
			return
		}

		for _, nodeID := range maskBits {
			// the node already used for the memory allocation
			if !singleNUMAHint && machineState[nodeID].NumberOfAssignments > 0 {
				// the node used for the single NUMA memory allocation, it can not be used for the multi NUMA node allocation
				if len(machineState[nodeID].Cells) == 1 {
					return
				}

				// the node already used with different group of nodes, it can not be use with in the current hint
				if !areGroupsEqual(machineState[nodeID].Cells, maskBits) {
					return
				}
			}
		}

		// verify that for all memory types the node mask has enough free resources
		for resourceName, requestedSize := range requestedResources {
			podReusableMemory := p.getPodReusableMemory(pod, mask, resourceName)
			if totalFreeSize[resourceName]+podReusableMemory < requestedSize {
				return
			}
		}

		// add the node mask as topology hint for all memory types
		for resourceName := range requestedResources {
			if _, ok := hints[string(resourceName)]; !ok {
				hints[string(resourceName)] = []topologymanager.TopologyHint{}
			}
			hints[string(resourceName)] = append(hints[string(resourceName)], topologymanager.TopologyHint{
				NUMANodeAffinity: mask,
				Preferred:        false,
			})
		}
	})

	// update hints preferred according to multiNUMAGroups, in case when it wasn't provided, the default
	// behaviour to prefer the minimal amount of NUMA nodes will be used
	for resourceName := range requestedResources {
		for i, hint := range hints[string(resourceName)] {
			hints[string(resourceName)][i].Preferred = p.isHintPreferred(hint.NUMANodeAffinity.GetBits(), minAffinitySize)
		}
	}

	return hints
}

func (p *staticPolicy) isHintPreferred(maskBits []int, minAffinitySize int) bool {
	return len(maskBits) == minAffinitySize
}

func areGroupsEqual(group1, group2 []int) bool {
	sort.Ints(group1)
	sort.Ints(group2)

	if len(group1) != len(group2) {
		return false
	}

	for i, elm := range group1 {
		if group2[i] != elm {
			return false
		}
	}
	return true
}

func (p *staticPolicy) validateState(logger klog.Logger, s state.State) error {
	machineState := s.GetMachineState()
	memoryAssignments := s.GetMemoryAssignments()

	if len(machineState) == 0 {
		// Machine state cannot be empty when assignments exist
		if len(memoryAssignments) != 0 {
			return fmt.Errorf("[memorymanager] machine state can not be empty when it has memory assignments")
		}

		defaultMachineState := p.getDefaultMachineState()
		s.SetMachineState(defaultMachineState)

		return nil
	}

	// calculate all memory assigned to containers
	expectedMachineState := p.getDefaultMachineState()
	for pod, container := range memoryAssignments {
		if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) {
			if podBlocks := s.GetPodMemoryBlocks(pod); len(podBlocks) > 0 {
				if err := p.updateExpectedMachineState(expectedMachineState, podBlocks, pod, ""); err != nil {
					return err
				}
				continue
			}
		}

		for containerName, blocks := range container {
			if err := p.updateExpectedMachineState(expectedMachineState, blocks, pod, containerName); err != nil {
				return err
			}
		}
	}

	// State has already been initialized from file (is not empty)
	// Validate that total size, system reserved and reserved memory not changed, it can happen, when:
	// - adding or removing physical memory bank from the node
	// - change of kubelet system-reserved, kube-reserved or pre-reserved-memory-zone parameters
	if !areMachineStatesEqual(logger, machineState, expectedMachineState) {
		return fmt.Errorf("[memorymanager] the expected machine state is different from the real one")
	}

	return nil
}

func (p *staticPolicy) updateExpectedMachineState(expectedMachineState state.NUMANodeMap, blocks []state.Block, podUID, containerName string) error {
	for _, b := range blocks {
		requestedSize := b.Size
		for _, nodeID := range b.NUMAAffinity {
			nodeState, ok := expectedMachineState[nodeID]
			if !ok {
				return fmt.Errorf("[memorymanager] (pod: %s, container: %s) the memory assignment uses the NUMA that does not exist", podUID, containerName)
			}

			nodeState.NumberOfAssignments++
			nodeState.Cells = b.NUMAAffinity

			memoryState, ok := nodeState.MemoryMap[b.Type]
			if !ok {
				return fmt.Errorf("[memorymanager] (pod: %s, container: %s) the memory assignment uses memory resource that does not exist", podUID, containerName)
			}

			if requestedSize == 0 {
				continue
			}

			// this node does not have enough memory continue to the next one
			if memoryState.Free <= 0 {
				continue
			}

			// the node has enough memory to satisfy the request
			if memoryState.Free >= requestedSize {
				memoryState.Reserved += requestedSize
				memoryState.Free -= requestedSize
				requestedSize = 0
				continue
			}

			// the node does not have enough memory, use the node remaining memory and move to the next node
			requestedSize -= memoryState.Free
			memoryState.Reserved += memoryState.Free
			memoryState.Free = 0
		}
	}
	return nil
}

func areMachineStatesEqual(logger klog.Logger, ms1, ms2 state.NUMANodeMap) bool {
	if len(ms1) != len(ms2) {
		logger.Info("Node states were different", "lengthNode1", len(ms1), "lengthNode2", len(ms2))
		return false
	}

	for nodeID, nodeState1 := range ms1 {
		nodeState2, ok := ms2[nodeID]
		if !ok {
			logger.Info("Node state didn't have node ID", "nodeID", nodeID)
			return false
		}

		if nodeState1.NumberOfAssignments != nodeState2.NumberOfAssignments {
			logger.Info("Node state had a different number of memory assignments.", "assignment1", nodeState1.NumberOfAssignments, "assignment2", nodeState2.NumberOfAssignments)
			return false
		}

		if !areGroupsEqual(nodeState1.Cells, nodeState2.Cells) {
			logger.Info("Node states had different groups", "stateNode1", nodeState1.Cells, "stateNode2", nodeState2.Cells)
			return false
		}

		if len(nodeState1.MemoryMap) != len(nodeState2.MemoryMap) {
			logger.Info("Node state had memory maps of different lengths", "lengthNode1", len(nodeState1.MemoryMap), "lengthNode2", len(nodeState2.MemoryMap))
			return false
		}

		for resourceName, memoryState1 := range nodeState1.MemoryMap {
			memoryState2, ok := nodeState2.MemoryMap[resourceName]
			if !ok {
				logger.Info("Memory state didn't have resource", "resource", resourceName)
				return false
			}

			if !areMemoryStatesEqual(logger, memoryState1, memoryState2, nodeID, resourceName) {
				return false
			}

			tmpState1 := state.MemoryTable{}
			tmpState2 := state.MemoryTable{}
			for _, nodeID := range nodeState1.Cells {
				tmpState1.Free += ms1[nodeID].MemoryMap[resourceName].Free
				tmpState1.Reserved += ms1[nodeID].MemoryMap[resourceName].Reserved
				tmpState2.Free += ms2[nodeID].MemoryMap[resourceName].Free
				tmpState2.Reserved += ms2[nodeID].MemoryMap[resourceName].Reserved
			}

			if tmpState1.Free != tmpState2.Free {
				logger.Info("NUMA node and resource had different memory states", "node", nodeID, "resource", resourceName, "field", "free", "free1", tmpState1.Free, "free2", tmpState2.Free, "memoryState1", *memoryState1, "memoryState2", *memoryState2)
				return false
			}
			if tmpState1.Reserved != tmpState2.Reserved {
				logger.Info("NUMA node and resource had different memory states", "node", nodeID, "resource", resourceName, "field", "reserved", "reserved1", tmpState1.Reserved, "reserved2", tmpState2.Reserved, "memoryState1", *memoryState1, "memoryState2", *memoryState2)
				return false
			}
		}
	}
	return true
}

func areMemoryStatesEqual(logger klog.Logger, memoryState1, memoryState2 *state.MemoryTable, nodeID int, resourceName v1.ResourceName) bool {
	loggerWithValues := klog.LoggerWithValues(logger, "node", nodeID, "resource", resourceName, "memoryState1", *memoryState1, "memoryState2", *memoryState2)
	if memoryState1.TotalMemSize != memoryState2.TotalMemSize {
		logger.Info("Memory states for the NUMA node and resource are different", "field", "TotalMemSize", "TotalMemSize1", memoryState1.TotalMemSize, "TotalMemSize2", memoryState2.TotalMemSize)
		return false
	}

	if memoryState1.SystemReserved != memoryState2.SystemReserved {
		loggerWithValues.Info("Memory states for the NUMA node and resource are different", "field", "SystemReserved", "SystemReserved1", memoryState1.SystemReserved, "SystemReserved2", memoryState2.SystemReserved)
		return false
	}

	if memoryState1.Allocatable != memoryState2.Allocatable {
		loggerWithValues.Info("Memory states for the NUMA node and resource are different", "field", "Allocatable", "Allocatable1", memoryState1.Allocatable, "Allocatable2", memoryState2.Allocatable)
		return false
	}
	return true
}

func (p *staticPolicy) getDefaultMachineState() state.NUMANodeMap {
	defaultMachineState := state.NUMANodeMap{}
	nodeHugepages := map[int]uint64{}
	for _, node := range p.machineInfo.Topology {
		defaultMachineState[node.Id] = &state.NUMANodeState{
			NumberOfAssignments: 0,
			MemoryMap:           map[v1.ResourceName]*state.MemoryTable{},
			Cells:               []int{node.Id},
		}

		// fill memory table with huge pages values
		for _, hugepage := range node.HugePages {
			hugepageQuantity := resource.NewQuantity(int64(hugepage.PageSize)*1024, resource.BinarySI)
			resourceName := corehelper.HugePageResourceName(*hugepageQuantity)
			systemReserved := p.getResourceSystemReserved(node.Id, resourceName)
			totalHugepagesSize := hugepage.NumPages * hugepage.PageSize * 1024
			allocatable := totalHugepagesSize - systemReserved
			defaultMachineState[node.Id].MemoryMap[resourceName] = &state.MemoryTable{
				Allocatable:    allocatable,
				Free:           allocatable,
				Reserved:       0,
				SystemReserved: systemReserved,
				TotalMemSize:   totalHugepagesSize,
			}
			if _, ok := nodeHugepages[node.Id]; !ok {
				nodeHugepages[node.Id] = 0
			}
			nodeHugepages[node.Id] += totalHugepagesSize
		}

		// fill memory table with regular memory values
		systemReserved := p.getResourceSystemReserved(node.Id, v1.ResourceMemory)

		allocatable := node.Memory - systemReserved
		// remove memory allocated by hugepages
		if allocatedByHugepages, ok := nodeHugepages[node.Id]; ok {
			allocatable -= allocatedByHugepages
		}
		defaultMachineState[node.Id].MemoryMap[v1.ResourceMemory] = &state.MemoryTable{
			Allocatable:    allocatable,
			Free:           allocatable,
			Reserved:       0,
			SystemReserved: systemReserved,
			TotalMemSize:   node.Memory,
		}
	}
	return defaultMachineState
}

func (p *staticPolicy) getResourceSystemReserved(nodeID int, resourceName v1.ResourceName) uint64 {
	var systemReserved uint64
	if nodeSystemReserved, ok := p.systemReserved[nodeID]; ok {
		if nodeMemorySystemReserved, ok := nodeSystemReserved[resourceName]; ok {
			systemReserved = nodeMemorySystemReserved
		}
	}
	return systemReserved
}

func (p *staticPolicy) getDefaultHint(machineState state.NUMANodeMap, pod *v1.Pod, requestedResources map[v1.ResourceName]uint64) (*topologymanager.TopologyHint, error) {
	hints := p.calculateHints(machineState, pod, requestedResources)
	if len(hints) < 1 {
		return nil, fmt.Errorf("[memorymanager] failed to get the default NUMA affinity, no NUMA nodes with enough memory is available")
	}

	// hints for all memory types should be the same, so we will check hints only for regular memory type
	return findBestHint(hints[string(v1.ResourceMemory)]), nil
}

func isAffinitySatisfyRequest(machineState state.NUMANodeMap, mask bitmask.BitMask, requestedResources map[v1.ResourceName]uint64) bool {
	totalFreeSize := map[v1.ResourceName]uint64{}
	for _, nodeID := range mask.GetBits() {
		for resourceName := range requestedResources {
			if _, ok := totalFreeSize[resourceName]; !ok {
				totalFreeSize[resourceName] = 0
			}
			totalFreeSize[resourceName] += machineState[nodeID].MemoryMap[resourceName].Free
		}
	}

	// verify that for all memory types the node mask has enough resources
	for resourceName, requestedSize := range requestedResources {
		if totalFreeSize[resourceName] < requestedSize {
			return false
		}
	}

	return true
}

// extendTopologyManagerHint extends the topology manager hint, in case when it does not satisfy to the container request
// the topology manager uses bitwise AND to merge all topology hints into the best one, so in case of the restricted policy,
// it possible that we will get the subset of hint that we provided to the topology manager, in this case we want to extend
// it to the original one
func (p *staticPolicy) extendTopologyManagerHint(machineState state.NUMANodeMap, pod *v1.Pod, requestedResources map[v1.ResourceName]uint64, mask bitmask.BitMask) (*topologymanager.TopologyHint, error) {
	hints := p.calculateHints(machineState, pod, requestedResources)

	var filteredHints []topologymanager.TopologyHint
	// hints for all memory types should be the same, so we will check hints only for regular memory type
	for _, hint := range hints[string(v1.ResourceMemory)] {
		affinityBits := hint.NUMANodeAffinity.GetBits()
		// filter all hints that does not include currentHint
		if isHintInGroup(mask.GetBits(), affinityBits) {
			filteredHints = append(filteredHints, hint)
		}
	}

	if len(filteredHints) < 1 {
		return nil, fmt.Errorf("[memorymanager] failed to find NUMA nodes to extend the current topology hint")
	}

	// try to find the preferred hint with the minimal number of NUMA nodes, relevant for the restricted policy
	return findBestHint(filteredHints), nil
}

func isHintInGroup(hint []int, group []int) bool {
	sort.Ints(hint)
	sort.Ints(group)

	hintIndex := 0
	for i := range group {
		if hintIndex == len(hint) {
			return true
		}

		if group[i] != hint[hintIndex] {
			continue
		}
		hintIndex++
	}

	return hintIndex == len(hint)
}

func findBestHint(hints []topologymanager.TopologyHint) *topologymanager.TopologyHint {
	// try to find the preferred hint with the minimal number of NUMA nodes, relevant for the restricted policy
	bestHint := topologymanager.TopologyHint{}
	for _, hint := range hints {
		if bestHint.NUMANodeAffinity == nil {
			bestHint = hint
			continue
		}

		// preferred of the current hint is true, when the extendedHint preferred is false
		if hint.Preferred && !bestHint.Preferred {
			bestHint = hint
			continue
		}

		// both hints has the same preferred value, but the current hint has less NUMA nodes than the extended one
		if hint.Preferred == bestHint.Preferred && hint.NUMANodeAffinity.IsNarrowerThan(bestHint.NUMANodeAffinity) {
			bestHint = hint
		}
	}
	return &bestHint
}

// GetAllocatableMemory returns the amount of allocatable memory for each NUMA node
func (p *staticPolicy) GetAllocatableMemory(s state.State) []state.Block {
	var allocatableMemory []state.Block
	machineState := s.GetMachineState()
	for numaNodeID, numaNodeState := range machineState {
		for resourceName, memoryTable := range numaNodeState.MemoryMap {
			if memoryTable.Allocatable == 0 {
				continue
			}

			block := state.Block{
				NUMAAffinity: []int{numaNodeID},
				Type:         resourceName,
				Size:         memoryTable.Allocatable,
			}
			allocatableMemory = append(allocatableMemory, block)
		}
	}
	return allocatableMemory
}

func (p *staticPolicy) updatePodReusableMemory(pod *v1.Pod, container *v1.Container, memoryBlocks []state.Block) {
	podUID := string(pod.UID)

	// If pod entries to m.initContainersReusableMemory other than the current pod exist, delete them.
	for uid := range p.initContainersReusableMemory {
		if podUID != uid {
			delete(p.initContainersReusableMemory, uid)
		}
	}

	if isRegularInitContainer(pod, container) {
		if _, ok := p.initContainersReusableMemory[podUID]; !ok {
			p.initContainersReusableMemory[podUID] = map[string]map[v1.ResourceName]uint64{}
		}

		for _, block := range memoryBlocks {
			blockBitMask, _ := bitmask.NewBitMask(block.NUMAAffinity...)
			blockBitMaskString := blockBitMask.String()

			if _, ok := p.initContainersReusableMemory[podUID][blockBitMaskString]; !ok {
				p.initContainersReusableMemory[podUID][blockBitMaskString] = map[v1.ResourceName]uint64{}
			}

			if blockReusableMemory := p.initContainersReusableMemory[podUID][blockBitMaskString][block.Type]; block.Size > blockReusableMemory {
				p.initContainersReusableMemory[podUID][blockBitMaskString][block.Type] = block.Size
			}
		}

		return
	}

	// update re-usable memory once it used by the app container
	for _, block := range memoryBlocks {
		blockBitMask, _ := bitmask.NewBitMask(block.NUMAAffinity...)
		if podReusableMemory := p.getPodReusableMemory(pod, blockBitMask, block.Type); podReusableMemory != 0 {
			if block.Size >= podReusableMemory {
				p.initContainersReusableMemory[podUID][blockBitMask.String()][block.Type] = 0
			} else {
				p.initContainersReusableMemory[podUID][blockBitMask.String()][block.Type] -= block.Size
			}
		}
	}
}

func (p *staticPolicy) updateInitContainersMemoryBlocks(logger klog.Logger, s state.State, pod *v1.Pod, container *v1.Container, containerMemoryBlocks []state.Block) {
	podUID := string(pod.UID)

	for _, containerBlock := range containerMemoryBlocks {
		blockSize := containerBlock.Size
		for _, initContainer := range pod.Spec.InitContainers {
			// we do not want to continue updates once we reach the current container
			if initContainer.Name == container.Name {
				break
			}

			if blockSize == 0 {
				break
			}

			if podutil.IsRestartableInitContainer(&initContainer) {
				// we should not reuse the resource from any restartable init
				// container
				continue
			}

			initContainerBlocks := s.GetMemoryBlocks(podUID, initContainer.Name)
			if len(initContainerBlocks) == 0 {
				continue
			}

			for i := range initContainerBlocks {
				initContainerBlock := &initContainerBlocks[i]
				if initContainerBlock.Size == 0 {
					continue
				}

				if initContainerBlock.Type != containerBlock.Type {
					continue
				}

				if !isNUMAAffinitiesEqual(logger, initContainerBlock.NUMAAffinity, containerBlock.NUMAAffinity) {
					continue
				}

				if initContainerBlock.Size > blockSize {
					initContainerBlock.Size -= blockSize
					blockSize = 0
				} else {
					blockSize -= initContainerBlock.Size
					initContainerBlock.Size = 0
				}
			}

			s.SetMemoryBlocks(podUID, initContainer.Name, initContainerBlocks)
		}
	}
}

func isRegularInitContainer(pod *v1.Pod, container *v1.Container) bool {
	for _, initContainer := range pod.Spec.InitContainers {
		if initContainer.Name == container.Name {
			return !podutil.IsRestartableInitContainer(&initContainer)
		}
	}

	return false
}

func isNUMAAffinitiesEqual(logger klog.Logger, numaAffinity1, numaAffinity2 []int) bool {
	bitMask1, err := bitmask.NewBitMask(numaAffinity1...)
	if err != nil {
		logger.Error(err, "failed to create bit mask", "numaAffinity1", numaAffinity1)
		return false
	}

	bitMask2, err := bitmask.NewBitMask(numaAffinity2...)
	if err != nil {
		logger.Error(err, "failed to create bit mask", "numaAffinity2", numaAffinity2)
		return false
	}

	return bitMask1.IsEqual(bitMask2)
}

func isAffinityViolatingNUMAAllocations(machineState state.NUMANodeMap, mask bitmask.BitMask) bool {
	maskBits := mask.GetBits()
	singleNUMAHint := len(maskBits) == 1
	for _, nodeID := range mask.GetBits() {
		// the node was never used for the memory allocation
		if machineState[nodeID].NumberOfAssignments == 0 {
			continue
		}
		if singleNUMAHint {
			continue
		}
		// the node used for the single NUMA memory allocation, it cannot be used for the multi NUMA node allocation
		if len(machineState[nodeID].Cells) == 1 {
			return true
		}
		// the node already used with a different group of nodes, it cannot be used within the current hint
		if !areGroupsEqual(machineState[nodeID].Cells, maskBits) {
			return true
		}
	}
	return false
}
