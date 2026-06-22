//go:build windows

/*
Copyright 2024 The Kubernetes Authors.

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

	"github.com/go-logr/logr"
	cadvisorapi "github.com/google/cadvisor/info/v1"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/klog/v2"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

// On Windows we want to use the same logic as the StaticPolicy to compute the memory topology hints
// but unlike linux based systems, on Windows systems numa nodes cannot be directly assigned or guaranteed via Windows APIs
// (windows scheduler will use the numa node that is closest to the cpu assigned therefor respecting the numa node assignment as a best effort). Because of this we don't want to have users specify "StaticPolicy" for the memory manager
// policy via kubelet configuration. Instead we want to use the "BestEffort" policy which will use the same logic as the StaticPolicy
// and doing so will reduce code duplication.

// policyTypeBestEffort is the BestEffort memory manager policy name. The policy
// is split across build-tagged files, so the constant is defined in both
// policy_best_effort_windows.go and policy_best_effort_other.go; the mutually
// exclusive build tags guarantee exactly one definition compiles per platform,
// keeping it available everywhere the shared policy switch and tests reference it.
const policyTypeBestEffort policyType = "BestEffort"

// newBestEffortPolicy is the platform constructor selector called by NewManager.
// On Windows it builds the BestEffort policy; the non-Windows build
// (policy_best_effort_other.go) rejects it, since BestEffort is Windows-only.
func newBestEffortPolicy(logger klog.Logger, machineInfo *cadvisorapi.MachineInfo, nodeAllocatableReservation v1.ResourceList, reservedMemory []kubeletconfig.MemoryReservation, affinity topologymanager.Store) (Policy, error) {
	systemReserved, err := getSystemReservedMemory(machineInfo, nodeAllocatableReservation, reservedMemory)
	if err != nil {
		return nil, err
	}
	return NewPolicyBestEffort(logger, machineInfo, systemReserved, affinity)
}

// bestEffortPolicy is implementation of the policy interface for the BestEffort policy
type bestEffortPolicy struct {
	static *staticPolicy
}

var _ Policy = &bestEffortPolicy{}

func NewPolicyBestEffort(logger logr.Logger, machineInfo *cadvisorapi.MachineInfo, reserved systemReservedMemory, affinity topologymanager.Store) (Policy, error) {
	p, err := NewPolicyStatic(logger, machineInfo, reserved, affinity)

	if err != nil {
		return nil, err
	}

	return &bestEffortPolicy{
		static: p.(*staticPolicy),
	}, nil
}

func (p *bestEffortPolicy) Name() string {
	return string(policyTypeBestEffort)
}

func (p *bestEffortPolicy) Start(logger logr.Logger, s state.State) error {
	return p.static.Start(logger, s)
}

func (p *bestEffortPolicy) RemoveContainer(logger logr.Logger, s state.State, podUID string, containerName string) {
	p.static.RemoveContainer(logger, s, podUID, containerName)
}

func (p *bestEffortPolicy) GetPodTopologyHints(logger logr.Logger, s state.State, pod *v1.Pod) map[string][]topologymanager.TopologyHint {
	// Pod-level resources are not supported on Windows.
	return nil
}

func (p *bestEffortPolicy) GetTopologyHints(logger logr.Logger, s state.State, pod *v1.Pod, container *v1.Container) map[string][]topologymanager.TopologyHint {
	return p.static.GetTopologyHints(logger, s, pod, container)
}

func (p *bestEffortPolicy) GetAllocatableMemory(s state.State) []state.Block {
	return p.static.GetAllocatableMemory(s)
}

func (p *bestEffortPolicy) AllocatePod(logger logr.Logger, s state.State, pod *v1.Pod) error {
	// Pod-level resources are not supported on Windows.
	return nil
}

// exclusiveCPUReporter is implemented by the Windows affinity store wrapper
// (cm.cpuFollowingStore) to report whether the CPU manager assigned a container
// exclusive CPUs. Allocate uses it to decide whether to follow the CPU manager's
// NUMA decision (no extend) or to fall back to its own calculation (extend).
type exclusiveCPUReporter interface {
	HasExclusiveCPUs(podUID, containerName string) bool
}

// Allocate mirrors staticPolicy.Allocate EXCEPT for the extend step, which is
// conditional. When the container is following the CPU manager's decision — i.e.
// it has exclusive CPUs, reported via the affinity store's HasExclusiveCPUs — the
// NUMA affinity must remain exactly those CPU nodes, so this policy does NOT
// extend the hint: the memory manager stays strictly in sync with the CPU manager
// and never diverges onto a different NUMA node. On Windows memory placement is
// best-effort and follows the assigned CPUs (there is no cpuset.mems), so
// extending toward nodes with free memory would only describe a placement the
// kernel never performs and re-introduce the CPU-affinity union at container
// create time.
//
// When there are no exclusive CPUs to follow (CPU manager policy "none", or a
// shared/non-Guaranteed container), there is no CPU decision to stay in sync with,
// so the policy falls back to staticPolicy's behavior and DOES extend the hint.
//
// Bookkeeping consequence (follow-CPU case): when the CPU's node(s) lack enough
// free memory for the request, the full request is still recorded against those
// node(s); the overflow beyond their free memory is left unattributed (the kernel
// spills it best-effort). This stays internally consistent across validateState
// and releaseMemory because both clamp identically.
//
// IMPORTANT: keep this in sync with staticPolicy.Allocate. The only intended
// difference is the conditional extendTopologyManagerHint step.
func (p *bestEffortPolicy) Allocate(logger logr.Logger, s state.State, pod *v1.Pod, container *v1.Container) (rerr error) {
	sp := p.static
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
		sp.updatePodReusableMemory(pod, container, blocks)

		logger.Info("Container already present in state, skipping")
		return nil
	}

	// Call Topology Manager to get the aligned affinity across all hint providers.
	// On Windows the affinity store is the cpuFollowingStore wrapper, so this is
	// the CPU manager's NUMA decision.
	hint := sp.affinity.GetAffinity(podUID, container.Name)
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
		defaultHint, err := sp.getDefaultHint(machineState, pod, requestedResources)
		if err != nil {
			return err
		}

		if !defaultHint.Preferred && bestHint.Preferred {
			return fmt.Errorf("[memorymanager] failed to find the default preferred hint")
		}
		bestHint = defaultHint
	}

	// staticPolicy.Allocate extends the hint here when the merged affinity does not
	// satisfy the request. We skip that ONLY when this container is following the
	// CPU manager's exclusive-CPU decision, so the memory affinity stays exactly
	// the CPU manager's nodes. Otherwise (CPU manager "none", or a shared
	// container) there is no CPU decision to follow, so we extend like the static
	// policy and let the memory manager do its own calculation.
	followCPU := false
	if r, ok := sp.affinity.(exclusiveCPUReporter); ok {
		followCPU = r.HasExclusiveCPUs(podUID, container.Name)
	}
	if !followCPU && !isAffinitySatisfyRequest(machineState, bestHint.NUMANodeAffinity, requestedResources) {
		extendedHint, err := sp.extendTopologyManagerHint(machineState, pod, requestedResources, bestHint.NUMANodeAffinity)
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

		podReusableMemory := sp.getPodReusableMemory(pod, bestHint.NUMANodeAffinity, resourceName)
		if podReusableMemory >= requestedSize {
			requestedSize = 0
		} else {
			requestedSize -= podReusableMemory
		}

		// Update nodes memory state
		sp.updateMachineState(machineState, maskBits, resourceName, requestedSize)
		metrics.ResourceManagerAllocationsTotal.WithLabelValues(metrics.ResourceManagerMemory, metrics.ResourceManagerNode).Inc()
	}

	sp.updatePodReusableMemory(pod, container, containerBlocks)

	s.SetMachineState(machineState)
	s.SetMemoryBlocks(podUID, container.Name, containerBlocks)

	// update init containers memory blocks to reflect the fact that we re-used init containers memory
	sp.updateInitContainersMemoryBlocks(logger, s, pod, container, containerBlocks)
	metrics.ResourceManagerContainerAssignments.WithLabelValues(metrics.ResourceManagerMemory, metrics.ResourceManagerExclusiveNode).Inc()

	logger.V(4).Info("Allocated exclusive memory")
	return nil
}
