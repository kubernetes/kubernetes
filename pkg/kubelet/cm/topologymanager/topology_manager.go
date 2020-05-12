/*
Copyright 2019 The Kubernetes Authors.

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

package topologymanager

import (
	"fmt"
	"sync"

	"k8s.io/api/core/v1"
	"k8s.io/klog"
	cputopology "k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

const (
	// maxAllowableNUMANodes specifies the maximum number of NUMA Nodes that
	// the TopologyManager supports on the underlying machine.
	//
	// At present, having more than this number of NUMA Nodes will result in a
	// state explosion when trying to enumerate possible NUMAAffinity masks and
	// generate hints for them. As such, if more NUMA Nodes than this are
	// present on a machine and the TopologyManager is enabled, an error will
	// be returned and the TopologyManager will not be loaded.
	maxAllowableNUMANodes = 8
)

//Manager interface provides methods for Kubelet to manage pod topology hints
type Manager interface {
	//Manager implements pod admit handler interface
	lifecycle.PodAdmitHandler
	//Adds a hint provider to manager to indicate the hint provider
	//wants to be consoluted when making topology hints
	AddHintProvider(HintProvider)
	//Adds pod to Manager for tracking
	AddContainer(pod *v1.Pod, containerID string) error
	//Removes pod from Manager tracking
	RemoveContainer(containerID string) error
	//Interface for storing pod topology hints
	Store
}

type manager struct {
	mutex sync.Mutex
	//The list of components registered with the Manager
	hintProviders []HintProvider
	//Mapping of a Pods mapping of Containers and their TopologyHints
	//Indexed by PodUID to ContainerName
	podTopologyHints map[string]map[string]TopologyHint
	//Mapping of PodUID to ContainerID for Adding/Removing Pods from PodTopologyHints mapping
	podMap map[string]string
	//Topology Manager Policy
	policy Policy
	//NUMA nodes of host machine
	numaNodes []int
}

// HintProvider is an interface for components that want to collaborate to
// achieve globally optimal concrete resource alignment with respect to
// NUMA locality.
type HintProvider interface {
	// GetTopologyHints returns a map of resource names to a list of possible
	// concrete resource allocations in terms of NUMA locality hints. Each hint
	// is optionally marked "preferred" and indicates the set of NUMA nodes
	// involved in the hypothetical allocation. The topology manager calls
	// this function for each hint provider, and merges the hints to produce
	// a consensus "best" hint. The hint providers may subsequently query the
	// topology manager to influence actual resource assignment.
	GetTopologyHints(pod *v1.Pod, container *v1.Container) map[string][]TopologyHint
	// Allocate triggers resource allocation to occur on the HintProvider after
	// all hints have been gathered and the aggregated Hint is available via a
	// call to Store.GetAffinity().
	Allocate(pod *v1.Pod, container *v1.Container) error
	// DeAllocate triggers resource de-allocation to occur on the HintProvider.
	// topology manager call this function to reclaim allocated resources,
	DeAllocate(pod *v1.Pod, container *v1.Container) error
}

//Store interface is to allow Hint Providers to retrieve pod affinity
type Store interface {
	GetAffinity(podUID string, containerName string) TopologyHint
}

//TopologyHint is a struct containing the NUMANodeAffinity for a Container
type TopologyHint struct {
	NUMANodeAffinity bitmask.BitMask
	// Preferred is set to true when the NUMANodeAffinity encodes a preferred
	// allocation for the Container. It is set to false otherwise.
	Preferred bool
}

// IsEqual checks if TopologyHint are equal
func (th *TopologyHint) IsEqual(topologyHint TopologyHint) bool {
	if th.Preferred == topologyHint.Preferred {
		if th.NUMANodeAffinity == nil || topologyHint.NUMANodeAffinity == nil {
			return th.NUMANodeAffinity == topologyHint.NUMANodeAffinity
		}
		return th.NUMANodeAffinity.IsEqual(topologyHint.NUMANodeAffinity)
	}
	return false
}

// LessThan checks if TopologyHint `a` is less than TopologyHint `b`
// this means that either `a` is a preferred hint and `b` is not
// or `a` NUMANodeAffinity attribute is narrower than `b` NUMANodeAffinity attribute.
func (th *TopologyHint) LessThan(other TopologyHint) bool {
	if th.Preferred != other.Preferred {
		return th.Preferred == true
	}
	return th.NUMANodeAffinity.IsNarrowerThan(other.NUMANodeAffinity)
}

var _ Manager = &manager{}

//NewManager creates a new TopologyManager based on provided policy
func NewManager(numaNodeInfo cputopology.NUMANodeInfo, topologyPolicyName string) (Manager, error) {
	klog.Infof("[topologymanager] Creating topology manager with %s policy", topologyPolicyName)

	var numaNodes []int
	for node := range numaNodeInfo {
		numaNodes = append(numaNodes, node)
	}

	if topologyPolicyName != PolicyNone && len(numaNodes) > maxAllowableNUMANodes {
		return nil, fmt.Errorf("unsupported on machines with more than %v NUMA Nodes", maxAllowableNUMANodes)
	}

	var policy Policy
	switch topologyPolicyName {

	case PolicyNone:
		policy = NewNonePolicy()

	case PolicyBestEffort:
		policy = NewBestEffortPolicy(numaNodes)

	case PolicyRestricted:
		policy = NewRestrictedPolicy(numaNodes)

	case PolicySingleNumaNode:
		policy = NewSingleNumaNodePolicy(numaNodes)

	case PolicyPodLevelSingleNumaNode:
		policy = NewPodLevelSingleNumaNodePolicy(numaNodes)

	default:
		return nil, fmt.Errorf("unknown policy: \"%s\"", topologyPolicyName)
	}

	var hp []HintProvider
	pth := make(map[string]map[string]TopologyHint)
	pm := make(map[string]string)
	manager := &manager{
		hintProviders:    hp,
		podTopologyHints: pth,
		podMap:           pm,
		policy:           policy,
		numaNodes:        numaNodes,
	}

	return manager, nil
}

func (m *manager) GetAffinity(podUID string, containerName string) TopologyHint {
	return m.podTopologyHints[podUID][containerName]
}

func (m *manager) accumulateProvidersHints(pod *v1.Pod, container *v1.Container) (providersHints []map[string][]TopologyHint) {
	// Loop through all hint providers and save an accumulated list of the
	// hints returned by each hint provider.
	for _, provider := range m.hintProviders {
		// Get the TopologyHints from a provider.
		hints := provider.GetTopologyHints(pod, container)
		providersHints = append(providersHints, hints)
		klog.Infof("[topologymanager] TopologyHints for pod '%v', container '%v': %v", pod.Name, container.Name, hints)
	}
	return providersHints
}

func (m *manager) allocateAlignedResources(pod *v1.Pod, container *v1.Container) error {
	for _, provider := range m.hintProviders {
		err := provider.Allocate(pod, container)
		if err != nil {
			return err
		}
	}
	return nil
}

// Collect Hints from hint providers and pass to policy to retrieve the best one.
func (m *manager) calculateAffinity(pod *v1.Pod, container *v1.Container) (TopologyHint, bool) {
	providersHints := m.accumulateProvidersHints(pod, container)
	bestHint, admit := m.policy.Merge(providersHints)
	klog.Infof("[topologymanager] ContainerTopologyHint: %v", bestHint)
	return bestHint, admit
}

func (m *manager) AddHintProvider(h HintProvider) {
	m.hintProviders = append(m.hintProviders, h)
}

func (m *manager) AddContainer(pod *v1.Pod, containerID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.podMap[containerID] = string(pod.UID)
	return nil
}

func (m *manager) RemoveContainer(containerID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	klog.Infof("[topologymanager] RemoveContainer - Container ID: %v", containerID)
	podUIDString := m.podMap[containerID]
	delete(m.podMap, containerID)
	if _, exists := m.podTopologyHints[podUIDString]; exists {
		delete(m.podTopologyHints[podUIDString], containerID)
		if len(m.podTopologyHints[podUIDString]) == 0 {
			delete(m.podTopologyHints, podUIDString)
		}
	}

	return nil
}

// Call DeAllocate function of all registered hint providers for all containers in a pod.
func (m *manager) reclaimAllResources(pod *v1.Pod) {
	klog.Infof("[topologymanager] pod(%v) is reject, reclaim all resources for the pod.", pod.UID)

	podUIDString := string(pod.UID)

	for _, container := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
		for _, provider := range m.hintProviders {
			err := provider.DeAllocate(pod, &container)
			if err != nil {
				klog.Errorf("[topologymanager] DeAllocate failed for container(%v) due to %v, which is unexpected", container.Name, err)
			}
		}

		if _, exists := m.podTopologyHints[podUIDString]; exists {
			delete(m.podTopologyHints[podUIDString], container.Name)
		}
	}

	// since this function touches m.podTopologyHints, it should be called only in Admit function.
	// otherwise we need to put a lock here
	if _, exists := m.podTopologyHints[podUIDString]; exists {
		delete(m.podTopologyHints, podUIDString)
	}
}

// Most major part of pod-level-single-numa-node is implemented here,
// Since hint provider and policy interfaces are designed by container basis.
func (m *manager) runPodBasisAdmitLogic(pod *v1.Pod) lifecycle.PodAdmitResult {
	// Loop all NUMA nodes
	for node := range m.numaNodes {
		currentNumaAffinity, _ := bitmask.NewBitMask(node)
		isAdmitted := true

		// Loop containers
		for _, container := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {

			// set empty slice of map to store hints from providers
			providersHints := []map[string][]TopologyHint{}

			// Loop through all registered hint providers to get topology hints
			//ex) []map[string][]TopologyHint{
			//      {"cpu" : {01, T}, {10, T}, {11, F}},
			//      {"gpu" : {01, T}, {10, T}, {11, F}},
			//      {"fpga" : nil}, // no preference
			//      nil, // no preference
			//    }
			for _, provider := range m.hintProviders {
				// Get the TopologyHints from a provider.
				hints := provider.GetTopologyHints(pod, &container)
				providersHints = append(providersHints, hints)
				klog.Infof("[topologymanager] TopologyHints for pod '%v', container '%v': %v", pod.Name, container.Name, hints)
			}

			// filter out hints that indicates ohter than current visiting numa node
			// so that policy.Merge can deal with  hints only for current visiting NUMA node.
			// it allows this policy running with low time complexity of hint merging algorithm.
			//ex) []map[string][]TopologyHint{  //assumption here is current numa node is 01.
			//      {"cpu" : {01, T}},
			//      {"gpu" : {01, T}},
			//      {"fpga" : nil}, // no preference
			//      nil, // no preference
			//    }
			providersHints = filterProvidersHintsForCurrentNumaNode(providersHints, currentNumaAffinity)

			// run hint merging algorithm for container
			bestHint, admit := m.policy.Merge(providersHints)

			// the policy found a container cannot bound on the current NUMA node.
			// the policy allows TopologyHint{nil, true} since it means no preference of topology.
			if !admit || (bestHint.NUMANodeAffinity != nil && !bestHint.NUMANodeAffinity.IsEqual(currentNumaAffinity)) {
				// revert resource pre-allocation for the pod
				m.reclaimAllResources(pod)

				// make to move to the next numa node
				isAdmitted = false
				break
			}

			// Assign PodTopologyHints : mapping PID, CName, bestHint
			klog.Infof("[topologymanager] Topology Affinity for (pod: %v container: %v): %v", pod.UID, container.Name, bestHint)
			if m.podTopologyHints[string(pod.UID)] == nil {
				m.podTopologyHints[string(pod.UID)] = make(map[string]TopologyHint)
			}
			m.podTopologyHints[string(pod.UID)][container.Name] = bestHint

			// Allocate resources
			err := m.allocateAlignedResources(pod, &container)
			if err != nil {
				m.reclaimAllResources(pod)
				return lifecycle.PodAdmitResult{
					Message: fmt.Sprintf("Allocate failed due to %v, which is unexpected", err),
					Reason:  "UnexpectedAdmissionError",
					Admit:   false,
				}
			}
		}

		if isAdmitted {
			// all containers in the pod get resource allocation from current NUMA node
			return lifecycle.PodAdmitResult{Admit: true}
		}
	}

	//If a Pod is not admitted on any numa node, reject the pod.
	return lifecycle.PodAdmitResult{
		Message: fmt.Sprintf("Resources cannot be allocated with Topology locality"),
		Reason:  "TopologyAffinityError",
		Admit:   false,
	}
}

func (m *manager) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	klog.Infof("[topologymanager] Topology Admit Handler")
	pod := attrs.Pod

	if m.policy.Name() == PolicyPodLevelSingleNumaNode {
		return m.runPodBasisAdmitLogic(pod)
	}

	for _, container := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
		if m.policy.Name() == PolicyNone {
			err := m.allocateAlignedResources(pod, &container)
			if err != nil {
				return lifecycle.PodAdmitResult{
					Message: fmt.Sprintf("Allocate failed due to %v, which is unexpected", err),
					Reason:  "UnexpectedAdmissionError",
					Admit:   false,
				}
			}
			continue
		}

		result, admit := m.calculateAffinity(pod, &container)
		if !admit {
			return lifecycle.PodAdmitResult{
				Message: "Resources cannot be allocated with Topology locality",
				Reason:  "TopologyAffinityError",
				Admit:   false,
			}
		}

		klog.Infof("[topologymanager] Topology Affinity for (pod: %v container: %v): %v", pod.UID, container.Name, result)
		if m.podTopologyHints[string(pod.UID)] == nil {
			m.podTopologyHints[string(pod.UID)] = make(map[string]TopologyHint)
		}
		m.podTopologyHints[string(pod.UID)][container.Name] = result

		err := m.allocateAlignedResources(pod, &container)
		if err != nil {
			return lifecycle.PodAdmitResult{
				Message: fmt.Sprintf("Allocate failed due to %v, which is unexpected", err),
				Reason:  "UnexpectedAdmissionError",
				Admit:   false,
			}
		}
	}

	return lifecycle.PodAdmitResult{Admit: true}
}

// This function returns filtered providersHints, filtered hints has
// the topology hint, which indicates no preference of topology,
// and the topology hint matched with given NUMA affinity.
func filterProvidersHintsForCurrentNumaNode(providersHints []map[string][]TopologyHint, currentAffinity bitmask.BitMask) []map[string][]TopologyHint {
	// set empty slice of map here
	filteredProvidersHints := []map[string][]TopologyHint{}
	for _, hints := range providersHints {
		// empty map indicates no hints are provided
		// assume that provider has no preference for topology-aware allocation
		if len(hints) == 0 {
			filteredProvidersHints = append(filteredProvidersHints, hints)
			continue
		}
		// Otherwise
		providerHints := make(map[string][]TopologyHint)
		for resource := range hints {
			// The function don't touch the below two type of hint.
			// nil slice of hint indicates no prerference for topology-aware allocation
			// empty slice of hint indecates no possible NUMA affinities
			if hints[resource] == nil || len(hints[resource]) == 0 {
				providerHints[resource] = hints[resource]
				continue
			}
			providerHints[resource] = make([]TopologyHint, 0)
			for _, hint := range hints[resource] {
				if !hint.NUMANodeAffinity.IsEqual(currentAffinity) {
					continue
				}
				providerHints[resource] = append(providerHints[resource], hint)
			}
		}
		filteredProvidersHints = append(filteredProvidersHints, providerHints)
	}
	return filteredProvidersHints
}
