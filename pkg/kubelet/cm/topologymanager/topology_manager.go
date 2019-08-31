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

	"k8s.io/api/core/v1"
	"k8s.io/klog"
	cputopology "k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/socketmask"
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
	//The list of components registered with the Manager
	hintProviders []HintProvider
	//Mapping of a Pods mapping of Containers and their TopologyHints
	//Indexed by PodUID to ContainerName
	podTopologyHints map[string]map[string]TopologyHint
	//Mapping of PodUID to ContainerID for Adding/Removing Pods from PodTopologyHints mapping
	podMap map[string]string
	//Topology Manager Policy
	policy Policy
	//List of NUMA Nodes available on the underlying machine
	numaNodes []int
}

//HintProvider interface is to be implemented by Hint Providers
type HintProvider interface {
	GetTopologyHints(pod v1.Pod, container v1.Container) map[string][]TopologyHint
}

//Store interface is to allow Hint Providers to retrieve pod affinity
type Store interface {
	GetAffinity(podUID string, containerName string) TopologyHint
}

//TopologyHint is a struct containing the NUMANodeAffinity for a Container
type TopologyHint struct {
	NUMANodeAffinity socketmask.SocketMask
	// Preferred is set to true when the NUMANodeAffinity encodes a preferred
	// allocation for the Container. It is set to false otherwise.
	Preferred bool
}

var _ Manager = &manager{}

//NewManager creates a new TopologyManager based on provided policy
func NewManager(numaNodeInfo cputopology.NUMANodeInfo, topologyPolicyName string) (Manager, error) {
	klog.Infof("[topologymanager] Creating topology manager with %s policy", topologyPolicyName)
	var policy Policy

	switch topologyPolicyName {

	case PolicyNone:
		policy = NewNonePolicy()

	case PolicyBestEffort:
		policy = NewBestEffortPolicy()

	case PolicyRestricted:
		policy = NewRestrictedPolicy()

	case PolicySingleNumaNode:
		policy = NewSingleNumaNodePolicy()

	default:
		return nil, fmt.Errorf("unknown policy: \"%s\"", topologyPolicyName)
	}

	var numaNodes []int
	for node := range numaNodeInfo {
		numaNodes = append(numaNodes, node)
	}

	if len(numaNodes) > maxAllowableNUMANodes {
		return nil, fmt.Errorf("unsupported on machines with more than %v NUMA Nodes", maxAllowableNUMANodes)
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

// Iterate over all permutations of hints in 'allProviderHints [][]TopologyHint'.
//
// This procedure is implemented as a recursive function over the set of hints
// in 'allproviderHints[i]'. It applies the function 'callback' to each
// permutation as it is found. It is the equivalent of:
//
// for i := 0; i < len(providerHints[0]); i++
//     for j := 0; j < len(providerHints[1]); j++
//         for k := 0; k < len(providerHints[2]); k++
//             ...
//             for z := 0; z < len(providerHints[-1]); z++
//                 permutation := []TopologyHint{
//                     providerHints[0][i],
//                     providerHints[1][j],
//                     providerHints[2][k],
//                     ...
//                     provideryHints[-1][z]
//                 }
//                 callback(permutation)
func (m *manager) iterateAllProviderTopologyHints(allProviderHints [][]TopologyHint, callback func([]TopologyHint)) {
	// Internal helper function to accumulate the permutation before calling the callback.
	var iterate func(i int, accum []TopologyHint)
	iterate = func(i int, accum []TopologyHint) {
		// Base case: we have looped through all providers and have a full permutation.
		if i == len(allProviderHints) {
			callback(accum)
			return
		}

		// Loop through all hints for provider 'i', and recurse to build the
		// the permutation of this hint with all hints from providers 'i++'.
		for j := range allProviderHints[i] {
			iterate(i+1, append(accum, allProviderHints[i][j]))
		}
	}
	iterate(0, []TopologyHint{})
}

// Merge the hints from all hint providers to find the best one.
func (m *manager) calculateAffinity(pod v1.Pod, container v1.Container) TopologyHint {
	// Set the default affinity as an any-numa affinity containing the list
	// of NUMA Nodes available on this machine.
	defaultAffinity, _ := socketmask.NewSocketMask(m.numaNodes...)

	// Loop through all hint providers and save an accumulated list of the
	// hints returned by each hint provider. If no hints are provided, assume
	// that provider has no preference for topology-aware allocation.
	var allProviderHints [][]TopologyHint
	for _, provider := range m.hintProviders {
		// Get the TopologyHints from a provider.
		hints := provider.GetTopologyHints(pod, container)

		// If hints is nil, insert a single, preferred any-numa hint into allProviderHints.
		if len(hints) == 0 {
			klog.Infof("[topologymanager] Hint Provider has no preference for NUMA affinity with any resource")
			allProviderHints = append(allProviderHints, []TopologyHint{{defaultAffinity, true}})
			continue
		}

		// Otherwise, accumulate the hints for each resource type into allProviderHints.
		for resource := range hints {
			if hints[resource] == nil {
				klog.Infof("[topologymanager] Hint Provider has no preference for NUMA affinity with resource '%s'", resource)
				allProviderHints = append(allProviderHints, []TopologyHint{{defaultAffinity, true}})
				continue
			}

			if len(hints[resource]) == 0 {
				klog.Infof("[topologymanager] Hint Provider has no possible NUMA affinities for resource '%s'", resource)
				allProviderHints = append(allProviderHints, []TopologyHint{{defaultAffinity, false}})
				continue
			}

			allProviderHints = append(allProviderHints, hints[resource])
		}
	}

	// Iterate over all permutations of hints in 'allProviderHints'. Merge the
	// hints in each permutation by taking the bitwise-and of their affinity masks.
	// Return the hint with the narrowest NUMANodeAffinity of all merged
	// permutations that have at least one NUMA ID set. If no merged mask can be
	// found that has at least one NUMA ID set, return the 'defaultAffinity'.
	bestHint := TopologyHint{defaultAffinity, false}
	m.iterateAllProviderTopologyHints(allProviderHints, func(permutation []TopologyHint) {
		// Get the NUMANodeAffinity from each hint in the permutation and see if any
		// of them encode unpreferred allocations.
		preferred := true
		var numaAffinities []socketmask.SocketMask
		for _, hint := range permutation {
			// Only consider hints that have an actual NUMANodeAffinity set.
			if hint.NUMANodeAffinity != nil {
				if !hint.Preferred {
					preferred = false
				}
				// Special case PolicySingleNumaNode to only prefer hints where
				// all providers have a single NUMA affinity set.
				if m.policy != nil && m.policy.Name() == PolicySingleNumaNode && hint.NUMANodeAffinity.Count() > 1 {
					preferred = false
				}
				numaAffinities = append(numaAffinities, hint.NUMANodeAffinity)
			}
		}

		// Merge the affinities using a bitwise-and operation.
		mergedAffinity, _ := socketmask.NewSocketMask(m.numaNodes...)
		mergedAffinity.And(numaAffinities...)

		// Build a mergedHintfrom the merged affinity mask, indicating if an
		// preferred allocation was used to generate the affinity mask or not.
		mergedHint := TopologyHint{mergedAffinity, preferred}

		// Only consider mergedHints that result in a NUMANodeAffinity > 0 to
		// replace the current bestHint.
		if mergedHint.NUMANodeAffinity.Count() == 0 {
			return
		}

		// If the current bestHint is non-preferred and the new mergedHint is
		// preferred, always choose the preferred hint over the non-preferred one.
		if mergedHint.Preferred && !bestHint.Preferred {
			bestHint = mergedHint
			return
		}

		// If the current bestHint is preferred and the new mergedHint is
		// non-preferred, never update bestHint, regardless of mergedHint's
		// narowness.
		if !mergedHint.Preferred && bestHint.Preferred {
			return
		}

		// If mergedHint and bestHint has the same preference, only consider
		// mergedHints that have a narrower NUMANodeAffinity than the
		// NUMANodeAffinity in the current bestHint.
		if !mergedHint.NUMANodeAffinity.IsNarrowerThan(bestHint.NUMANodeAffinity) {
			return
		}

		// In all other cases, update bestHint to the current mergedHint
		bestHint = mergedHint
	})

	klog.Infof("[topologymanager] ContainerTopologyHint: %v", bestHint)

	return bestHint
}

func (m *manager) AddHintProvider(h HintProvider) {
	m.hintProviders = append(m.hintProviders, h)
}

func (m *manager) AddContainer(pod *v1.Pod, containerID string) error {
	m.podMap[containerID] = string(pod.UID)
	return nil
}

func (m *manager) RemoveContainer(containerID string) error {
	podUIDString := m.podMap[containerID]
	delete(m.podTopologyHints, podUIDString)
	delete(m.podMap, containerID)
	klog.Infof("[topologymanager] RemoveContainer - Container ID: %v podTopologyHints: %v", containerID, m.podTopologyHints)
	return nil
}

func (m *manager) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	klog.Infof("[topologymanager] Topology Admit Handler")
	if m.policy.Name() == "none" {
		klog.Infof("[topologymanager] Skipping calculate topology affinity as policy: none")
		return lifecycle.PodAdmitResult{
			Admit: true,
		}
	}
	pod := attrs.Pod
	c := make(map[string]TopologyHint)
	klog.Infof("[topologymanager] Pod QoS Level: %v", pod.Status.QOSClass)

	if pod.Status.QOSClass == v1.PodQOSGuaranteed {
		for _, container := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
			result := m.calculateAffinity(*pod, container)
			admitPod := m.policy.CanAdmitPodResult(&result)
			if !admitPod.Admit {
				return admitPod
			}
			c[container.Name] = result
		}
		m.podTopologyHints[string(pod.UID)] = c
		klog.Infof("[topologymanager] Topology Affinity for Pod: %v are %v", pod.UID, m.podTopologyHints[string(pod.UID)])

	} else {
		klog.Infof("[topologymanager] Topology Manager only affinitises Guaranteed pods.")
	}

	return lifecycle.PodAdmitResult{
		Admit: true,
	}
}
