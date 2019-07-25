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
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/socketmask"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
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
}

//HintProvider interface is to be implemented by Hint Providers
type HintProvider interface {
	GetTopologyHints(pod v1.Pod, container v1.Container) map[string][]TopologyHint
}

//Store interface is to allow Hint Providers to retrieve pod affinity
type Store interface {
	GetAffinity(podUID string, containerName string) TopologyHint
}

//TopologyHint is a struct containing a SocketMask for a Container
type TopologyHint struct {
	SocketAffinity socketmask.SocketMask
	// Preferred is set to true when the SocketMask encodes a preferred
	// allocation for the Container. It is set to false otherwise.
	Preferred bool
}

var _ Manager = &manager{}

//NewManager creates a new TopologyManager based on provided policy
func NewManager(topologyPolicyName string) (Manager, error) {
	klog.Infof("[topologymanager] Creating topology manager with %s policy", topologyPolicyName)
	var policy Policy

	switch topologyPolicyName {

	case PolicyNone:
		policy = NewNonePolicy()

	case PolicyBestEffort:
		policy = NewBestEffortPolicy()

	case PolicyStrict:
		policy = NewStrictPolicy()

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
	// Set the default hint to return from this function as an any-socket
	// affinity with an unpreferred allocation. This will only be returned if
	// no better hint can be found when merging hints from each hint provider.
	defaultAffinity, _ := socketmask.NewSocketMask()
	defaultAffinity.Fill()
	defaultHint := TopologyHint{defaultAffinity, false}

	// Loop through all hint providers and save an accumulated list of the
	// hints returned by each hint provider. If no hints are provided, assume
	// that provider has no preference for topology-aware allocation.
	var allProviderHints [][]TopologyHint
	for _, provider := range m.hintProviders {
		// Get the TopologyHints from a provider.
		hints := provider.GetTopologyHints(pod, container)

		// If hints is nil, insert a single, preferred any-socket hint into allProviderHints.
		if hints == nil || len(hints) == 0 {
			klog.Infof("[topologymanager] Hint Provider has no preference for socket affinity with any resource")
			affinity, _ := socketmask.NewSocketMask()
			affinity.Fill()
			allProviderHints = append(allProviderHints, []TopologyHint{{affinity, true}})
			continue
		}

		// Otherwise, accumulate the hints for each resource type into allProviderHints.
		for resource := range hints {
			if hints[resource] == nil {
				klog.Infof("[topologymanager] Hint Provider has no preference for socket affinity with resource '%s'", resource)
				affinity, _ := socketmask.NewSocketMask()
				affinity.Fill()
				allProviderHints = append(allProviderHints, []TopologyHint{{affinity, true}})
				continue
			}

			if len(hints[resource]) == 0 {
				klog.Infof("[topologymanager] Hint Provider has no possible socket affinities for resource '%s'", resource)
				affinity, _ := socketmask.NewSocketMask()
				affinity.Fill()
				allProviderHints = append(allProviderHints, []TopologyHint{{affinity, false}})
				continue
			}

			allProviderHints = append(allProviderHints, hints[resource])
		}
	}

	// Iterate over all permutations of hints in 'allProviderHints'. Merge the
	// hints in each permutation by taking the bitwise-and of their affinity masks.
	// Return the hint with the narrowest SocketAffinity of all merged
	// permutations that have at least one socket set. If no merged mask can be
	// found that has at least one socket set, return the 'defaultHint'.
	bestHint := defaultHint
	m.iterateAllProviderTopologyHints(allProviderHints, func(permutation []TopologyHint) {
		// Get the SocketAffinity from each hint in the permutation and see if any
		// of them encode unpreferred allocations.
		preferred := true
		var socketAffinities []socketmask.SocketMask
		for _, hint := range permutation {
			// Only consider hints that have an actual SocketAffinity set.
			if hint.SocketAffinity != nil {
				if !hint.Preferred {
					preferred = false
				}
				socketAffinities = append(socketAffinities, hint.SocketAffinity)
			}
		}

		// Merge the affinities using a bitwise-and operation.
		mergedAffinity, _ := socketmask.NewSocketMask()
		mergedAffinity.Fill()
		mergedAffinity.And(socketAffinities...)

		// Build a mergedHintfrom the merged affinity mask, indicating if an
		// preferred allocation was used to generate the affinity mask or not.
		mergedHint := TopologyHint{mergedAffinity, preferred}

		// Only consider mergedHints that result in a SocketAffinity > 0 to
		// replace the current bestHint.
		if mergedHint.SocketAffinity.Count() == 0 {
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
		// mergedHints that have a narrower SocketAffinity than the
		// SocketAffinity in the current bestHint.
		if !mergedHint.SocketAffinity.IsNarrowerThan(bestHint.SocketAffinity) {
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
			admitPod := m.policy.CanAdmitPodResult(result.Preferred)
			if admitPod.Admit == false {
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
