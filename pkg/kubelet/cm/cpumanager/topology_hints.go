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

package cpumanager

import (
	"k8s.io/api/core/v1"
	"k8s.io/klog"

	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/socketmask"
)

func (m *manager) GetTopologyHints(pod v1.Pod, container v1.Container) map[string][]topologymanager.TopologyHint {
	// The 'none' policy does not generate topology hints.
	if m.policy.Name() == string(PolicyNone) {
		return nil
	}

	// For all other policies, if there are no CPU resources requested for this
	// container, we do not generate any topology hints.
	if _, ok := container.Resources.Requests[v1.ResourceCPU]; !ok {
		return nil
	}

	// Otherwise, attempt to generate TopologyHints for the CPUManager.
	// For now, this implementation assumes the 'static' CPUManager policy.
	// TODO: Generalize this so that its applicable to future CPUManager polices.

	// Get a count of how many guaranteed CPUs have been requested.
	requested := m.policy.(*staticPolicy).guaranteedCPUs(&pod, &container)

	// If there are no guaranteed CPUs being requested, we do not generate
	// any topology hints. This can happen, for example, because init
	// containers don't have to have guaranteed CPUs in order for the pod
	// to still be in the Guaranteed QOS tier.
	if requested == 0 {
		return nil
	}

	// Get a list of available CPUs.
	available := m.policy.(*staticPolicy).assignableCPUs(m.state)

	// Generate hints.
	cpuHints := m.generateCPUTopologyHints(available, requested)
	klog.Infof("[cpumanager] TopologyHints generated for pod '%v', container '%v': %v", pod.Name, container.Name, cpuHints)

	return map[string][]topologymanager.TopologyHint{
		string(v1.ResourceCPU): cpuHints,
	}
}

// generateCPUtopologyHints generates a set of TopologyHints given the set of
// available CPUs and the number of CPUs being requested.
//
// It follows the convention of marking all hints that have the same number of
// bits set as the narrowest matching SocketAffinity with 'Preferred: true', and
// marking all others with 'Preferred: false'.
func (m *manager) generateCPUTopologyHints(availableCPUs cpuset.CPUSet, request int) []topologymanager.TopologyHint {
	// Discover topology in order to establish the number
	// of available CPUs per socket.
	topo, err := topology.Discover(m.machineInfo)
	if err != nil {
		klog.Warningf("[cpu manager] Error discovering topology for TopologyHint generation")
		return nil
	}

	// Initialize minAffinity to a full affinity mask.
	minAffinity, _ := socketmask.NewSocketMask()
	minAffinity.Fill()

	// Iterate through all combinations of socketMasks and build hints from them.
	hints := []topologymanager.TopologyHint{}
	socketmask.IterateSocketMasks(topo.CPUDetails.Sockets().ToSlice(), func(mask socketmask.SocketMask) {
		// Check to see if we have enough CPUs available on the current
		// SocketMask to satisfy the CPU request.
		numMatching := 0
		for _, c := range availableCPUs.ToSlice() {
			if mask.IsSet(topo.CPUDetails[c].SocketID) {
				numMatching++
			}
		}

		// If we don't, then move onto the next combination.
		if numMatching < request {
			return
		}

		// Otherwise, create a new hint from the SocketMask and add it to the
		// list of hints.  We set all hint preferences to 'false' on the first
		// pass through.
		hints = append(hints, topologymanager.TopologyHint{
			SocketAffinity: mask,
			Preferred:      false,
		})

		// Update minAffinity if relevant
		if mask.IsNarrowerThan(minAffinity) {
			minAffinity = mask
		}
	})

	// Loop back through all hints and update the 'Preferred' field based on
	// counting the number of bits sets in the affinity mask and comparing it
	// to the minAffinity. Only those with an equal number of bits set will be
	// considered preferred.
	for i := range hints {
		if hints[i].SocketAffinity.Count() == minAffinity.Count() {
			hints[i].Preferred = true
		}
	}

	return hints
}
