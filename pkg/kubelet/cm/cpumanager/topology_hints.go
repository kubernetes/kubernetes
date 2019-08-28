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
// bits set as the narrowest matching NUMANodeAffinity with 'Preferred: true', and
// marking all others with 'Preferred: false'.
func (m *manager) generateCPUTopologyHints(availableCPUs cpuset.CPUSet, request int) []topologymanager.TopologyHint {
	// Initialize minAffinitySize to include all NUMA Nodes.
	minAffinitySize := m.topology.CPUDetails.NUMANodes().Size()
	// Initialize minSocketsOnMinAffinity to include all Sockets.
	minSocketsOnMinAffinity := m.topology.CPUDetails.Sockets().Size()

	// Iterate through all combinations of socketMasks and build hints from them.
	hints := []topologymanager.TopologyHint{}
	socketmask.IterateSocketMasks(m.topology.CPUDetails.NUMANodes().ToSlice(), func(mask socketmask.SocketMask) {
		// First, update minAffinitySize and minSocketsOnMinAffinity for the
		// current request size.
		cpusInMask := m.topology.CPUDetails.CPUsInNUMANodes(mask.GetSockets()...).Size()
		socketsInMask := m.topology.CPUDetails.SocketsInNUMANodes(mask.GetSockets()...).Size()
		if cpusInMask >= request && mask.Count() < minAffinitySize {
			minAffinitySize = mask.Count()
			if socketsInMask < minSocketsOnMinAffinity {
				minSocketsOnMinAffinity = socketsInMask
			}
		}

		// Then check to see if we have enough CPUs available on the current
		// SocketMask to satisfy the CPU request.
		numMatching := 0
		for _, c := range availableCPUs.ToSlice() {
			if mask.IsSet(m.topology.CPUDetails[c].NUMANodeID) {
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
			NUMANodeAffinity: mask,
			Preferred:        false,
		})
	})

	// Loop back through all hints and update the 'Preferred' field based on
	// counting the number of bits sets in the affinity mask and comparing it
	// to the minAffinitySize. Only those with an equal number of bits set (and
	// with a minimal set of sockets) will be considered preferred.
	for i := range hints {
		if hints[i].NUMANodeAffinity.Count() == minAffinitySize {
			nodes := hints[i].NUMANodeAffinity.GetSockets()
			numSockets := m.topology.CPUDetails.SocketsInNUMANodes(nodes...).Size()
			if numSockets == minSocketsOnMinAffinity {
				hints[i].Preferred = true
			}
		}
	}

	return hints
}
