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

package devicemanager

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
)

// GetTopologyHints implements the TopologyManager HintProvider Interface which
// ensures the Device Manager is consulted when Topology Aware Hints for each
// container are created.
func (m *ManagerImpl) GetTopologyHints(pod *v1.Pod, container *v1.Container) map[string][]topologymanager.TopologyHint {
	// Garbage collect any stranded device resources before providing TopologyHints
	m.UpdateAllocatedDevices()

	// Loop through all device resources and generate TopologyHints for them..
	deviceHints := make(map[string][]topologymanager.TopologyHint)
	for resourceObj, requestedObj := range container.Resources.Limits {
		resource := string(resourceObj)
		requested := int(requestedObj.Value())

		// Only consider resources associated with a device plugin.
		if m.isDevicePluginResource(resource) {
			// Only consider devices that actually container topology information.
			if aligned := m.deviceHasTopologyAlignment(resource); !aligned {
				klog.Infof("[devicemanager] Resource '%v' does not have a topology preference", resource)
				deviceHints[resource] = nil
				continue
			}

			// Short circuit to regenerate the same hints if there are already
			// devices allocated to the Container. This might happen after a
			// kubelet restart, for example.
			allocated := m.podDevices.containerDevices(string(pod.UID), container.Name, resource)
			if allocated.Len() > 0 {
				if allocated.Len() != requested {
					klog.Errorf("[devicemanager] Resource '%v' already allocated to (pod %v, container %v) with different number than request: requested: %d, allocated: %d", resource, string(pod.UID), container.Name, requested, allocated.Len())
					deviceHints[resource] = []topologymanager.TopologyHint{}
					continue
				}
				klog.Infof("[devicemanager] Regenerating TopologyHints for resource '%v' already allocated to (pod %v, container %v)", resource, string(pod.UID), container.Name)
				deviceHints[resource] = m.generateDeviceTopologyHints(resource, allocated, sets.String{}, requested)
				continue
			}

			// Get the list of available devices, for which TopologyHints should be generated.
			available := m.getAvailableDevices(resource)
			reusable := m.devicesToReuse[string(pod.UID)][resource]
			if available.Union(reusable).Len() < requested {
				klog.Errorf("[devicemanager] Unable to generate topology hints: requested number of devices unavailable for '%s': requested: %d, available: %d", resource, requested, available.Union(reusable).Len())
				deviceHints[resource] = []topologymanager.TopologyHint{}
				continue
			}

			// Generate TopologyHints for this resource given the current
			// request size and the list of available devices.
			deviceHints[resource] = m.generateDeviceTopologyHints(resource, available, reusable, requested)
		}
	}

	return deviceHints
}

func (m *ManagerImpl) deviceHasTopologyAlignment(resource string) bool {
	// If any device has Topology set, we assume they care about alignment.
	for device := range m.allDevices[resource] {
		if m.allDevices[resource][device].Topology != nil {
			return true
		}
	}
	return false
}

func (m *ManagerImpl) getAvailableDevices(resource string) sets.String {
	// Strip all devices in use from the list of healthy ones.
	return m.healthyDevices[resource].Difference(m.allocatedDevices[resource])
}

func (m *ManagerImpl) generateDeviceTopologyHints(resource string, available sets.String, reusable sets.String, request int) []topologymanager.TopologyHint {
	// Initialize minAffinitySize to include all NUMA Nodes
	minAffinitySize := len(m.numaNodes)

	// Iterate through all combinations of NUMA Nodes and build hints from them.
	hints := []topologymanager.TopologyHint{}
	bitmask.IterateBitMasks(m.numaNodes, func(mask bitmask.BitMask) {
		// First, update minAffinitySize for the current request size.
		devicesInMask := 0
		for _, device := range m.allDevices[resource] {
			if mask.AnySet(m.getNUMANodeIds(device.Topology)) {
				devicesInMask++
			}
		}
		if devicesInMask >= request && mask.Count() < minAffinitySize {
			minAffinitySize = mask.Count()
		}

		// Then check to see if all of the reusable devices are part of the bitmask.
		numMatching := 0
		for d := range reusable {
			// Skip the device if it doesn't specify any topology info.
			if m.allDevices[resource][d].Topology == nil {
				continue
			}
			// Otherwise disregard this mask if its NUMANode isn't part of it.
			if !mask.AnySet(m.getNUMANodeIds(m.allDevices[resource][d].Topology)) {
				return
			}
			numMatching++
		}

		// Finally, check to see if enough available devices remain on the
		// current NUMA node combination to satisfy the device request.
		for d := range available {
			if mask.AnySet(m.getNUMANodeIds(m.allDevices[resource][d].Topology)) {
				numMatching++
			}
		}

		// If they don't, then move onto the next combination.
		if numMatching < request {
			return
		}

		// Otherwise, create a new hint from the NUMA mask and add it to the
		// list of hints.  We set all hint preferences to 'false' on the first
		// pass through.
		hints = append(hints, topologymanager.TopologyHint{
			NUMANodeAffinity: mask,
			Preferred:        false,
		})
	})

	// Loop back through all hints and update the 'Preferred' field based on
	// counting the number of bits sets in the affinity mask and comparing it
	// to the minAffinity. Only those with an equal number of bits set will be
	// considered preferred.
	for i := range hints {
		if hints[i].NUMANodeAffinity.Count() == minAffinitySize {
			hints[i].Preferred = true
		}
	}

	return hints
}

func (m *ManagerImpl) getNUMANodeIds(topology *pluginapi.TopologyInfo) []int {
	if topology == nil {
		return nil
	}
	var ids []int
	for _, n := range topology.Nodes {
		ids = append(ids, int(n.ID))
	}
	return ids
}
