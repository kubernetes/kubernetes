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

package devicemanager

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/ktesting"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
)

// markAllDevicesHealthy marks every registered device as healthy, the normal
// state of a node. calculateDeviceScore reads healthyDevices for its allocatable
// pool, so a fixture that sets allDevices alone describes a node whose hardware
// has all failed.
func markAllDevicesHealthy(m *ManagerImpl) {
	m.healthyDevices = map[string]sets.Set[string]{}
	for resource, devices := range m.allDevices {
		m.healthyDevices[resource] = sets.New[string]()
		for deviceID := range devices {
			m.healthyDevices[resource].Insert(deviceID)
		}
	}
}

func TestDeviceManagerScoreCalculation(t *testing.T) {
	resource := "gpu"

	testCases := []struct {
		name             string
		allDevices       map[string]DeviceInstances
		allocatedDevices map[string]sets.Set[string]
		// unhealthyDevices lists device IDs that are registered but not
		// allocatable. Everything else in allDevices is treated as healthy.
		unhealthyDevices sets.Set[string]
		numaNodes        []int
		expectedScores   map[int]int64 // NUMA node ID -> expected score
	}{
		{
			name: "no devices allocated",
			allDevices: map[string]DeviceInstances{
				resource: {
					"gpu0": {ID: "gpu0", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu1": {ID: "gpu1", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu2": {ID: "gpu2", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu3": {ID: "gpu3", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
				},
			},
			allocatedDevices: map[string]sets.Set[string]{
				resource: sets.New[string](),
			},
			numaNodes: []int{0},
			expectedScores: map[int]int64{
				0: 1, // max(1, 0/4) = 1 (scored but empty)
			},
		},
		{
			name: "half devices allocated",
			allDevices: map[string]DeviceInstances{
				resource: {
					"gpu0": {ID: "gpu0", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu1": {ID: "gpu1", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu2": {ID: "gpu2", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu3": {ID: "gpu3", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
				},
			},
			allocatedDevices: map[string]sets.Set[string]{
				resource: sets.New[string]("gpu0", "gpu1"),
			},
			numaNodes: []int{0},
			expectedScores: map[int]int64{
				0: 50, // 2/4 = 50%
			},
		},
		{
			name: "all devices allocated",
			allDevices: map[string]DeviceInstances{
				resource: {
					"gpu0": {ID: "gpu0", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu1": {ID: "gpu1", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
				},
			},
			allocatedDevices: map[string]sets.Set[string]{
				resource: sets.New[string]("gpu0", "gpu1"),
			},
			numaNodes: []int{0},
			expectedScores: map[int]int64{
				0: 100, // 2/2 = 100%
			},
		},
		{
			name: "asymmetric allocation across NUMA nodes",
			allDevices: map[string]DeviceInstances{
				resource: {
					"gpu0": {ID: "gpu0", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu1": {ID: "gpu1", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu2": {ID: "gpu2", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu3": {ID: "gpu3", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu4": {ID: "gpu4", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 1}}}},
					"gpu5": {ID: "gpu5", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 1}}}},
					"gpu6": {ID: "gpu6", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 1}}}},
					"gpu7": {ID: "gpu7", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 1}}}},
				},
			},
			allocatedDevices: map[string]sets.Set[string]{
				resource: sets.New[string]("gpu0", "gpu1", "gpu2", "gpu4"), // 3/4 on NUMA 0, 1/4 on NUMA 1
			},
			numaNodes: []int{0, 1},
			expectedScores: map[int]int64{
				0: 75, // 3/4 = 75%
				1: 25, // 1/4 = 25%
			},
		},
		{
			// Devices per NUMA zone are few and not always a power of two, so
			// most real scores land on a division that does not terminate and
			// get truncated: 1 of 3 is 33.3 and 2 of 3 is 66.6.
			name: "odd device counts truncate towards zero",
			allDevices: map[string]DeviceInstances{
				resource: {
					"gpu0": {ID: "gpu0", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu1": {ID: "gpu1", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu2": {ID: "gpu2", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu3": {ID: "gpu3", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 1}}}},
					"gpu4": {ID: "gpu4", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 1}}}},
					"gpu5": {ID: "gpu5", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 1}}}},
				},
			},
			allocatedDevices: map[string]sets.Set[string]{
				resource: sets.New[string]("gpu0", "gpu3", "gpu4"),
			},
			numaNodes: []int{0, 1},
			expectedScores: map[int]int64{
				0: 33, // 1*100/3 = 33.3 -> 33
				1: 66, // 2*100/3 = 66.6 -> 66
			},
		},
		{
			name: "rounding for small percentages",
			allDevices: map[string]DeviceInstances{
				resource: {
					"gpu0": {ID: "gpu0", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu1": {ID: "gpu1", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu2": {ID: "gpu2", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu3": {ID: "gpu3", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu4": {ID: "gpu4", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu5": {ID: "gpu5", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu6": {ID: "gpu6", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu7": {ID: "gpu7", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
				},
			},
			allocatedDevices: map[string]sets.Set[string]{
				resource: sets.New[string]("gpu0"), // 1/8 = 12.5% -> rounds to 12
			},
			numaNodes: []int{0},
			expectedScores: map[int]int64{
				0: 12, // 1*100/8 = 12 (integer division)
			},
		},
		{
			name: "1 of 20 allocated, low utilization",
			allDevices: map[string]DeviceInstances{
				resource: {
					"gpu0":  {ID: "gpu0", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu1":  {ID: "gpu1", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu2":  {ID: "gpu2", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu3":  {ID: "gpu3", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu4":  {ID: "gpu4", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu5":  {ID: "gpu5", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu6":  {ID: "gpu6", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu7":  {ID: "gpu7", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu8":  {ID: "gpu8", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu9":  {ID: "gpu9", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu10": {ID: "gpu10", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu11": {ID: "gpu11", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu12": {ID: "gpu12", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu13": {ID: "gpu13", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu14": {ID: "gpu14", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu15": {ID: "gpu15", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu16": {ID: "gpu16", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu17": {ID: "gpu17", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu18": {ID: "gpu18", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu19": {ID: "gpu19", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
				},
			},
			allocatedDevices: map[string]sets.Set[string]{
				resource: sets.New[string]("gpu0"),
			},
			numaNodes: []int{0},
			expectedScores: map[int]int64{
				0: 5, // 1*100/20 = 5
			},
		},
		{
			// Unhealthy devices are not allocatable, so they leave the
			// denominator. Counting them would report 4 of 8 = 50 and make a
			// node that has just lost hardware look emptier than the identical
			// node next to it, which is the wrong way round for packing.
			name: "unhealthy devices leave the allocatable pool",
			allDevices: map[string]DeviceInstances{
				resource: {
					"gpu0": {ID: "gpu0", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu1": {ID: "gpu1", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu2": {ID: "gpu2", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu3": {ID: "gpu3", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu4": {ID: "gpu4", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu5": {ID: "gpu5", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu6": {ID: "gpu6", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
					"gpu7": {ID: "gpu7", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
				},
			},
			allocatedDevices: map[string]sets.Set[string]{
				resource: sets.New[string]("gpu0", "gpu1", "gpu2", "gpu3"),
			},
			unhealthyDevices: sets.New[string]("gpu6", "gpu7"),
			numaNodes:        []int{0},
			expectedScores: map[int]int64{
				0: 66, // 4 allocated of 6 healthy = 66, not 4 of 8 = 50
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			healthyDevices := map[string]sets.Set[string]{}
			for res, devices := range tc.allDevices {
				healthyDevices[res] = sets.New[string]()
				for deviceID := range devices {
					if !tc.unhealthyDevices.Has(deviceID) {
						healthyDevices[res].Insert(deviceID)
					}
				}
			}

			m := ManagerImpl{
				allDevices:       tc.allDevices,
				healthyDevices:   healthyDevices,
				allocatedDevices: tc.allocatedDevices,
				numaNodes:        tc.numaNodes,
			}

			for numaNode, expectedScore := range tc.expectedScores {
				mask, _ := bitmask.NewBitMask(numaNode)
				score := m.calculateDeviceScore(resource, mask)
				if score != expectedScore {
					t.Errorf("NUMA node %d: expected score %d, got %d", numaNode, expectedScore, score)
				}
			}
		})
	}
}

func TestGenerateDeviceTopologyHintsWithScore(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	resource := "gpu"

	t.Run("hints include score based on allocation", func(t *testing.T) {
		m := ManagerImpl{
			allDevices: NewResourceDeviceInstances(),
			allocatedDevices: map[string]sets.Set[string]{
				resource: sets.New[string]("gpu0", "gpu1"), // 2 allocated
			},
			numaNodes: []int{0, 1},
		}
		m.allDevices[resource] = DeviceInstances{
			"gpu0": {ID: "gpu0", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
			"gpu1": {ID: "gpu1", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
			"gpu2": {ID: "gpu2", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
			"gpu3": {ID: "gpu3", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 0}}}},
			"gpu4": {ID: "gpu4", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 1}}}},
			"gpu5": {ID: "gpu5", Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: 1}}}},
		}
		markAllDevicesHealthy(&m)

		// Request 1 device from the available pool (gpu2, gpu3, gpu4, gpu5)
		hints := m.generateDeviceTopologyHints(logger, resource, sets.New[string]("gpu2", "gpu3", "gpu4", "gpu5"), nil, 1)

		// Find hints for each NUMA node
		var node0Hint, node1Hint *int64
		for i, hint := range hints {
			if hint.NUMANodeAffinity.IsSet(0) && hint.NUMANodeAffinity.Count() == 1 {
				node0Hint = &hints[i].Score
			}
			if hint.NUMANodeAffinity.IsSet(1) && hint.NUMANodeAffinity.Count() == 1 {
				node1Hint = &hints[i].Score
			}
		}

		if node0Hint == nil {
			t.Fatal("expected hint for NUMA node 0")
		}
		if node1Hint == nil {
			t.Fatal("expected hint for NUMA node 1")
		}

		// NUMA 0: 2 allocated out of 4 total = 50%
		if *node0Hint != 50 {
			t.Errorf("NUMA 0 hint: expected score 50, got %d", *node0Hint)
		}

		// NUMA 1: 0 allocated out of 2 total = max(1, 0) = 1 (scored but empty)
		if *node1Hint != 1 {
			t.Errorf("NUMA 1 hint: expected score 1, got %d", *node1Hint)
		}
	})
}
