/*
Copyright 2026 The Kubernetes Authors.

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
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
	ktesting "k8s.io/kubernetes/test/utils/ktesting"
)

func TestDeviceManagerScoreCalculation(t *testing.T) {
	resource := "gpu"

	testCases := []struct {
		name             string
		allDevices       map[string]DeviceInstances
		allocatedDevices map[string]sets.Set[string]
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
				0: 0, // 0/4 = 0%
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
			name: "ensure minimum score of 1 when any allocated",
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
				resource: sets.New[string]("gpu0"), // 1/200 would round to 0, but we enforce min score of 1
			},
			numaNodes: []int{0},
			expectedScores: map[int]int64{
				0: 1, // max(1, 1*100/200) = max(1, 0) = 1
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			m := ManagerImpl{
				allDevices:       tc.allDevices,
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

		// NUMA 1: 0 allocated out of 2 total = 0%
		if *node1Hint != 0 {
			t.Errorf("NUMA 1 hint: expected score 0, got %d", *node1Hint)
		}
	})
}
