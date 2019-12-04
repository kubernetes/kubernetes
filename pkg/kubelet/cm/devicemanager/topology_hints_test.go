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
	"reflect"
	"sort"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
)

type mockAffinityStore struct {
	hint topologymanager.TopologyHint
}

func (m *mockAffinityStore) GetAffinity(podUID string, containerName string) topologymanager.TopologyHint {
	return m.hint
}

func makeNUMADevice(id string, numa int) pluginapi.Device {
	return pluginapi.Device{
		ID:       id,
		Topology: &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: int64(numa)}}},
	}
}

func makeSocketMask(sockets ...int) bitmask.BitMask {
	mask, _ := bitmask.NewBitMask(sockets...)
	return mask
}

func TestGetTopologyHints(t *testing.T) {
	tcases := []struct {
		description      string
		podUID           string
		containerName    string
		request          map[string]string
		devices          map[string][]pluginapi.Device
		allocatedDevices map[string]map[string]map[string][]string
		expectedHints    map[string][]topologymanager.TopologyHint
	}{
		{
			description:   "Single Request, no alignment",
			podUID:        "fakePod",
			containerName: "fakeContainer",
			request: map[string]string{
				"testdevice": "1",
			},
			devices: map[string][]pluginapi.Device{
				"testdevice": {
					{ID: "Dev1"},
					{ID: "Dev2"},
				},
			},
			expectedHints: map[string][]topologymanager.TopologyHint{
				"testdevice": nil,
			},
		},
		{
			description:   "Single Request, only one with alignment",
			podUID:        "fakePod",
			containerName: "fakeContainer",
			request: map[string]string{
				"testdevice": "1",
			},
			devices: map[string][]pluginapi.Device{
				"testdevice": {
					{ID: "Dev1"},
					makeNUMADevice("Dev2", 1),
				},
			},
			expectedHints: map[string][]topologymanager.TopologyHint{
				"testdevice": {
					{
						NUMANodeAffinity: makeSocketMask(1),
						Preferred:        true,
					},
					{
						NUMANodeAffinity: makeSocketMask(0, 1),
						Preferred:        false,
					},
				},
			},
		},
		{
			description:   "Single Request, one device per socket",
			podUID:        "fakePod",
			containerName: "fakeContainer",
			request: map[string]string{
				"testdevice": "1",
			},
			devices: map[string][]pluginapi.Device{
				"testdevice": {
					makeNUMADevice("Dev1", 0),
					makeNUMADevice("Dev2", 1),
				},
			},
			expectedHints: map[string][]topologymanager.TopologyHint{
				"testdevice": {
					{
						NUMANodeAffinity: makeSocketMask(0),
						Preferred:        true,
					},
					{
						NUMANodeAffinity: makeSocketMask(1),
						Preferred:        true,
					},
					{
						NUMANodeAffinity: makeSocketMask(0, 1),
						Preferred:        false,
					},
				},
			},
		},
		{
			description:   "Request for 2, one device per socket",
			podUID:        "fakePod",
			containerName: "fakeContainer",
			request: map[string]string{
				"testdevice": "2",
			},
			devices: map[string][]pluginapi.Device{
				"testdevice": {
					makeNUMADevice("Dev1", 0),
					makeNUMADevice("Dev2", 1),
				},
			},
			expectedHints: map[string][]topologymanager.TopologyHint{
				"testdevice": {
					{
						NUMANodeAffinity: makeSocketMask(0, 1),
						Preferred:        true,
					},
				},
			},
		},
		{
			description:   "Request for 2, 2 devices per socket",
			podUID:        "fakePod",
			containerName: "fakeContainer",
			request: map[string]string{
				"testdevice": "2",
			},
			devices: map[string][]pluginapi.Device{
				"testdevice": {
					makeNUMADevice("Dev1", 0),
					makeNUMADevice("Dev2", 1),
					makeNUMADevice("Dev3", 0),
					makeNUMADevice("Dev4", 1),
				},
			},
			expectedHints: map[string][]topologymanager.TopologyHint{
				"testdevice": {
					{
						NUMANodeAffinity: makeSocketMask(0),
						Preferred:        true,
					},
					{
						NUMANodeAffinity: makeSocketMask(1),
						Preferred:        true,
					},
					{
						NUMANodeAffinity: makeSocketMask(0, 1),
						Preferred:        false,
					},
				},
			},
		},
		{
			description:   "Request for 2, optimal on 1 NUMA node, forced cross-NUMA",
			podUID:        "fakePod",
			containerName: "fakeContainer",
			request: map[string]string{
				"testdevice": "2",
			},
			devices: map[string][]pluginapi.Device{
				"testdevice": {
					makeNUMADevice("Dev1", 0),
					makeNUMADevice("Dev2", 1),
					makeNUMADevice("Dev3", 0),
					makeNUMADevice("Dev4", 1),
				},
			},
			allocatedDevices: map[string]map[string]map[string][]string{
				"fakePod": {
					"fakeOtherContainer": {
						"testdevice": {"Dev1", "Dev2"},
					},
				},
			},
			expectedHints: map[string][]topologymanager.TopologyHint{
				"testdevice": {
					{
						NUMANodeAffinity: makeSocketMask(0, 1),
						Preferred:        false,
					},
				},
			},
		},
		{
			description:   "2 device types, mixed configuration",
			podUID:        "fakePod",
			containerName: "fakeContainer",
			request: map[string]string{
				"testdevice1": "2",
				"testdevice2": "1",
			},
			devices: map[string][]pluginapi.Device{
				"testdevice1": {
					makeNUMADevice("Dev1", 0),
					makeNUMADevice("Dev2", 1),
					makeNUMADevice("Dev3", 0),
					makeNUMADevice("Dev4", 1),
				},
				"testdevice2": {
					makeNUMADevice("Dev1", 0),
				},
			},
			expectedHints: map[string][]topologymanager.TopologyHint{
				"testdevice1": {
					{
						NUMANodeAffinity: makeSocketMask(0),
						Preferred:        true,
					},
					{
						NUMANodeAffinity: makeSocketMask(1),
						Preferred:        true,
					},
					{
						NUMANodeAffinity: makeSocketMask(0, 1),
						Preferred:        false,
					},
				},
				"testdevice2": {
					{
						NUMANodeAffinity: makeSocketMask(0),
						Preferred:        true,
					},
					{
						NUMANodeAffinity: makeSocketMask(0, 1),
						Preferred:        false,
					},
				},
			},
		},
		{
			description:   "Single device type, more requested than available",
			podUID:        "fakePod",
			containerName: "fakeContainer",
			request: map[string]string{
				"testdevice": "6",
			},
			devices: map[string][]pluginapi.Device{
				"testdevice": {
					makeNUMADevice("Dev1", 0),
					makeNUMADevice("Dev2", 0),
					makeNUMADevice("Dev3", 1),
					makeNUMADevice("Dev4", 1),
				},
			},
			expectedHints: map[string][]topologymanager.TopologyHint{
				"testdevice": {},
			},
		},
		{
			description:   "Single device type, all already allocated to container",
			podUID:        "fakePod",
			containerName: "fakeContainer",
			request: map[string]string{
				"testdevice": "2",
			},
			devices: map[string][]pluginapi.Device{
				"testdevice": {
					makeNUMADevice("Dev1", 0),
					makeNUMADevice("Dev2", 0),
				},
			},
			allocatedDevices: map[string]map[string]map[string][]string{
				"fakePod": {
					"fakeContainer": {
						"testdevice": {"Dev1", "Dev2"},
					},
				},
			},
			expectedHints: map[string][]topologymanager.TopologyHint{
				"testdevice": {
					{
						NUMANodeAffinity: makeSocketMask(0),
						Preferred:        true,
					},
					{
						NUMANodeAffinity: makeSocketMask(0, 1),
						Preferred:        false,
					},
				},
			},
		},
		{
			description:   "Single device type, less already allocated to container than requested",
			podUID:        "fakePod",
			containerName: "fakeContainer",
			request: map[string]string{
				"testdevice": "4",
			},
			devices: map[string][]pluginapi.Device{
				"testdevice": {
					makeNUMADevice("Dev1", 0),
					makeNUMADevice("Dev2", 0),
					makeNUMADevice("Dev3", 1),
					makeNUMADevice("Dev4", 1),
				},
			},
			allocatedDevices: map[string]map[string]map[string][]string{
				"fakePod": {
					"fakeContainer": {
						"testdevice": {"Dev1", "Dev2"},
					},
				},
			},
			expectedHints: map[string][]topologymanager.TopologyHint{
				"testdevice": {},
			},
		},
		{
			description:   "Single device type, more already allocated to container than requested",
			podUID:        "fakePod",
			containerName: "fakeContainer",
			request: map[string]string{
				"testdevice": "2",
			},
			devices: map[string][]pluginapi.Device{
				"testdevice": {
					makeNUMADevice("Dev1", 0),
					makeNUMADevice("Dev2", 0),
					makeNUMADevice("Dev3", 1),
					makeNUMADevice("Dev4", 1),
				},
			},
			allocatedDevices: map[string]map[string]map[string][]string{
				"fakePod": {
					"fakeContainer": {
						"testdevice": {"Dev1", "Dev2", "Dev3", "Dev4"},
					},
				},
			},
			expectedHints: map[string][]topologymanager.TopologyHint{
				"testdevice": {},
			},
		},
	}

	for _, tc := range tcases {
		resourceList := v1.ResourceList{}
		for r := range tc.request {
			resourceList[v1.ResourceName(r)] = resource.MustParse(tc.request[r])
		}

		pod := makePod(resourceList)
		pod.UID = types.UID(tc.podUID)
		pod.Spec.Containers[0].Name = tc.containerName

		m := ManagerImpl{
			allDevices:       make(map[string]map[string]pluginapi.Device),
			healthyDevices:   make(map[string]sets.String),
			allocatedDevices: make(map[string]sets.String),
			podDevices:       make(podDevices),
			sourcesReady:     &sourcesReadyStub{},
			activePods:       func() []*v1.Pod { return []*v1.Pod{pod} },
			numaNodes:        []int{0, 1},
		}

		for r := range tc.devices {
			m.allDevices[r] = make(map[string]pluginapi.Device)
			m.healthyDevices[r] = sets.NewString()

			for _, d := range tc.devices[r] {
				m.allDevices[r][d.ID] = d
				m.healthyDevices[r].Insert(d.ID)
			}
		}

		for p := range tc.allocatedDevices {
			for c := range tc.allocatedDevices[p] {
				for r, devices := range tc.allocatedDevices[p][c] {
					m.podDevices.insert(p, c, r, sets.NewString(devices...), nil)

					m.allocatedDevices[r] = sets.NewString()
					for _, d := range devices {
						m.allocatedDevices[r].Insert(d)
					}
				}
			}
		}

		hints := m.GetTopologyHints(*pod, pod.Spec.Containers[0])

		for r := range tc.expectedHints {
			sort.SliceStable(hints[r], func(i, j int) bool {
				return hints[r][i].LessThan(hints[r][j])
			})
			sort.SliceStable(tc.expectedHints[r], func(i, j int) bool {
				return tc.expectedHints[r][i].LessThan(tc.expectedHints[r][j])
			})
			if !reflect.DeepEqual(hints[r], tc.expectedHints[r]) {
				t.Errorf("%v: Expected result to be %v, got %v", tc.description, tc.expectedHints[r], hints[r])
			}
		}
	}
}

func TestTopologyAlignedAllocation(t *testing.T) {
	tcases := []struct {
		description        string
		resource           string
		request            int
		devices            []pluginapi.Device
		allocatedDevices   []string
		hint               topologymanager.TopologyHint
		expectedAllocation int
		expectedAlignment  map[int]int
	}{
		{
			description: "Single Request, no alignment",
			resource:    "resource",
			request:     1,
			devices: []pluginapi.Device{
				{ID: "Dev1"},
				{ID: "Dev2"},
			},
			hint: topologymanager.TopologyHint{
				NUMANodeAffinity: makeSocketMask(0, 1),
				Preferred:        true,
			},
			expectedAllocation: 1,
			expectedAlignment:  map[int]int{},
		},
		{
			description: "Request for 1, partial alignment",
			resource:    "resource",
			request:     1,
			devices: []pluginapi.Device{
				{ID: "Dev1"},
				makeNUMADevice("Dev2", 1),
			},
			hint: topologymanager.TopologyHint{
				NUMANodeAffinity: makeSocketMask(1),
				Preferred:        true,
			},
			expectedAllocation: 1,
			expectedAlignment:  map[int]int{1: 1},
		},
		{
			description: "Single Request, socket 0",
			resource:    "resource",
			request:     1,
			devices: []pluginapi.Device{
				makeNUMADevice("Dev1", 0),
				makeNUMADevice("Dev2", 1),
			},
			hint: topologymanager.TopologyHint{
				NUMANodeAffinity: makeSocketMask(0),
				Preferred:        true,
			},
			expectedAllocation: 1,
			expectedAlignment:  map[int]int{0: 1},
		},
		{
			description: "Single Request, socket 1",
			resource:    "resource",
			request:     1,
			devices: []pluginapi.Device{
				makeNUMADevice("Dev1", 0),
				makeNUMADevice("Dev2", 1),
			},
			hint: topologymanager.TopologyHint{
				NUMANodeAffinity: makeSocketMask(1),
				Preferred:        true,
			},
			expectedAllocation: 1,
			expectedAlignment:  map[int]int{1: 1},
		},
		{
			description: "Request for 2, socket 0",
			resource:    "resource",
			request:     2,
			devices: []pluginapi.Device{
				makeNUMADevice("Dev1", 0),
				makeNUMADevice("Dev2", 1),
				makeNUMADevice("Dev3", 0),
				makeNUMADevice("Dev4", 1),
			},
			hint: topologymanager.TopologyHint{
				NUMANodeAffinity: makeSocketMask(0),
				Preferred:        true,
			},
			expectedAllocation: 2,
			expectedAlignment:  map[int]int{0: 2},
		},
		{
			description: "Request for 2, socket 1",
			resource:    "resource",
			request:     2,
			devices: []pluginapi.Device{
				makeNUMADevice("Dev1", 0),
				makeNUMADevice("Dev2", 1),
				makeNUMADevice("Dev3", 0),
				makeNUMADevice("Dev4", 1),
			},
			hint: topologymanager.TopologyHint{
				NUMANodeAffinity: makeSocketMask(1),
				Preferred:        true,
			},
			expectedAllocation: 2,
			expectedAlignment:  map[int]int{1: 2},
		},
		{
			description: "Request for 4, unsatisfiable, prefer socket 0",
			resource:    "resource",
			request:     4,
			devices: []pluginapi.Device{
				makeNUMADevice("Dev1", 0),
				makeNUMADevice("Dev2", 1),
				makeNUMADevice("Dev3", 0),
				makeNUMADevice("Dev4", 1),
				makeNUMADevice("Dev5", 0),
				makeNUMADevice("Dev6", 1),
			},
			hint: topologymanager.TopologyHint{
				NUMANodeAffinity: makeSocketMask(0),
				Preferred:        true,
			},
			expectedAllocation: 4,
			expectedAlignment:  map[int]int{0: 3, 1: 1},
		},
		{
			description: "Request for 4, unsatisfiable, prefer socket 1",
			resource:    "resource",
			request:     4,
			devices: []pluginapi.Device{
				makeNUMADevice("Dev1", 0),
				makeNUMADevice("Dev2", 1),
				makeNUMADevice("Dev3", 0),
				makeNUMADevice("Dev4", 1),
				makeNUMADevice("Dev5", 0),
				makeNUMADevice("Dev6", 1),
			},
			hint: topologymanager.TopologyHint{
				NUMANodeAffinity: makeSocketMask(1),
				Preferred:        true,
			},
			expectedAllocation: 4,
			expectedAlignment:  map[int]int{0: 1, 1: 3},
		},
		{
			description: "Request for 4, multisocket",
			resource:    "resource",
			request:     4,
			devices: []pluginapi.Device{
				makeNUMADevice("Dev1", 0),
				makeNUMADevice("Dev2", 1),
				makeNUMADevice("Dev3", 2),
				makeNUMADevice("Dev4", 3),
				makeNUMADevice("Dev5", 0),
				makeNUMADevice("Dev6", 1),
				makeNUMADevice("Dev7", 2),
				makeNUMADevice("Dev8", 3),
			},
			hint: topologymanager.TopologyHint{
				NUMANodeAffinity: makeSocketMask(1, 3),
				Preferred:        true,
			},
			expectedAllocation: 4,
			expectedAlignment:  map[int]int{1: 2, 3: 2},
		},
	}
	for _, tc := range tcases {
		m := ManagerImpl{
			allDevices:            make(map[string]map[string]pluginapi.Device),
			healthyDevices:        make(map[string]sets.String),
			allocatedDevices:      make(map[string]sets.String),
			podDevices:            make(podDevices),
			sourcesReady:          &sourcesReadyStub{},
			activePods:            func() []*v1.Pod { return []*v1.Pod{} },
			topologyAffinityStore: &mockAffinityStore{tc.hint},
		}

		m.allDevices[tc.resource] = make(map[string]pluginapi.Device)
		m.healthyDevices[tc.resource] = sets.NewString()

		for _, d := range tc.devices {
			m.allDevices[tc.resource][d.ID] = d
			m.healthyDevices[tc.resource].Insert(d.ID)
		}

		allocated, err := m.devicesToAllocate("podUID", "containerName", tc.resource, tc.request, sets.NewString())
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}

		if len(allocated) != tc.expectedAllocation {
			t.Errorf("%v. expected allocation: %v but got: %v", tc.description, tc.expectedAllocation, len(allocated))
		}

		alignment := make(map[int]int)
		if m.deviceHasTopologyAlignment(tc.resource) {
			for d := range allocated {
				if m.allDevices[tc.resource][d].Topology != nil {
					alignment[int(m.allDevices[tc.resource][d].Topology.Nodes[0].ID)]++
				}
			}
		}

		if !reflect.DeepEqual(alignment, tc.expectedAlignment) {
			t.Errorf("%v. expected alignment: %v but got: %v", tc.description, tc.expectedAlignment, alignment)
		}
	}
}
