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
	"reflect"
	"sort"
	"testing"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/socketmask"
)

func topologyHintLessThan(a topologymanager.TopologyHint, b topologymanager.TopologyHint) bool {
	if a.Preferred != b.Preferred {
		return a.Preferred == true
	}
	return a.SocketAffinity.IsNarrowerThan(b.SocketAffinity)
}

func TestGetTopologyHints(t *testing.T) {
	testPod1 := makePod("2", "2")
	testContainer1 := &testPod1.Spec.Containers[0]
	testPod2 := makePod("5", "5")
	testContainer2 := &testPod2.Spec.Containers[0]
	testPod3 := makePod("7", "7")
	testContainer3 := &testPod3.Spec.Containers[0]
	testPod4 := makePod("11", "11")
	testContainer4 := &testPod4.Spec.Containers[0]

	firstSocketMask, _ := socketmask.NewSocketMask(0)
	secondSocketMask, _ := socketmask.NewSocketMask(1)
	crossSocketMask, _ := socketmask.NewSocketMask(0, 1)

	m := manager{
		policy: &staticPolicy{},
		machineInfo: &cadvisorapi.MachineInfo{
			NumCores: 12,
			Topology: []cadvisorapi.Node{
				{Id: 0,
					Cores: []cadvisorapi.Core{
						{Id: 0, Threads: []int{0, 6}},
						{Id: 1, Threads: []int{1, 7}},
						{Id: 2, Threads: []int{2, 8}},
					},
				},
				{Id: 1,
					Cores: []cadvisorapi.Core{
						{Id: 0, Threads: []int{3, 9}},
						{Id: 1, Threads: []int{4, 10}},
						{Id: 2, Threads: []int{5, 11}},
					},
				},
			},
		},
		state: &mockState{
			defaultCPUSet: cpuset.NewCPUSet(2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
		},
	}

	tcases := []struct {
		name          string
		pod           v1.Pod
		container     v1.Container
		expectedHints []topologymanager.TopologyHint
	}{
		{
			name:      "Request 2 CPUs; 4 available on Socket 0, 6 available on Socket 1",
			pod:       *testPod1,
			container: *testContainer1,
			expectedHints: []topologymanager.TopologyHint{
				{
					SocketAffinity: firstSocketMask,
					Preferred:      true,
				},
				{
					SocketAffinity: secondSocketMask,
					Preferred:      true,
				},
				{
					SocketAffinity: crossSocketMask,
					Preferred:      false,
				},
			},
		},
		{
			name:      "Request 5 CPUs; 4 available on Socket 0, 6 available on Socket 1",
			pod:       *testPod2,
			container: *testContainer2,
			expectedHints: []topologymanager.TopologyHint{
				{
					SocketAffinity: secondSocketMask,
					Preferred:      true,
				},
				{
					SocketAffinity: crossSocketMask,
					Preferred:      false,
				},
			},
		},
		{
			name:      "Request 7 CPUs, 4 available on Socket 0, 6 available on Socket 1",
			pod:       *testPod3,
			container: *testContainer3,
			expectedHints: []topologymanager.TopologyHint{
				{
					SocketAffinity: crossSocketMask,
					Preferred:      true,
				},
			},
		},
		{
			name:          "Request 11 CPUs, 4 available on Socket 0, 6 available on Socket 1",
			pod:           *testPod4,
			container:     *testContainer4,
			expectedHints: nil,
		},
	}
	for _, tc := range tcases {
		hints := m.GetTopologyHints(tc.pod, tc.container)[string(v1.ResourceCPU)]
		if len(tc.expectedHints) == 0 && len(hints) == 0 {
			continue
		}
		sort.SliceStable(hints, func(i, j int) bool {
			return topologyHintLessThan(hints[i], hints[j])
		})
		sort.SliceStable(tc.expectedHints, func(i, j int) bool {
			return topologyHintLessThan(tc.expectedHints[i], tc.expectedHints[j])
		})
		if !reflect.DeepEqual(tc.expectedHints, hints) {
			t.Errorf("Expected in result to be %v , got %v", tc.expectedHints, hints)
		}
	}
}
