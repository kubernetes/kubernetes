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
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/socketmask"
)

func TestGetTopologyHints(t *testing.T) {
	tcases := []struct {
		name     string
		amount   int64
		expected topologymanager.TopologyHints
	}{
		{
			name:   "Socket Affinity includes {0, 1}, {1, 0}, {1, 1}",
			amount: 1,
			expected: topologymanager.TopologyHints{
				SocketAffinity: []socketmask.SocketMask{{0, 1}, {1, 0}, {1, 1}},
				Affinity:       true,
			},
		},
		{
			name:   "Socket Affinity includes {1, 1}",
			amount: 2,
			expected: topologymanager.TopologyHints{
				SocketAffinity: []socketmask.SocketMask{{1, 1}},
				Affinity:       false,
			},
		},
	}

	for _, tc := range tcases {
		m := ManagerImpl{}

		testPod := v1.Pod{}
		testContainer := v1.Container{}

		testDevices := make(map[string][]pluginapi.Device)
		testDevice1 := pluginapi.Device{ID: "ID", Health: pluginapi.Healthy, Topology: &pluginapi.TopologyInfo{Socket: int64(0)}}
		testDevice2 := pluginapi.Device{ID: "ID", Health: pluginapi.Healthy, Topology: &pluginapi.TopologyInfo{Socket: int64(1)}}
		testDevices["testdevice"] = []pluginapi.Device{testDevice1, testDevice2}

		name := v1.ResourceName("testdevice")

		testResourceList := make(map[v1.ResourceName]resource.Quantity)
		testResourceList[name] = *resource.NewQuantity(tc.amount, "")

		testContainer.Resources.Requests = testResourceList
		actual := m.GetTopologyHints(testPod, testContainer)
		if reflect.DeepEqual(actual, tc.expected) {
			t.Errorf("Expected in result to be %v, got %v", tc.expected, actual)
		}
	}
}
