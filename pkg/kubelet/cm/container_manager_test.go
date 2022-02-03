/*
Copyright 2022 The Kubernetes Authors.

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

package cm

import (
	"testing"

	"github.com/stretchr/testify/require"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	podresourcesapi "k8s.io/kubelet/pkg/apis/podresources/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/devicemanager"
)

func TestContainerDevicesFromResourceDeviceInstances(t *testing.T) {
	instances := devicemanager.ResourceDeviceInstances{
		"foo": devicemanager.DeviceInstances{
			"dev-foo1": pluginapi.Device{
				ID: "foo1",
				Topology: &pluginapi.TopologyInfo{
					Nodes: []*pluginapi.NUMANode{
						{
							ID: 0,
						},
					},
				},
			},
			"dev-foo2": pluginapi.Device{
				ID: "foo2",
			},
		},
		"bar": devicemanager.DeviceInstances{
			"dev-bar3": pluginapi.Device{
				ID: "bar3",
				Topology: &pluginapi.TopologyInfo{
					Nodes: []*pluginapi.NUMANode{
						{
							ID: 0,
						},
					},
				},
			},
		},
		"baz": devicemanager.DeviceInstances{
			"dev-baz2": pluginapi.Device{
				ID: "baz2",
			},
		},
	}

	expected := []*podresourcesapi.ContainerDevices{
		{
			ResourceName: "foo",
			DeviceIds:    []string{"dev-foo1"},
			Topology: &podresourcesapi.TopologyInfo{
				Nodes: []*podresourcesapi.NUMANode{
					{
						ID: 0,
					},
				},
			},
		},
		{
			ResourceName: "foo",
			DeviceIds:    []string{"dev-foo2"},
		},
		{
			ResourceName: "bar",
			DeviceIds:    []string{"dev-bar3"},
			Topology: &podresourcesapi.TopologyInfo{
				Nodes: []*podresourcesapi.NUMANode{
					{
						ID: 0,
					},
				},
			},
		},
		{
			ResourceName: "baz",
			DeviceIds:    []string{"dev-baz2"},
		},
	}

	converted := containerDevicesFromResourceDeviceInstances(instances)

	require.ElementsMatch(t, expected, converted)
}
