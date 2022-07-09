/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	podresourcesapi "k8s.io/kubelet/pkg/apis/podresources/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/devicemanager"
)

func Test_parsePercentage(t *testing.T) {

	// Part 1 of the test, iterate through all values 0-100%
	t.Run("validTestswithPercent", func(t *testing.T) {
		for i := 0; i < 100; i++ {
			value := fmt.Sprintf("%d", i) + "%"
			if p, e := parsePercentage(value); p != int64(i) || e != nil {
				t.Errorf("parsePercentage() failed for value %d, error: %v", i, e)
			}
		}
	})

	// Part 2 give a few error values
	tests := []struct {
		name    string
		v       string
		want    int64
		wantErr bool
	}{
		{
			name:    "invalid1",
			v:       "105%",
			want:    0,
			wantErr: true,
		},
		{
			name:    "invalid2",
			v:       "-87",
			want:    0,
			wantErr: true,
		},
		{
			name:    "invalid3",
			v:       "258",
			want:    0,
			wantErr: true,
		},
		{
			name:    "invalid4",
			v:       "-38%",
			want:    0,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parsePercentage(tt.v)
			if (err != nil) != tt.wantErr {
				t.Errorf("parsePercentage() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("parsePercentage() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestParseQOSReserved(t *testing.T) {
	// Part 1 Tests all valid combinations of memory type and percentage.
	t.Run("validTestswithPercent", func(t *testing.T) {
		for i := 0; i < 100; i++ {
			value := map[string]string{
				"memory": fmt.Sprintf("%d", i) + "%",
			}
			want := map[v1.ResourceName]int64{
				v1.ResourceName("memory"): int64(i),
			}
			if got, e := ParseQOSReserved(value); got != &want && e != nil {
				t.Errorf("ParseQOSReserved() failed, got %v, wanted %v, error was: %v", got, want, e)
			}
		}
	})

	// Part 2 tests a few error values
	tests := []struct {
		name    string
		m       map[string]string
		want    *map[v1.ResourceName]int64
		wantErr bool
	}{
		{
			name: "invalid1",
			m: map[string]string{
				"cpu": "86%",
			},
			want:    nil,
			wantErr: true,
		},
		{
			name: "invalid2",
			m: map[string]string{
				"cpu": "-85%",
			},
			want:    nil,
			wantErr: true,
		},
		{
			name: "invalid3",
			m: map[string]string{
				"memory": "105%",
			},
			want:    nil,
			wantErr: true,
		},
		{
			name: "invalid4",
			m: map[string]string{
				"memory": "-35",
			},
			want:    nil,
			wantErr: true,
		},
		{
			name: "invalid5",
			m: map[string]string{
				"cpu": "105%",
			},
			want:    nil,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseQOSReserved(tt.m)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseQOSReserved() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ParseQOSReserved() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_containerDevicesFromResourceDeviceInstances(t *testing.T) {
	var nilDevs []*podresourcesapi.ContainerDevices

	tests := []struct {
		name string
		devs devicemanager.ResourceDeviceInstances
		want []*podresourcesapi.ContainerDevices
	}{
		{
			name: "nil1",
			devs: devicemanager.NewResourceDeviceInstances(),
			want: nilDevs,
		},
		{
			name: "noNodesTest1",
			devs: devicemanager.ResourceDeviceInstances{
				"dev1": devicemanager.DeviceInstances{
					"node1": pluginapi.Device{
						Topology: nil,
					},
				},
			},
			want: []*podresourcesapi.ContainerDevices{
				{
					ResourceName: "dev1",
					DeviceIds:    []string{"node1"},
					Topology:     nil,
				},
			},
		},
		{
			name: "noNodesTest2",
			devs: devicemanager.ResourceDeviceInstances{
				"somedevice": devicemanager.DeviceInstances{
					"node1": pluginapi.Device{
						Topology: nil,
					},
					"node2": pluginapi.Device{
						Topology: nil,
					},
					"node3": pluginapi.Device{
						Topology: nil,
					},
				},
			},
			want: []*podresourcesapi.ContainerDevices{
				{
					ResourceName: "somedevice",
					DeviceIds:    []string{"node1"},
					Topology:     nil,
				},
				{
					ResourceName: "somedevice",
					DeviceIds:    []string{"node2"},
					Topology:     nil,
				},
				{
					ResourceName: "somedevice",
					DeviceIds:    []string{"node3"},
					Topology:     nil,
				},
			},
		},
		{
			name: "nodesTest1",
			devs: devicemanager.ResourceDeviceInstances{
				"somedevice": devicemanager.DeviceInstances{
					"instance1": pluginapi.Device{
						Topology: &pluginapi.TopologyInfo{
							Nodes: []*pluginapi.NUMANode{
								{
									ID: 1,
								},
							},
						},
					},
				},
			},
			want: []*podresourcesapi.ContainerDevices{
				{
					ResourceName: "somedevice",
					DeviceIds:    []string{"instance1"},
					Topology: &podresourcesapi.TopologyInfo{
						Nodes: []*podresourcesapi.NUMANode{
							{
								ID: 1,
							},
						},
					},
				},
			},
		},
		{
			name: "nodesTest2",
			devs: devicemanager.ResourceDeviceInstances{
				"somedevice": devicemanager.DeviceInstances{
					"instance1": pluginapi.Device{
						Topology: &pluginapi.TopologyInfo{
							Nodes: []*pluginapi.NUMANode{
								{
									ID: 1,
								},
								{
									ID: 2,
								},
								{
									ID: 3,
								},
							},
						},
					},
					"instance2": pluginapi.Device{
						Topology: &pluginapi.TopologyInfo{
							Nodes: []*pluginapi.NUMANode{
								{
									ID: 1,
								},
								{
									ID: 2,
								},
								{
									ID: 3,
								},
							},
						},
					},
				},
			},
			want: []*podresourcesapi.ContainerDevices{
				{
					ResourceName: "somedevice",
					DeviceIds:    []string{"instance1"},
					Topology: &podresourcesapi.TopologyInfo{
						Nodes: []*podresourcesapi.NUMANode{
							{
								ID: 1,
							},
						},
					},
				},
				{
					ResourceName: "somedevice",
					DeviceIds:    []string{"instance1"},
					Topology: &podresourcesapi.TopologyInfo{
						Nodes: []*podresourcesapi.NUMANode{
							{
								ID: 2,
							},
						},
					},
				},
				{
					ResourceName: "somedevice",
					DeviceIds:    []string{"instance1"},
					Topology: &podresourcesapi.TopologyInfo{
						Nodes: []*podresourcesapi.NUMANode{
							{
								ID: 3,
							},
						},
					},
				},
				{
					ResourceName: "somedevice",
					DeviceIds:    []string{"instance2"},
					Topology: &podresourcesapi.TopologyInfo{
						Nodes: []*podresourcesapi.NUMANode{
							{
								ID: 1,
							},
						},
					},
				},
				{
					ResourceName: "somedevice",
					DeviceIds:    []string{"instance2"},
					Topology: &podresourcesapi.TopologyInfo{
						Nodes: []*podresourcesapi.NUMANode{
							{
								ID: 2,
							},
						},
					},
				},
				{
					ResourceName: "somedevice",
					DeviceIds:    []string{"instance2"},
					Topology: &podresourcesapi.TopologyInfo{
						Nodes: []*podresourcesapi.NUMANode{
							{
								ID: 3,
							},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := containerDevicesFromResourceDeviceInstances(tt.devs); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("containerDevicesFromResourceDeviceInstances() = %v, want %v", got, tt.want)
			}
		})
	}
}
