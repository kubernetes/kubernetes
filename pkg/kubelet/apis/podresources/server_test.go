/*
Copyright 2018 The Kubernetes Authors.

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

package podresources

import (
	"context"
	"testing"

	"github.com/stretchr/testify/mock"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	podresourcesapi "k8s.io/kubelet/pkg/apis/podresources/v1alpha1"
)

type mockProvider struct {
	mock.Mock
}

func (m *mockProvider) GetPods() []*v1.Pod {
	args := m.Called()
	return args.Get(0).([]*v1.Pod)
}

func (m *mockProvider) GetDevices(podUID, containerName string) []*podresourcesapi.ContainerDevices {
	args := m.Called(podUID, containerName)
	return args.Get(0).([]*podresourcesapi.ContainerDevices)
}

func (m *mockProvider) UpdateAllocatedDevices() {
	m.Called()
}

func (m *mockProvider) GetAllDevices() map[string]map[string]pluginapi.Device {
	args := m.Called()
	return args.Get(0).(map[string]map[string]pluginapi.Device)
}

func (m *mockProvider) GetAllCPUs() []int64 {
	args := m.Called()
	return args.Get(0).([]int64)
}

func TestListPodResources(t *testing.T) {
	podName := "pod-name"
	podNamespace := "pod-namespace"
	podUID := types.UID("pod-uid")
	containerName := "container-name"

	devs := []*podresourcesapi.ContainerDevices{
		{
			ResourceName: "resource",
			DeviceIds:    []string{"dev0", "dev1"},
		},
	}

	for _, tc := range []struct {
		desc             string
		pods             []*v1.Pod
		devices          []*podresourcesapi.ContainerDevices
		expectedResponse *podresourcesapi.ListPodResourcesResponse
	}{
		{
			desc:             "no pods",
			pods:             []*v1.Pod{},
			devices:          []*podresourcesapi.ContainerDevices{},
			expectedResponse: &podresourcesapi.ListPodResourcesResponse{},
		},
		{
			desc: "pod without devices",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      podName,
						Namespace: podNamespace,
						UID:       podUID,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: containerName,
							},
						},
					},
				},
			},
			devices: []*podresourcesapi.ContainerDevices{},
			expectedResponse: &podresourcesapi.ListPodResourcesResponse{
				PodResources: []*podresourcesapi.PodResources{
					{
						Name:      podName,
						Namespace: podNamespace,
						Containers: []*podresourcesapi.ContainerResources{
							{
								Name:    containerName,
								Devices: []*podresourcesapi.ContainerDevices{},
							},
						},
					},
				},
			},
		},
		{
			desc: "pod with devices",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      podName,
						Namespace: podNamespace,
						UID:       podUID,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: containerName,
							},
						},
					},
				},
			},
			devices: devs,
			expectedResponse: &podresourcesapi.ListPodResourcesResponse{
				PodResources: []*podresourcesapi.PodResources{
					{
						Name:      podName,
						Namespace: podNamespace,
						Containers: []*podresourcesapi.ContainerResources{
							{
								Name:    containerName,
								Devices: devs,
							},
						},
					},
				},
			},
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			m := new(mockProvider)
			m.On("GetPods").Return(tc.pods)
			m.On("GetDevices", string(podUID), containerName).Return(tc.devices)
			m.On("UpdateAllocatedDevices").Return()
			server := NewPodResourcesServer(m, m, m)

			resp, err := server.List(context.TODO(), &podresourcesapi.ListPodResourcesRequest{})
			if err != nil {
				t.Errorf("want err = %v, got %q", nil, err)
			}
			if tc.expectedResponse.String() != resp.String() {
				t.Errorf("want resp = %s, got %s", tc.expectedResponse.String(), resp.String())
			}
		})
	}
}

func TestAllocatableResourcess(t *testing.T) {
	allDevs := map[string]map[string]pluginapi.Device{
		"resource": map[string]pluginapi.Device{
			"dev0": pluginapi.Device{
				ID:     "GPU-fef8089b-4820-abfc-e83e-94318197576e",
				Health: "Healthy",
				Topology: &pluginapi.TopologyInfo{
					Nodes: []*pluginapi.NUMANode{
						&pluginapi.NUMANode{
							ID: 0,
						},
					},
				},
			},
			"dev1": pluginapi.Device{
				ID:     "VF-8536e1e8-9dc6-4645-9aea-882db92e31e7",
				Health: "Healthy",
				Topology: &pluginapi.TopologyInfo{
					Nodes: []*pluginapi.NUMANode{
						&pluginapi.NUMANode{
							ID: 1,
						},
					},
				},
			},
		},
	}
	allCPUs := []int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

	for _, tc := range []struct {
		desc                                 string
		allCPUs                              []int64
		allDevices                           map[string]map[string]pluginapi.Device
		expectedAllocatableResourcesResponse *podresourcesapi.AllocatableResourcesResponse
	}{
		{
			desc:                                 "no devices, no CPUs",
			allCPUs:                              []int64{},
			allDevices:                           map[string]map[string]pluginapi.Device{},
			expectedAllocatableResourcesResponse: &podresourcesapi.AllocatableResourcesResponse{},
		},
		{
			desc:       "no devices, all CPUs",
			allCPUs:    allCPUs,
			allDevices: map[string]map[string]pluginapi.Device{},
			expectedAllocatableResourcesResponse: &podresourcesapi.AllocatableResourcesResponse{
				CpuIds: allCPUs,
			},
		},
		{
			desc:       "with devices, all CPUs",
			allCPUs:    allCPUs,
			allDevices: allDevs,
			expectedAllocatableResourcesResponse: &podresourcesapi.AllocatableResourcesResponse{
				CpuIds: allCPUs,
				Devices: []*podresourcesapi.ContainerDevices{
					&podresourcesapi.ContainerDevices{
						ResourceName: "resource",
						DeviceIds:    []string{"dev0", "dev1"},
					},
				},
			},
		},
		{
			desc:       "with devices, no CPUs",
			allCPUs:    []int64{},
			allDevices: allDevs,
			expectedAllocatableResourcesResponse: &podresourcesapi.AllocatableResourcesResponse{
				Devices: []*podresourcesapi.ContainerDevices{
					&podresourcesapi.ContainerDevices{
						ResourceName: "resource",
						DeviceIds:    []string{"dev0", "dev1"},
					},
				},
			},
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			m := new(mockProvider)
			m.On("UpdateAllocatedDevices").Return()
			m.On("GetAllDevices").Return(tc.allDevices)
			m.On("GetAllCPUs").Return(tc.allCPUs)
			server := NewPodResourcesServer(m, m, m)

			resp, err := server.GetAllocatableResources(context.TODO(), &podresourcesapi.AllocatableResourcesRequest{})
			if err != nil {
				t.Errorf("want err = %v, got %q", nil, err)
			}
			if tc.expectedAllocatableResourcesResponse.String() != resp.String() {
				t.Errorf("want resp = %s, got %s", tc.expectedAllocatableResourcesResponse.String(), resp.String())
			}
		})
	}
}
