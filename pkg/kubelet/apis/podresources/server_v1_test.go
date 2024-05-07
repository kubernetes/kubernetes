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
	"fmt"
	"reflect"
	"sort"
	"testing"

	"github.com/golang/mock/gomock"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	podresourcesapi "k8s.io/kubelet/pkg/apis/podresources/v1"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
	podresourcetest "k8s.io/kubernetes/pkg/kubelet/apis/podresources/testing"
)

func TestListPodResourcesV1(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.KubeletPodResourcesDynamicResources, true)()

	podName := "pod-name"
	podNamespace := "pod-namespace"
	podUID := types.UID("pod-uid")
	containerName := "container-name"
	numaID := int64(1)

	mockCtrl := gomock.NewController(t)
	defer mockCtrl.Finish()

	devs := []*podresourcesapi.ContainerDevices{
		{
			ResourceName: "resource",
			DeviceIds:    []string{"dev0", "dev1"},
			Topology:     &podresourcesapi.TopologyInfo{Nodes: []*podresourcesapi.NUMANode{{ID: numaID}}},
		},
	}

	cpus := []int64{12, 23, 30}

	memory := []*podresourcesapi.ContainerMemory{
		{
			MemoryType: "memory",
			Size_:      1073741824,
			Topology:   &podresourcesapi.TopologyInfo{Nodes: []*podresourcesapi.NUMANode{{ID: numaID}}},
		},
		{
			MemoryType: "hugepages-1Gi",
			Size_:      1073741824,
			Topology:   &podresourcesapi.TopologyInfo{Nodes: []*podresourcesapi.NUMANode{{ID: numaID}}},
		},
	}

	containers := []v1.Container{
		{
			Name: containerName,
		},
	}
	pods := []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      podName,
				Namespace: podNamespace,
				UID:       podUID,
			},
			Spec: v1.PodSpec{
				Containers: containers,
			},
		},
	}

	pluginCDIDevices := []*podresourcesapi.CDIDevice{{Name: "dra-dev0"}, {Name: "dra-dev1"}}
	draDevs := []*podresourcesapi.DynamicResource{
		{
			ClassName:      "resource-class",
			ClaimName:      "claim-name",
			ClaimNamespace: "default",
			ClaimResources: []*podresourcesapi.ClaimResource{{CDIDevices: pluginCDIDevices}},
		},
	}

	for _, tc := range []struct {
		desc             string
		pods             []*v1.Pod
		devices          []*podresourcesapi.ContainerDevices
		cpus             []int64
		memory           []*podresourcesapi.ContainerMemory
		dynamicResources []*podresourcesapi.DynamicResource
		expectedResponse *podresourcesapi.ListPodResourcesResponse
	}{
		{
			desc:             "no pods",
			pods:             []*v1.Pod{},
			devices:          []*podresourcesapi.ContainerDevices{},
			cpus:             []int64{},
			memory:           []*podresourcesapi.ContainerMemory{},
			dynamicResources: []*podresourcesapi.DynamicResource{},
			expectedResponse: &podresourcesapi.ListPodResourcesResponse{},
		},
		{
			desc:             "pod without devices",
			pods:             pods,
			devices:          []*podresourcesapi.ContainerDevices{},
			cpus:             []int64{},
			memory:           []*podresourcesapi.ContainerMemory{},
			dynamicResources: []*podresourcesapi.DynamicResource{},
			expectedResponse: &podresourcesapi.ListPodResourcesResponse{
				PodResources: []*podresourcesapi.PodResources{
					{
						Name:      podName,
						Namespace: podNamespace,
						Containers: []*podresourcesapi.ContainerResources{
							{
								Name:             containerName,
								Devices:          []*podresourcesapi.ContainerDevices{},
								DynamicResources: []*podresourcesapi.DynamicResource{},
							},
						},
					},
				},
			},
		},
		{
			desc:             "pod with devices",
			pods:             pods,
			devices:          devs,
			cpus:             cpus,
			memory:           memory,
			dynamicResources: []*podresourcesapi.DynamicResource{},
			expectedResponse: &podresourcesapi.ListPodResourcesResponse{
				PodResources: []*podresourcesapi.PodResources{
					{
						Name:      podName,
						Namespace: podNamespace,
						Containers: []*podresourcesapi.ContainerResources{
							{
								Name:             containerName,
								Devices:          devs,
								CpuIds:           cpus,
								Memory:           memory,
								DynamicResources: []*podresourcesapi.DynamicResource{},
							},
						},
					},
				},
			},
		},
		{
			desc:             "pod with dynamic resources",
			pods:             pods,
			devices:          []*podresourcesapi.ContainerDevices{},
			cpus:             cpus,
			memory:           memory,
			dynamicResources: draDevs,
			expectedResponse: &podresourcesapi.ListPodResourcesResponse{
				PodResources: []*podresourcesapi.PodResources{
					{
						Name:      podName,
						Namespace: podNamespace,
						Containers: []*podresourcesapi.ContainerResources{
							{
								Name:             containerName,
								Devices:          []*podresourcesapi.ContainerDevices{},
								CpuIds:           cpus,
								Memory:           memory,
								DynamicResources: draDevs,
							},
						},
					},
				},
			},
		},
		{
			desc:             "pod with dynamic resources and devices",
			pods:             pods,
			devices:          devs,
			cpus:             cpus,
			memory:           memory,
			dynamicResources: draDevs,
			expectedResponse: &podresourcesapi.ListPodResourcesResponse{
				PodResources: []*podresourcesapi.PodResources{
					{
						Name:      podName,
						Namespace: podNamespace,
						Containers: []*podresourcesapi.ContainerResources{
							{
								Name:             containerName,
								Devices:          devs,
								CpuIds:           cpus,
								Memory:           memory,
								DynamicResources: draDevs,
							},
						},
					},
				},
			},
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			mockDevicesProvider := podresourcetest.NewMockDevicesProvider(mockCtrl)
			mockPodsProvider := podresourcetest.NewMockPodsProvider(mockCtrl)
			mockCPUsProvider := podresourcetest.NewMockCPUsProvider(mockCtrl)
			mockMemoryProvider := podresourcetest.NewMockMemoryProvider(mockCtrl)
			mockDynamicResourcesProvider := podresourcetest.NewMockDynamicResourcesProvider(mockCtrl)

			mockPodsProvider.EXPECT().GetPods().Return(tc.pods).AnyTimes()
			mockDevicesProvider.EXPECT().GetDevices(string(podUID), containerName).Return(tc.devices).AnyTimes()
			mockCPUsProvider.EXPECT().GetCPUs(string(podUID), containerName).Return(tc.cpus).AnyTimes()
			mockMemoryProvider.EXPECT().GetMemory(string(podUID), containerName).Return(tc.memory).AnyTimes()
			mockDynamicResourcesProvider.EXPECT().GetDynamicResources(pods[0], &containers[0]).Return(tc.dynamicResources).AnyTimes()
			mockDevicesProvider.EXPECT().UpdateAllocatedDevices().Return().AnyTimes()
			mockCPUsProvider.EXPECT().GetAllocatableCPUs().Return([]int64{}).AnyTimes()
			mockDevicesProvider.EXPECT().GetAllocatableDevices().Return([]*podresourcesapi.ContainerDevices{}).AnyTimes()
			mockMemoryProvider.EXPECT().GetAllocatableMemory().Return([]*podresourcesapi.ContainerMemory{}).AnyTimes()

			providers := PodResourcesProviders{
				Pods:             mockPodsProvider,
				Devices:          mockDevicesProvider,
				Cpus:             mockCPUsProvider,
				Memory:           mockMemoryProvider,
				DynamicResources: mockDynamicResourcesProvider,
			}
			server := NewV1PodResourcesServer(providers)
			resp, err := server.List(context.TODO(), &podresourcesapi.ListPodResourcesRequest{})
			if err != nil {
				t.Errorf("want err = %v, got %q", nil, err)
			}
			if !equalListResponse(tc.expectedResponse, resp) {
				t.Errorf("want resp = %s, got %s", tc.expectedResponse.String(), resp.String())
			}
		})
	}
}

func TestListPodResourcesWithInitContainersV1(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.KubeletPodResourcesDynamicResources, true)()

	podName := "pod-name"
	podNamespace := "pod-namespace"
	podUID := types.UID("pod-uid")
	initContainerName := "init-container-name"
	containerName := "container-name"
	numaID := int64(1)
	containerRestartPolicyAlways := v1.ContainerRestartPolicyAlways

	devs := []*podresourcesapi.ContainerDevices{
		{
			ResourceName: "resource",
			DeviceIds:    []string{"dev0", "dev1"},
			Topology:     &podresourcesapi.TopologyInfo{Nodes: []*podresourcesapi.NUMANode{{ID: numaID}}},
		},
	}

	cpus := []int64{12, 23, 30}

	memory := []*podresourcesapi.ContainerMemory{
		{
			MemoryType: "memory",
			Size_:      1073741824,
			Topology:   &podresourcesapi.TopologyInfo{Nodes: []*podresourcesapi.NUMANode{{ID: numaID}}},
		},
		{
			MemoryType: "hugepages-1Gi",
			Size_:      1073741824,
			Topology:   &podresourcesapi.TopologyInfo{Nodes: []*podresourcesapi.NUMANode{{ID: numaID}}},
		},
	}

	containers := []v1.Container{
		{
			Name: containerName,
		},
	}

	for _, tc := range []struct {
		desc     string
		pods     []*v1.Pod
		mockFunc func(
			[]*v1.Pod,
			*podresourcetest.MockDevicesProvider,
			*podresourcetest.MockCPUsProvider,
			*podresourcetest.MockMemoryProvider,
			*podresourcetest.MockDynamicResourcesProvider)
		sidecarContainersEnabled bool
		expectedResponse         *podresourcesapi.ListPodResourcesResponse
	}{
		{
			desc: "pod having an init container",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      podName,
						Namespace: podNamespace,
						UID:       podUID,
					},
					Spec: v1.PodSpec{
						InitContainers: []v1.Container{
							{
								Name: initContainerName,
							},
						},
						Containers: containers,
					},
				},
			},
			mockFunc: func(
				pods []*v1.Pod,
				devicesProvider *podresourcetest.MockDevicesProvider,
				cpusProvider *podresourcetest.MockCPUsProvider,
				memoryProvider *podresourcetest.MockMemoryProvider,
				dynamicResourcesProvider *podresourcetest.MockDynamicResourcesProvider) {
				devicesProvider.EXPECT().UpdateAllocatedDevices().Return().AnyTimes()
				devicesProvider.EXPECT().GetDevices(string(podUID), containerName).Return(devs).AnyTimes()
				cpusProvider.EXPECT().GetCPUs(string(podUID), containerName).Return(cpus).AnyTimes()
				memoryProvider.EXPECT().GetMemory(string(podUID), containerName).Return(memory).AnyTimes()
				dynamicResourcesProvider.EXPECT().GetDynamicResources(pods[0], &pods[0].Spec.Containers[0]).Return([]*podresourcesapi.DynamicResource{}).AnyTimes()

			},
			expectedResponse: &podresourcesapi.ListPodResourcesResponse{
				PodResources: []*podresourcesapi.PodResources{
					{
						Name:      podName,
						Namespace: podNamespace,
						Containers: []*podresourcesapi.ContainerResources{
							{
								Name:             containerName,
								Devices:          devs,
								CpuIds:           cpus,
								Memory:           memory,
								DynamicResources: []*podresourcesapi.DynamicResource{},
							},
						},
					},
				},
			},
		},
		{
			desc: "pod having an init container with SidecarContainers enabled",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      podName,
						Namespace: podNamespace,
						UID:       podUID,
					},
					Spec: v1.PodSpec{
						InitContainers: []v1.Container{
							{
								Name: initContainerName,
							},
						},
						Containers: containers,
					},
				},
			},
			mockFunc: func(
				pods []*v1.Pod,
				devicesProvider *podresourcetest.MockDevicesProvider,
				cpusProvider *podresourcetest.MockCPUsProvider,
				memoryProvider *podresourcetest.MockMemoryProvider,
				dynamicResourcesProvider *podresourcetest.MockDynamicResourcesProvider) {
				devicesProvider.EXPECT().UpdateAllocatedDevices().Return().AnyTimes()
				devicesProvider.EXPECT().GetDevices(string(podUID), containerName).Return(devs).AnyTimes()
				cpusProvider.EXPECT().GetCPUs(string(podUID), containerName).Return(cpus).AnyTimes()
				memoryProvider.EXPECT().GetMemory(string(podUID), containerName).Return(memory).AnyTimes()
				dynamicResourcesProvider.EXPECT().GetDynamicResources(pods[0], &pods[0].Spec.Containers[0]).Return([]*podresourcesapi.DynamicResource{}).AnyTimes()

			},
			sidecarContainersEnabled: true,
			expectedResponse: &podresourcesapi.ListPodResourcesResponse{
				PodResources: []*podresourcesapi.PodResources{
					{
						Name:      podName,
						Namespace: podNamespace,
						Containers: []*podresourcesapi.ContainerResources{
							{
								Name:             containerName,
								Devices:          devs,
								CpuIds:           cpus,
								Memory:           memory,
								DynamicResources: []*podresourcesapi.DynamicResource{},
							},
						},
					},
				},
			},
		},
		{
			desc: "pod having a restartable init container with SidecarContainers disabled",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      podName,
						Namespace: podNamespace,
						UID:       podUID,
					},
					Spec: v1.PodSpec{
						InitContainers: []v1.Container{
							{
								Name:          initContainerName,
								RestartPolicy: &containerRestartPolicyAlways,
							},
						},
						Containers: containers,
					},
				},
			},
			mockFunc: func(
				pods []*v1.Pod,
				devicesProvider *podresourcetest.MockDevicesProvider,
				cpusProvider *podresourcetest.MockCPUsProvider,
				memoryProvider *podresourcetest.MockMemoryProvider,
				dynamicResourcesProvider *podresourcetest.MockDynamicResourcesProvider) {
				devicesProvider.EXPECT().UpdateAllocatedDevices().Return().AnyTimes()

				devicesProvider.EXPECT().GetDevices(string(podUID), containerName).Return(devs).AnyTimes()
				cpusProvider.EXPECT().GetCPUs(string(podUID), containerName).Return(cpus).AnyTimes()
				memoryProvider.EXPECT().GetMemory(string(podUID), containerName).Return(memory).AnyTimes()
				dynamicResourcesProvider.EXPECT().GetDynamicResources(pods[0], &pods[0].Spec.Containers[0]).Return([]*podresourcesapi.DynamicResource{}).AnyTimes()

			},
			expectedResponse: &podresourcesapi.ListPodResourcesResponse{
				PodResources: []*podresourcesapi.PodResources{
					{
						Name:      podName,
						Namespace: podNamespace,
						Containers: []*podresourcesapi.ContainerResources{
							{
								Name:             containerName,
								Devices:          devs,
								CpuIds:           cpus,
								Memory:           memory,
								DynamicResources: []*podresourcesapi.DynamicResource{},
							},
						},
					},
				},
			},
		},
		{
			desc: "pod having an init container with SidecarContainers enabled",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      podName,
						Namespace: podNamespace,
						UID:       podUID,
					},
					Spec: v1.PodSpec{
						InitContainers: []v1.Container{
							{
								Name:          initContainerName,
								RestartPolicy: &containerRestartPolicyAlways,
							},
						},
						Containers: containers,
					},
				},
			},
			mockFunc: func(
				pods []*v1.Pod,
				devicesProvider *podresourcetest.MockDevicesProvider,
				cpusProvider *podresourcetest.MockCPUsProvider,
				memoryProvider *podresourcetest.MockMemoryProvider,
				dynamicResourcesProvider *podresourcetest.MockDynamicResourcesProvider) {
				devicesProvider.EXPECT().UpdateAllocatedDevices().Return().AnyTimes()

				devicesProvider.EXPECT().GetDevices(string(podUID), initContainerName).Return(devs).AnyTimes()
				cpusProvider.EXPECT().GetCPUs(string(podUID), initContainerName).Return(cpus).AnyTimes()
				memoryProvider.EXPECT().GetMemory(string(podUID), initContainerName).Return(memory).AnyTimes()
				dynamicResourcesProvider.EXPECT().GetDynamicResources(pods[0], &pods[0].Spec.InitContainers[0]).Return([]*podresourcesapi.DynamicResource{}).AnyTimes()

				devicesProvider.EXPECT().GetDevices(string(podUID), containerName).Return(devs).AnyTimes()
				cpusProvider.EXPECT().GetCPUs(string(podUID), containerName).Return(cpus).AnyTimes()
				memoryProvider.EXPECT().GetMemory(string(podUID), containerName).Return(memory).AnyTimes()
				dynamicResourcesProvider.EXPECT().GetDynamicResources(pods[0], &pods[0].Spec.Containers[0]).Return([]*podresourcesapi.DynamicResource{}).AnyTimes()

			},
			sidecarContainersEnabled: true,
			expectedResponse: &podresourcesapi.ListPodResourcesResponse{
				PodResources: []*podresourcesapi.PodResources{
					{
						Name:      podName,
						Namespace: podNamespace,
						Containers: []*podresourcesapi.ContainerResources{
							{
								Name:             initContainerName,
								Devices:          devs,
								CpuIds:           cpus,
								Memory:           memory,
								DynamicResources: []*podresourcesapi.DynamicResource{},
							},
							{
								Name:             containerName,
								Devices:          devs,
								CpuIds:           cpus,
								Memory:           memory,
								DynamicResources: []*podresourcesapi.DynamicResource{},
							},
						},
					},
				},
			},
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.SidecarContainers, tc.sidecarContainersEnabled)()

			mockCtrl := gomock.NewController(t)
			defer mockCtrl.Finish()

			mockDevicesProvider := podresourcetest.NewMockDevicesProvider(mockCtrl)
			mockPodsProvider := podresourcetest.NewMockPodsProvider(mockCtrl)
			mockCPUsProvider := podresourcetest.NewMockCPUsProvider(mockCtrl)
			mockMemoryProvider := podresourcetest.NewMockMemoryProvider(mockCtrl)
			mockDynamicResourcesProvider := podresourcetest.NewMockDynamicResourcesProvider(mockCtrl)

			mockPodsProvider.EXPECT().GetPods().Return(tc.pods).AnyTimes()
			tc.mockFunc(tc.pods, mockDevicesProvider, mockCPUsProvider, mockMemoryProvider, mockDynamicResourcesProvider)

			providers := PodResourcesProviders{
				Pods:             mockPodsProvider,
				Devices:          mockDevicesProvider,
				Cpus:             mockCPUsProvider,
				Memory:           mockMemoryProvider,
				DynamicResources: mockDynamicResourcesProvider,
			}
			server := NewV1PodResourcesServer(providers)
			resp, err := server.List(context.TODO(), &podresourcesapi.ListPodResourcesRequest{})
			if err != nil {
				t.Errorf("want err = %v, got %q", nil, err)
			}
			if !equalListResponse(tc.expectedResponse, resp) {
				t.Errorf("want resp = %s, got %s", tc.expectedResponse.String(), resp.String())
			}
		})
	}
}

func TestAllocatableResources(t *testing.T) {
	mockCtrl := gomock.NewController(t)
	defer mockCtrl.Finish()

	allDevs := []*podresourcesapi.ContainerDevices{
		{
			ResourceName: "resource",
			DeviceIds:    []string{"dev0"},
			Topology: &podresourcesapi.TopologyInfo{
				Nodes: []*podresourcesapi.NUMANode{
					{
						ID: 0,
					},
				},
			},
		},
		{
			ResourceName: "resource",
			DeviceIds:    []string{"dev1"},
			Topology: &podresourcesapi.TopologyInfo{
				Nodes: []*podresourcesapi.NUMANode{
					{
						ID: 1,
					},
				},
			},
		},
		{
			ResourceName: "resource-nt",
			DeviceIds:    []string{"devA"},
		},
		{
			ResourceName: "resource-mm",
			DeviceIds:    []string{"devM0"},
			Topology: &podresourcesapi.TopologyInfo{
				Nodes: []*podresourcesapi.NUMANode{
					{
						ID: 0,
					},
				},
			},
		},
		{
			ResourceName: "resource-mm",
			DeviceIds:    []string{"devMM"},
			Topology: &podresourcesapi.TopologyInfo{
				Nodes: []*podresourcesapi.NUMANode{
					{
						ID: 0,
					},
					{
						ID: 1,
					},
				},
			},
		},
	}

	allCPUs := []int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

	allMemory := []*podresourcesapi.ContainerMemory{
		{
			MemoryType: "memory",
			Size_:      5368709120,
			Topology: &podresourcesapi.TopologyInfo{
				Nodes: []*podresourcesapi.NUMANode{
					{
						ID: 0,
					},
				},
			},
		},
		{
			MemoryType: "hugepages-2Mi",
			Size_:      1073741824,
			Topology: &podresourcesapi.TopologyInfo{
				Nodes: []*podresourcesapi.NUMANode{
					{
						ID: 0,
					},
				},
			},
		},
		{
			MemoryType: "memory",
			Size_:      5368709120,
			Topology: &podresourcesapi.TopologyInfo{
				Nodes: []*podresourcesapi.NUMANode{
					{
						ID: 1,
					},
				},
			},
		},
		{
			MemoryType: "hugepages-2Mi",
			Size_:      1073741824,
			Topology: &podresourcesapi.TopologyInfo{
				Nodes: []*podresourcesapi.NUMANode{
					{
						ID: 1,
					},
				},
			},
		},
	}

	for _, tc := range []struct {
		desc                                 string
		allCPUs                              []int64
		allDevices                           []*podresourcesapi.ContainerDevices
		allMemory                            []*podresourcesapi.ContainerMemory
		expectedAllocatableResourcesResponse *podresourcesapi.AllocatableResourcesResponse
	}{
		{
			desc:                                 "no devices, no CPUs",
			allCPUs:                              []int64{},
			allDevices:                           []*podresourcesapi.ContainerDevices{},
			allMemory:                            []*podresourcesapi.ContainerMemory{},
			expectedAllocatableResourcesResponse: &podresourcesapi.AllocatableResourcesResponse{},
		},
		{
			desc:       "no devices, all CPUs",
			allCPUs:    allCPUs,
			allDevices: []*podresourcesapi.ContainerDevices{},
			expectedAllocatableResourcesResponse: &podresourcesapi.AllocatableResourcesResponse{
				CpuIds: allCPUs,
			},
		},
		{
			desc:       "no devices, no CPUs, all memory",
			allCPUs:    []int64{},
			allDevices: []*podresourcesapi.ContainerDevices{},
			expectedAllocatableResourcesResponse: &podresourcesapi.AllocatableResourcesResponse{
				Memory: allMemory,
			},
		},
		{
			desc:       "with devices, all CPUs",
			allCPUs:    allCPUs,
			allDevices: allDevs,
			expectedAllocatableResourcesResponse: &podresourcesapi.AllocatableResourcesResponse{
				CpuIds: allCPUs,
				Devices: []*podresourcesapi.ContainerDevices{
					{
						ResourceName: "resource",
						DeviceIds:    []string{"dev0"},
						Topology: &podresourcesapi.TopologyInfo{
							Nodes: []*podresourcesapi.NUMANode{
								{
									ID: 0,
								},
							},
						},
					},
					{
						ResourceName: "resource",
						DeviceIds:    []string{"dev1"},
						Topology: &podresourcesapi.TopologyInfo{
							Nodes: []*podresourcesapi.NUMANode{
								{
									ID: 1,
								},
							},
						},
					},
					{
						ResourceName: "resource-nt",
						DeviceIds:    []string{"devA"},
					},
					{
						ResourceName: "resource-mm",
						DeviceIds:    []string{"devM0"},
						Topology: &podresourcesapi.TopologyInfo{
							Nodes: []*podresourcesapi.NUMANode{
								{
									ID: 0,
								},
							},
						},
					},
					{
						ResourceName: "resource-mm",
						DeviceIds:    []string{"devMM"},
						Topology: &podresourcesapi.TopologyInfo{
							Nodes: []*podresourcesapi.NUMANode{
								{
									ID: 0,
								},
								{
									ID: 1,
								},
							},
						},
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
					{
						ResourceName: "resource",
						DeviceIds:    []string{"dev0"},
						Topology: &podresourcesapi.TopologyInfo{
							Nodes: []*podresourcesapi.NUMANode{
								{
									ID: 0,
								},
							},
						},
					},
					{
						ResourceName: "resource",
						DeviceIds:    []string{"dev1"},
						Topology: &podresourcesapi.TopologyInfo{
							Nodes: []*podresourcesapi.NUMANode{
								{
									ID: 1,
								},
							},
						},
					},
					{
						ResourceName: "resource-nt",
						DeviceIds:    []string{"devA"},
					},
					{
						ResourceName: "resource-mm",
						DeviceIds:    []string{"devM0"},
						Topology: &podresourcesapi.TopologyInfo{
							Nodes: []*podresourcesapi.NUMANode{
								{
									ID: 0,
								},
							},
						},
					},
					{
						ResourceName: "resource-mm",
						DeviceIds:    []string{"devMM"},
						Topology: &podresourcesapi.TopologyInfo{
							Nodes: []*podresourcesapi.NUMANode{
								{
									ID: 0,
								},
								{
									ID: 1,
								},
							},
						},
					},
				},
			},
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			mockDevicesProvider := podresourcetest.NewMockDevicesProvider(mockCtrl)
			mockPodsProvider := podresourcetest.NewMockPodsProvider(mockCtrl)
			mockCPUsProvider := podresourcetest.NewMockCPUsProvider(mockCtrl)
			mockMemoryProvider := podresourcetest.NewMockMemoryProvider(mockCtrl)

			mockDevicesProvider.EXPECT().GetDevices("", "").Return([]*podresourcesapi.ContainerDevices{}).AnyTimes()
			mockCPUsProvider.EXPECT().GetCPUs("", "").Return([]int64{}).AnyTimes()
			mockMemoryProvider.EXPECT().GetMemory("", "").Return([]*podresourcesapi.ContainerMemory{}).AnyTimes()
			mockDevicesProvider.EXPECT().UpdateAllocatedDevices().Return().AnyTimes()
			mockDevicesProvider.EXPECT().GetAllocatableDevices().Return(tc.allDevices).AnyTimes()
			mockCPUsProvider.EXPECT().GetAllocatableCPUs().Return(tc.allCPUs).AnyTimes()
			mockMemoryProvider.EXPECT().GetAllocatableMemory().Return(tc.allMemory).AnyTimes()

			providers := PodResourcesProviders{
				Pods:    mockPodsProvider,
				Devices: mockDevicesProvider,
				Cpus:    mockCPUsProvider,
				Memory:  mockMemoryProvider,
			}
			server := NewV1PodResourcesServer(providers)

			resp, err := server.GetAllocatableResources(context.TODO(), &podresourcesapi.AllocatableResourcesRequest{})
			if err != nil {
				t.Errorf("want err = %v, got %q", nil, err)
			}

			if !equalAllocatableResourcesResponse(tc.expectedAllocatableResourcesResponse, resp) {
				t.Errorf("want resp = %s, got %s", tc.expectedAllocatableResourcesResponse.String(), resp.String())
			}
		})
	}
}

func TestGetPodResourcesV1(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.KubeletPodResourcesGet, true)()
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.KubeletPodResourcesDynamicResources, true)()

	podName := "pod-name"
	podNamespace := "pod-namespace"
	podUID := types.UID("pod-uid")
	containerName := "container-name"
	numaID := int64(1)

	mockCtrl := gomock.NewController(t)
	defer mockCtrl.Finish()

	devs := []*podresourcesapi.ContainerDevices{
		{
			ResourceName: "resource",
			DeviceIds:    []string{"dev0", "dev1"},
			Topology:     &podresourcesapi.TopologyInfo{Nodes: []*podresourcesapi.NUMANode{{ID: numaID}}},
		},
	}

	cpus := []int64{12, 23, 30}

	memory := []*podresourcesapi.ContainerMemory{
		{
			MemoryType: "memory",
			Size_:      1073741824,
			Topology:   &podresourcesapi.TopologyInfo{Nodes: []*podresourcesapi.NUMANode{{ID: numaID}}},
		},
		{
			MemoryType: "hugepages-1Gi",
			Size_:      1073741824,
			Topology:   &podresourcesapi.TopologyInfo{Nodes: []*podresourcesapi.NUMANode{{ID: numaID}}},
		},
	}

	containers := []v1.Container{
		{
			Name: containerName,
		},
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: podNamespace,
			UID:       podUID,
		},
		Spec: v1.PodSpec{
			Containers: containers,
		},
	}

	pluginCDIDevices := []*podresourcesapi.CDIDevice{{Name: "dra-dev0"}, {Name: "dra-dev1"}}
	draDevs := []*podresourcesapi.DynamicResource{
		{
			ClassName:      "resource-class",
			ClaimName:      "claim-name",
			ClaimNamespace: "default",
			ClaimResources: []*podresourcesapi.ClaimResource{{CDIDevices: pluginCDIDevices}},
		},
	}

	for _, tc := range []struct {
		desc             string
		err              error
		exist            bool
		pod              *v1.Pod
		devices          []*podresourcesapi.ContainerDevices
		cpus             []int64
		memory           []*podresourcesapi.ContainerMemory
		dynamicResources []*podresourcesapi.DynamicResource
		expectedResponse *podresourcesapi.GetPodResourcesResponse
	}{
		{
			desc:             "pod not exist",
			err:              fmt.Errorf("pod %s in namespace %s not found", podName, podNamespace),
			exist:            false,
			pod:              nil,
			devices:          []*podresourcesapi.ContainerDevices{},
			cpus:             []int64{},
			memory:           []*podresourcesapi.ContainerMemory{},
			dynamicResources: []*podresourcesapi.DynamicResource{},

			expectedResponse: &podresourcesapi.GetPodResourcesResponse{},
		},
		{
			desc:             "pod without devices",
			err:              nil,
			exist:            true,
			pod:              pod,
			devices:          []*podresourcesapi.ContainerDevices{},
			cpus:             []int64{},
			memory:           []*podresourcesapi.ContainerMemory{},
			dynamicResources: []*podresourcesapi.DynamicResource{},
			expectedResponse: &podresourcesapi.GetPodResourcesResponse{
				PodResources: &podresourcesapi.PodResources{
					Name:      podName,
					Namespace: podNamespace,
					Containers: []*podresourcesapi.ContainerResources{
						{
							Name:             containerName,
							Devices:          []*podresourcesapi.ContainerDevices{},
							DynamicResources: []*podresourcesapi.DynamicResource{},
						},
					},
				},
			},
		},
		{
			desc:             "pod with devices",
			err:              nil,
			exist:            true,
			pod:              pod,
			devices:          devs,
			cpus:             cpus,
			memory:           memory,
			dynamicResources: draDevs,
			expectedResponse: &podresourcesapi.GetPodResourcesResponse{
				PodResources: &podresourcesapi.PodResources{
					Name:      podName,
					Namespace: podNamespace,
					Containers: []*podresourcesapi.ContainerResources{
						{
							Name:             containerName,
							Devices:          devs,
							CpuIds:           cpus,
							Memory:           memory,
							DynamicResources: draDevs,
						},
					},
				},
			},
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			mockDevicesProvider := podresourcetest.NewMockDevicesProvider(mockCtrl)
			mockPodsProvider := podresourcetest.NewMockPodsProvider(mockCtrl)
			mockCPUsProvider := podresourcetest.NewMockCPUsProvider(mockCtrl)
			mockMemoryProvider := podresourcetest.NewMockMemoryProvider(mockCtrl)
			mockDynamicResourcesProvider := podresourcetest.NewMockDynamicResourcesProvider(mockCtrl)

			mockPodsProvider.EXPECT().GetPodByName(podNamespace, podName).Return(tc.pod, tc.exist).AnyTimes()
			mockDevicesProvider.EXPECT().GetDevices(string(podUID), containerName).Return(tc.devices).AnyTimes()
			mockCPUsProvider.EXPECT().GetCPUs(string(podUID), containerName).Return(tc.cpus).AnyTimes()
			mockMemoryProvider.EXPECT().GetMemory(string(podUID), containerName).Return(tc.memory).AnyTimes()
			mockDynamicResourcesProvider.EXPECT().GetDynamicResources(pod, &containers[0]).Return(tc.dynamicResources).AnyTimes()
			mockDevicesProvider.EXPECT().UpdateAllocatedDevices().Return().AnyTimes()
			mockCPUsProvider.EXPECT().GetAllocatableCPUs().Return([]int64{}).AnyTimes()
			mockDevicesProvider.EXPECT().GetAllocatableDevices().Return([]*podresourcesapi.ContainerDevices{}).AnyTimes()
			mockMemoryProvider.EXPECT().GetAllocatableMemory().Return([]*podresourcesapi.ContainerMemory{}).AnyTimes()

			providers := PodResourcesProviders{
				Pods:             mockPodsProvider,
				Devices:          mockDevicesProvider,
				Cpus:             mockCPUsProvider,
				Memory:           mockMemoryProvider,
				DynamicResources: mockDynamicResourcesProvider,
			}
			server := NewV1PodResourcesServer(providers)
			podReq := &podresourcesapi.GetPodResourcesRequest{PodName: podName, PodNamespace: podNamespace}
			resp, err := server.Get(context.TODO(), podReq)

			if err != nil {
				if err.Error() != tc.err.Error() {
					t.Errorf("want exit = %v, got %v", tc.err, err)
				}
			} else {
				if err != tc.err {
					t.Errorf("want exit = %v, got %v", tc.err, err)
				} else {
					if !equalGetResponse(tc.expectedResponse, resp) {
						t.Errorf("want resp = %s, got %s", tc.expectedResponse.String(), resp.String())
					}
				}
			}
		})
	}

}

func TestGetPodResourcesWithInitContainersV1(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.KubeletPodResourcesGet, true)()
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.KubeletPodResourcesDynamicResources, true)()

	podName := "pod-name"
	podNamespace := "pod-namespace"
	podUID := types.UID("pod-uid")
	initContainerName := "init-container-name"
	containerName := "container-name"
	numaID := int64(1)
	containerRestartPolicyAlways := v1.ContainerRestartPolicyAlways

	devs := []*podresourcesapi.ContainerDevices{
		{
			ResourceName: "resource",
			DeviceIds:    []string{"dev0", "dev1"},
			Topology:     &podresourcesapi.TopologyInfo{Nodes: []*podresourcesapi.NUMANode{{ID: numaID}}},
		},
	}

	cpus := []int64{12, 23, 30}

	memory := []*podresourcesapi.ContainerMemory{
		{
			MemoryType: "memory",
			Size_:      1073741824,
			Topology:   &podresourcesapi.TopologyInfo{Nodes: []*podresourcesapi.NUMANode{{ID: numaID}}},
		},
		{
			MemoryType: "hugepages-1Gi",
			Size_:      1073741824,
			Topology:   &podresourcesapi.TopologyInfo{Nodes: []*podresourcesapi.NUMANode{{ID: numaID}}},
		},
	}

	containers := []v1.Container{
		{
			Name: containerName,
		},
	}

	for _, tc := range []struct {
		desc     string
		pod      *v1.Pod
		mockFunc func(
			*v1.Pod,
			*podresourcetest.MockDevicesProvider,
			*podresourcetest.MockCPUsProvider,
			*podresourcetest.MockMemoryProvider,
			*podresourcetest.MockDynamicResourcesProvider)
		sidecarContainersEnabled bool
		expectedResponse         *podresourcesapi.GetPodResourcesResponse
	}{
		{
			desc: "pod having an init container",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					Namespace: podNamespace,
					UID:       podUID,
				},
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name: initContainerName,
						},
					},
					Containers: containers,
				},
			},
			mockFunc: func(
				pod *v1.Pod,
				devicesProvider *podresourcetest.MockDevicesProvider,
				cpusProvider *podresourcetest.MockCPUsProvider,
				memoryProvider *podresourcetest.MockMemoryProvider,
				dynamicResourcesProvider *podresourcetest.MockDynamicResourcesProvider) {
				devicesProvider.EXPECT().UpdateAllocatedDevices().Return().AnyTimes()
				devicesProvider.EXPECT().GetDevices(string(podUID), containerName).Return(devs).AnyTimes()
				cpusProvider.EXPECT().GetCPUs(string(podUID), containerName).Return(cpus).AnyTimes()
				memoryProvider.EXPECT().GetMemory(string(podUID), containerName).Return(memory).AnyTimes()
				dynamicResourcesProvider.EXPECT().GetDynamicResources(pod, &pod.Spec.Containers[0]).Return([]*podresourcesapi.DynamicResource{}).AnyTimes()

			},
			expectedResponse: &podresourcesapi.GetPodResourcesResponse{
				PodResources: &podresourcesapi.PodResources{
					Name:      podName,
					Namespace: podNamespace,
					Containers: []*podresourcesapi.ContainerResources{
						{
							Name:             containerName,
							Devices:          devs,
							CpuIds:           cpus,
							Memory:           memory,
							DynamicResources: []*podresourcesapi.DynamicResource{},
						},
					},
				},
			},
		},
		{
			desc: "pod having an init container with SidecarContainers enabled",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					Namespace: podNamespace,
					UID:       podUID,
				},
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name: initContainerName,
						},
					},
					Containers: containers,
				},
			},
			mockFunc: func(
				pod *v1.Pod,
				devicesProvider *podresourcetest.MockDevicesProvider,
				cpusProvider *podresourcetest.MockCPUsProvider,
				memoryProvider *podresourcetest.MockMemoryProvider,
				dynamicResourcesProvider *podresourcetest.MockDynamicResourcesProvider) {
				devicesProvider.EXPECT().UpdateAllocatedDevices().Return().AnyTimes()
				devicesProvider.EXPECT().GetDevices(string(podUID), containerName).Return(devs).AnyTimes()
				cpusProvider.EXPECT().GetCPUs(string(podUID), containerName).Return(cpus).AnyTimes()
				memoryProvider.EXPECT().GetMemory(string(podUID), containerName).Return(memory).AnyTimes()
				dynamicResourcesProvider.EXPECT().GetDynamicResources(pod, &pod.Spec.Containers[0]).Return([]*podresourcesapi.DynamicResource{}).AnyTimes()

			},
			sidecarContainersEnabled: true,
			expectedResponse: &podresourcesapi.GetPodResourcesResponse{
				PodResources: &podresourcesapi.PodResources{
					Name:      podName,
					Namespace: podNamespace,
					Containers: []*podresourcesapi.ContainerResources{
						{
							Name:             containerName,
							Devices:          devs,
							CpuIds:           cpus,
							Memory:           memory,
							DynamicResources: []*podresourcesapi.DynamicResource{},
						},
					},
				},
			},
		},
		{
			desc: "pod having a restartable init container with SidecarContainers disabled",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					Namespace: podNamespace,
					UID:       podUID,
				},
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name:          initContainerName,
							RestartPolicy: &containerRestartPolicyAlways,
						},
					},
					Containers: containers,
				},
			},
			mockFunc: func(
				pod *v1.Pod,
				devicesProvider *podresourcetest.MockDevicesProvider,
				cpusProvider *podresourcetest.MockCPUsProvider,
				memoryProvider *podresourcetest.MockMemoryProvider,
				dynamicResourcesProvider *podresourcetest.MockDynamicResourcesProvider) {
				devicesProvider.EXPECT().UpdateAllocatedDevices().Return().AnyTimes()

				devicesProvider.EXPECT().GetDevices(string(podUID), containerName).Return(devs).AnyTimes()
				cpusProvider.EXPECT().GetCPUs(string(podUID), containerName).Return(cpus).AnyTimes()
				memoryProvider.EXPECT().GetMemory(string(podUID), containerName).Return(memory).AnyTimes()
				dynamicResourcesProvider.EXPECT().GetDynamicResources(pod, &pod.Spec.Containers[0]).Return([]*podresourcesapi.DynamicResource{}).AnyTimes()

			},
			expectedResponse: &podresourcesapi.GetPodResourcesResponse{
				PodResources: &podresourcesapi.PodResources{
					Name:      podName,
					Namespace: podNamespace,
					Containers: []*podresourcesapi.ContainerResources{
						{
							Name:             containerName,
							Devices:          devs,
							CpuIds:           cpus,
							Memory:           memory,
							DynamicResources: []*podresourcesapi.DynamicResource{},
						},
					},
				},
			},
		},
		{
			desc: "pod having an init container with SidecarContainers enabled",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					Namespace: podNamespace,
					UID:       podUID,
				},
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name:          initContainerName,
							RestartPolicy: &containerRestartPolicyAlways,
						},
					},
					Containers: containers,
				},
			},
			mockFunc: func(
				pod *v1.Pod,
				devicesProvider *podresourcetest.MockDevicesProvider,
				cpusProvider *podresourcetest.MockCPUsProvider,
				memoryProvider *podresourcetest.MockMemoryProvider,
				dynamicResourcesProvider *podresourcetest.MockDynamicResourcesProvider) {
				devicesProvider.EXPECT().UpdateAllocatedDevices().Return().AnyTimes()

				devicesProvider.EXPECT().GetDevices(string(podUID), initContainerName).Return(devs).AnyTimes()
				cpusProvider.EXPECT().GetCPUs(string(podUID), initContainerName).Return(cpus).AnyTimes()
				memoryProvider.EXPECT().GetMemory(string(podUID), initContainerName).Return(memory).AnyTimes()
				dynamicResourcesProvider.EXPECT().GetDynamicResources(pod, &pod.Spec.InitContainers[0]).Return([]*podresourcesapi.DynamicResource{}).AnyTimes()

				devicesProvider.EXPECT().GetDevices(string(podUID), containerName).Return(devs).AnyTimes()
				cpusProvider.EXPECT().GetCPUs(string(podUID), containerName).Return(cpus).AnyTimes()
				memoryProvider.EXPECT().GetMemory(string(podUID), containerName).Return(memory).AnyTimes()
				dynamicResourcesProvider.EXPECT().GetDynamicResources(pod, &pod.Spec.Containers[0]).Return([]*podresourcesapi.DynamicResource{}).AnyTimes()

			},
			sidecarContainersEnabled: true,
			expectedResponse: &podresourcesapi.GetPodResourcesResponse{
				PodResources: &podresourcesapi.PodResources{
					Name:      podName,
					Namespace: podNamespace,
					Containers: []*podresourcesapi.ContainerResources{
						{
							Name:             initContainerName,
							Devices:          devs,
							CpuIds:           cpus,
							Memory:           memory,
							DynamicResources: []*podresourcesapi.DynamicResource{},
						},
						{
							Name:             containerName,
							Devices:          devs,
							CpuIds:           cpus,
							Memory:           memory,
							DynamicResources: []*podresourcesapi.DynamicResource{},
						},
					},
				},
			},
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.SidecarContainers, tc.sidecarContainersEnabled)()

			mockCtrl := gomock.NewController(t)
			defer mockCtrl.Finish()

			mockDevicesProvider := podresourcetest.NewMockDevicesProvider(mockCtrl)
			mockPodsProvider := podresourcetest.NewMockPodsProvider(mockCtrl)
			mockCPUsProvider := podresourcetest.NewMockCPUsProvider(mockCtrl)
			mockMemoryProvider := podresourcetest.NewMockMemoryProvider(mockCtrl)
			mockDynamicResourcesProvider := podresourcetest.NewMockDynamicResourcesProvider(mockCtrl)

			mockPodsProvider.EXPECT().GetPodByName(podNamespace, podName).Return(tc.pod, true).AnyTimes()
			tc.mockFunc(tc.pod, mockDevicesProvider, mockCPUsProvider, mockMemoryProvider, mockDynamicResourcesProvider)

			providers := PodResourcesProviders{
				Pods:             mockPodsProvider,
				Devices:          mockDevicesProvider,
				Cpus:             mockCPUsProvider,
				Memory:           mockMemoryProvider,
				DynamicResources: mockDynamicResourcesProvider,
			}
			server := NewV1PodResourcesServer(providers)
			podReq := &podresourcesapi.GetPodResourcesRequest{PodName: podName, PodNamespace: podNamespace}
			resp, err := server.Get(context.TODO(), podReq)
			if err != nil {
				t.Errorf("want err = %v, got %q", nil, err)
			}
			if !equalGetResponse(tc.expectedResponse, resp) {
				t.Errorf("want resp = %s, got %s", tc.expectedResponse.String(), resp.String())
			}
		})
	}
}

func equalListResponse(respA, respB *podresourcesapi.ListPodResourcesResponse) bool {
	if len(respA.PodResources) != len(respB.PodResources) {
		return false
	}
	for idx := 0; idx < len(respA.PodResources); idx++ {
		podResA := respA.PodResources[idx]
		podResB := respB.PodResources[idx]
		if podResA.Name != podResB.Name {
			return false
		}
		if podResA.Namespace != podResB.Namespace {
			return false
		}
		if len(podResA.Containers) != len(podResB.Containers) {
			return false
		}
		for jdx := 0; jdx < len(podResA.Containers); jdx++ {
			cntA := podResA.Containers[jdx]
			cntB := podResB.Containers[jdx]

			if cntA.Name != cntB.Name {
				return false
			}
			if !equalInt64s(cntA.CpuIds, cntB.CpuIds) {
				return false
			}

			if !equalContainerDevices(cntA.Devices, cntB.Devices) {
				return false
			}

			if !equalDynamicResources(cntA.DynamicResources, cntB.DynamicResources) {
				return false
			}
		}
	}
	return true
}

func equalDynamicResources(draResA, draResB []*podresourcesapi.DynamicResource) bool {
	if len(draResA) != len(draResB) {
		return false
	}

	for idx := 0; idx < len(draResA); idx++ {
		cntDraResA := draResA[idx]
		cntDraResB := draResB[idx]

		if cntDraResA.ClassName != cntDraResB.ClassName {
			return false
		}
		if cntDraResA.ClaimName != cntDraResB.ClaimName {
			return false
		}
		if cntDraResA.ClaimNamespace != cntDraResB.ClaimNamespace {
			return false
		}
		if len(cntDraResA.ClaimResources) != len(cntDraResB.ClaimResources) {
			return false
		}
		for i := 0; i < len(cntDraResA.ClaimResources); i++ {
			claimResA := cntDraResA.ClaimResources[i]
			claimResB := cntDraResB.ClaimResources[i]
			if len(claimResA.CDIDevices) != len(claimResB.CDIDevices) {
				return false
			}
			for y := 0; y < len(claimResA.CDIDevices); y++ {
				cdiDeviceA := claimResA.CDIDevices[y]
				cdiDeviceB := claimResB.CDIDevices[y]
				if cdiDeviceA.Name != cdiDeviceB.Name {
					return false
				}
			}
		}
	}

	return true
}

func equalContainerDevices(devA, devB []*podresourcesapi.ContainerDevices) bool {
	if len(devA) != len(devB) {
		return false
	}

	for idx := 0; idx < len(devA); idx++ {
		cntDevA := devA[idx]
		cntDevB := devB[idx]

		if cntDevA.ResourceName != cntDevB.ResourceName {
			return false
		}
		if !equalTopology(cntDevA.Topology, cntDevB.Topology) {
			return false
		}
		if !equalStrings(cntDevA.DeviceIds, cntDevB.DeviceIds) {
			return false
		}
	}

	return true
}

func equalInt64s(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	aCopy := append([]int64{}, a...)
	sort.Slice(aCopy, func(i, j int) bool { return aCopy[i] < aCopy[j] })
	bCopy := append([]int64{}, b...)
	sort.Slice(bCopy, func(i, j int) bool { return bCopy[i] < bCopy[j] })
	return reflect.DeepEqual(aCopy, bCopy)
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	aCopy := append([]string{}, a...)
	sort.Strings(aCopy)
	bCopy := append([]string{}, b...)
	sort.Strings(bCopy)
	return reflect.DeepEqual(aCopy, bCopy)
}

func equalTopology(a, b *podresourcesapi.TopologyInfo) bool {
	if a == nil && b != nil {
		return false
	}
	if a != nil && b == nil {
		return false
	}
	return reflect.DeepEqual(a, b)
}

func equalAllocatableResourcesResponse(respA, respB *podresourcesapi.AllocatableResourcesResponse) bool {
	if !equalInt64s(respA.CpuIds, respB.CpuIds) {
		return false
	}
	return equalContainerDevices(respA.Devices, respB.Devices)
}

func equalGetResponse(ResA, ResB *podresourcesapi.GetPodResourcesResponse) bool {
	podResA := ResA.PodResources
	podResB := ResB.PodResources
	if podResA.Name != podResB.Name {
		return false
	}
	if podResA.Namespace != podResB.Namespace {
		return false
	}
	if len(podResA.Containers) != len(podResB.Containers) {
		return false
	}
	for jdx := 0; jdx < len(podResA.Containers); jdx++ {
		cntA := podResA.Containers[jdx]
		cntB := podResB.Containers[jdx]

		if cntA.Name != cntB.Name {
			return false
		}
		if !equalInt64s(cntA.CpuIds, cntB.CpuIds) {
			return false
		}

		if !equalContainerDevices(cntA.Devices, cntB.Devices) {
			return false
		}

		if !equalDynamicResources(cntA.DynamicResources, cntB.DynamicResources) {
			return false
		}

	}
	return true
}
