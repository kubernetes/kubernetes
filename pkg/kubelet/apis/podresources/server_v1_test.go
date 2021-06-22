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
	"reflect"
	"sort"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	podresourcesapi "k8s.io/kubelet/pkg/apis/podresources/v1"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	"k8s.io/kubernetes/pkg/kubelet/cm/devicemanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
)

func TestListPodResourcesV1(t *testing.T) {
	podName := "pod-name"
	podNamespace := "pod-namespace"
	podUID := types.UID("pod-uid")
	containerName := "container-name"
	numaID := int64(1)

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

	for _, tc := range []struct {
		desc             string
		pods             []*v1.Pod
		devices          []*podresourcesapi.ContainerDevices
		cpus             []int64
		memory           []*podresourcesapi.ContainerMemory
		expectedResponse *podresourcesapi.ListPodResourcesResponse
	}{
		{
			desc:             "no pods",
			pods:             []*v1.Pod{},
			devices:          []*podresourcesapi.ContainerDevices{},
			cpus:             []int64{},
			memory:           []*podresourcesapi.ContainerMemory{},
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
			cpus:    []int64{},
			memory:  []*podresourcesapi.ContainerMemory{},
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
			cpus:    cpus,
			memory:  memory,
			expectedResponse: &podresourcesapi.ListPodResourcesResponse{
				PodResources: []*podresourcesapi.PodResources{
					{
						Name:      podName,
						Namespace: podNamespace,
						Containers: []*podresourcesapi.ContainerResources{
							{
								Name:    containerName,
								Devices: devs,
								CpuIds:  cpus,
								Memory:  memory,
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
			m.On("GetCPUs", string(podUID), containerName).Return(tc.cpus)
			m.On("GetMemory", string(podUID), containerName).Return(tc.memory)
			m.On("UpdateAllocatedDevices").Return()
			m.On("GetAllocatableCPUs").Return(cpuset.CPUSet{})
			m.On("GetAllocatableDevices").Return(devicemanager.NewResourceDeviceInstances())
			m.On("GetAllocatableMemory").Return([]state.Block{})
			server := NewV1PodResourcesServer(m, m, m, m)
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
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.KubeletPodResourcesGetAllocatable, true)()

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
			m := new(mockProvider)
			m.On("GetDevices", "", "").Return([]*podresourcesapi.ContainerDevices{})
			m.On("GetCPUs", "", "").Return([]int64{})
			m.On("GetMemory", "", "").Return([]*podresourcesapi.ContainerMemory{})
			m.On("UpdateAllocatedDevices").Return()
			m.On("GetAllocatableDevices").Return(tc.allDevices)
			m.On("GetAllocatableCPUs").Return(tc.allCPUs)
			m.On("GetAllocatableMemory").Return(tc.allMemory)
			server := NewV1PodResourcesServer(m, m, m, m)

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
