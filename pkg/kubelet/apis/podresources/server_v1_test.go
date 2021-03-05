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
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	podresourcesapi "k8s.io/kubelet/pkg/apis/podresources/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	"k8s.io/kubernetes/pkg/kubelet/cm/devicemanager"
)

func TestListPodResourcesV1(t *testing.T) {
	podName := "pod-name"
	podNamespace := "pod-namespace"
	podUID := types.UID("pod-uid")
	containerName := "container-name"
	numaID := int64(1)

	devs := devicemanager.ResourceDeviceInstances{
		"resource": devicemanager.DeviceInstances{
			"dev0": pluginapi.Device{
				Topology: &pluginapi.TopologyInfo{
					Nodes: []*pluginapi.NUMANode{{ID: numaID}},
				},
			},
			"dev1": pluginapi.Device{
				Topology: &pluginapi.TopologyInfo{
					Nodes: []*pluginapi.NUMANode{{ID: numaID}},
				},
			},
		},
	}

	cpus := cpuset.NewCPUSet(12, 23, 30)

	for _, tc := range []struct {
		desc             string
		pods             []*v1.Pod
		devices          devicemanager.ResourceDeviceInstances
		cpus             cpuset.CPUSet
		expectedResponse *podresourcesapi.ListPodResourcesResponse
	}{
		{
			desc:             "no pods",
			pods:             []*v1.Pod{},
			devices:          devicemanager.NewResourceDeviceInstances(),
			cpus:             cpuset.CPUSet{},
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
			devices: devicemanager.NewResourceDeviceInstances(),
			cpus:    cpuset.CPUSet{},
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
			expectedResponse: &podresourcesapi.ListPodResourcesResponse{
				PodResources: []*podresourcesapi.PodResources{
					{
						Name:      podName,
						Namespace: podNamespace,
						Containers: []*podresourcesapi.ContainerResources{
							{
								Name:    containerName,
								Devices: containerDevicesFromResourceDeviceInstances(devs),
								CpuIds:  cpus.ToSliceNoSortInt64(),
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
			m.On("UpdateAllocatedDevices").Return()
			m.On("GetAllocatableCPUs").Return(cpuset.CPUSet{})
			m.On("GetAllocatableDevices").Return(devicemanager.NewResourceDeviceInstances())
			server := NewV1PodResourcesServer(m, m, m)
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
	allDevs := devicemanager.ResourceDeviceInstances{
		"resource": {
			"dev0": {
				ID:     "GPU-fef8089b-4820-abfc-e83e-94318197576e",
				Health: "Healthy",
				Topology: &pluginapi.TopologyInfo{
					Nodes: []*pluginapi.NUMANode{
						{
							ID: 0,
						},
					},
				},
			},
			"dev1": {
				ID:     "VF-8536e1e8-9dc6-4645-9aea-882db92e31e7",
				Health: "Healthy",
				Topology: &pluginapi.TopologyInfo{
					Nodes: []*pluginapi.NUMANode{
						{
							ID: 1,
						},
					},
				},
			},
		},
	}
	allCPUs := cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

	for _, tc := range []struct {
		desc                                 string
		allCPUs                              cpuset.CPUSet
		allDevices                           devicemanager.ResourceDeviceInstances
		expectedAllocatableResourcesResponse *podresourcesapi.AllocatableResourcesResponse
	}{
		{
			desc:                                 "no devices, no CPUs",
			allCPUs:                              cpuset.CPUSet{},
			allDevices:                           devicemanager.NewResourceDeviceInstances(),
			expectedAllocatableResourcesResponse: &podresourcesapi.AllocatableResourcesResponse{},
		},
		{
			desc:       "no devices, all CPUs",
			allCPUs:    allCPUs,
			allDevices: devicemanager.NewResourceDeviceInstances(),
			expectedAllocatableResourcesResponse: &podresourcesapi.AllocatableResourcesResponse{
				CpuIds: allCPUs.ToSliceNoSortInt64(),
			},
		},
		{
			desc:       "with devices, all CPUs",
			allCPUs:    allCPUs,
			allDevices: allDevs,
			expectedAllocatableResourcesResponse: &podresourcesapi.AllocatableResourcesResponse{
				CpuIds: allCPUs.ToSliceNoSortInt64(),
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
				},
			},
		},
		{
			desc:       "with devices, no CPUs",
			allCPUs:    cpuset.CPUSet{},
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
				},
			},
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			m := new(mockProvider)
			m.On("GetDevices", "", "").Return([]*podresourcesapi.ContainerDevices{})
			m.On("GetCPUs", "", "").Return(cpuset.CPUSet{})
			m.On("UpdateAllocatedDevices").Return()
			m.On("GetAllocatableDevices").Return(tc.allDevices)
			m.On("GetAllocatableCPUs").Return(tc.allCPUs)
			server := NewV1PodResourcesServer(m, m, m)

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

	// the ordering of container devices in the response is not defined,
	// so we need to do a full scan, failing at first mismatch
	for idx := 0; idx < len(devA); idx++ {
		if !containsContainerDevice(devA[idx], devB) {
			return false
		}
	}

	return true
}

func containsContainerDevice(cntDev *podresourcesapi.ContainerDevices, devs []*podresourcesapi.ContainerDevices) bool {
	for idx := 0; idx < len(devs); idx++ {
		if equalContainerDevice(cntDev, devs[idx]) {
			return true
		}
	}
	return false
}

func equalContainerDevice(cntDevA, cntDevB *podresourcesapi.ContainerDevices) bool {
	if cntDevA.ResourceName != cntDevB.ResourceName {
		return false
	}
	if !equalTopology(cntDevA.Topology, cntDevB.Topology) {
		return false
	}
	if !equalStrings(cntDevA.DeviceIds, cntDevB.DeviceIds) {
		return false
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
