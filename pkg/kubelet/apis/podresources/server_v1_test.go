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
	podresourcesapi "k8s.io/kubelet/pkg/apis/podresources/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
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

	cpus := cpuset.NewCPUSet(12, 23, 30)

	for _, tc := range []struct {
		desc             string
		pods             []*v1.Pod
		devices          []*podresourcesapi.ContainerDevices
		cpus             cpuset.CPUSet
		expectedResponse *podresourcesapi.ListPodResourcesResponse
	}{
		{
			desc:             "no pods",
			pods:             []*v1.Pod{},
			devices:          []*podresourcesapi.ContainerDevices{},
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
			devices: []*podresourcesapi.ContainerDevices{},
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
								Devices: devs,
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

			if len(cntA.Devices) != len(cntB.Devices) {
				return false
			}

			for kdx := 0; kdx < len(cntA.Devices); kdx++ {
				cntDevA := cntA.Devices[kdx]
				cntDevB := cntB.Devices[kdx]

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
