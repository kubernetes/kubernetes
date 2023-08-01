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

	"github.com/golang/mock/gomock"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	podresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
	"k8s.io/kubelet/pkg/apis/podresources/v1alpha1"
	podresourcetest "k8s.io/kubernetes/pkg/kubelet/apis/podresources/testing"
)

func TestListPodResourcesV1alpha1(t *testing.T) {
	podName := "pod-name"
	podNamespace := "pod-namespace"
	podUID := types.UID("pod-uid")
	containerName := "container-name"

	devs := []*podresourcesv1.ContainerDevices{
		{
			ResourceName: "resource",
			DeviceIds:    []string{"dev0", "dev1"},
		},
	}

	mockCtrl := gomock.NewController(t)
	defer mockCtrl.Finish()

	for _, tc := range []struct {
		desc             string
		pods             []*v1.Pod
		devices          []*podresourcesv1.ContainerDevices
		expectedResponse *v1alpha1.ListPodResourcesResponse
	}{
		{
			desc:             "no pods",
			pods:             []*v1.Pod{},
			devices:          []*podresourcesv1.ContainerDevices{},
			expectedResponse: &v1alpha1.ListPodResourcesResponse{},
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
			devices: []*podresourcesv1.ContainerDevices{},
			expectedResponse: &v1alpha1.ListPodResourcesResponse{
				PodResources: []*v1alpha1.PodResources{
					{
						Name:      podName,
						Namespace: podNamespace,
						Containers: []*v1alpha1.ContainerResources{
							{
								Name:    containerName,
								Devices: []*v1alpha1.ContainerDevices{},
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
			expectedResponse: &v1alpha1.ListPodResourcesResponse{
				PodResources: []*v1alpha1.PodResources{
					{
						Name:      podName,
						Namespace: podNamespace,
						Containers: []*v1alpha1.ContainerResources{
							{
								Name:    containerName,
								Devices: v1DevicesToAlphaV1(devs),
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

			mockPodsProvider.EXPECT().GetPods().Return(tc.pods).AnyTimes()
			mockDevicesProvider.EXPECT().GetDevices(string(podUID), containerName).Return(tc.devices).AnyTimes()
			mockDevicesProvider.EXPECT().UpdateAllocatedDevices().Return().AnyTimes()

			providers := PodResourcesProviders{
				Pods:    mockPodsProvider,
				Devices: mockDevicesProvider,
			}
			server := NewV1alpha1PodResourcesServer(providers)
			resp, err := server.List(context.TODO(), &v1alpha1.ListPodResourcesRequest{})
			if err != nil {
				t.Errorf("want err = %v, got %q", nil, err)
			}
			if tc.expectedResponse.String() != resp.String() {
				t.Errorf("want resp = %s, got %s", tc.expectedResponse.String(), resp.String())
			}
		})
	}
}
