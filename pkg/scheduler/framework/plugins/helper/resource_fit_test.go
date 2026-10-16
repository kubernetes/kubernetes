/*
Copyright The Kubernetes Authors.

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

package helper

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

func TestPodInfoWithAllocatedResources(t *testing.T) {
	pod := &v1.Pod{
		Spec: v1.PodSpec{
			NodeName: "node1",
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("200m"),
				},
			},
			InitContainers: []v1.Container{
				{
					Name: "init",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("300m"),
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name: "app",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("400m"),
							v1.ResourceMemory: resource.MustParse("400Mi"),
						},
					},
				},
			},
		},
		Status: v1.PodStatus{
			// Status.AllocatedResources includes a memory entry not present in Spec.Resources.Requests.
			AllocatedResources: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("999Mi"),
			},
			InitContainerStatuses: []v1.ContainerStatus{
				{
					Name: "init",
					AllocatedResources: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("50m"),
					},
				},
			},
			ContainerStatuses: []v1.ContainerStatus{
				{
					Name: "app",
					AllocatedResources: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("75m"),
						v1.ResourceMemory: resource.MustParse("100Mi"),
					},
				},
			},
		},
	}

	pi, err := framework.NewPodInfo(pod)
	if err != nil {
		t.Fatal(err)
	}

	gotPI := podInfoWithAllocatedResources(pi)
	gotPod := gotPI.GetPod()

	if got := gotPod.Spec.Resources.Requests[v1.ResourceCPU]; !got.Equal(resource.MustParse("100m")) {
		t.Errorf("expected pod-level CPU request 100m, got %s", got.String())
	}
	if _, exists := gotPod.Spec.Resources.Requests[v1.ResourceMemory]; exists {
		t.Errorf("expected pod-level Requests not to gain keys absent from original Spec.Resources.Requests")
	}
	if got := gotPod.Spec.InitContainers[0].Resources.Requests[v1.ResourceCPU]; !got.Equal(resource.MustParse("50m")) {
		t.Errorf("expected init container CPU request 50m, got %s", got.String())
	}
	if got := gotPod.Spec.Containers[0].Resources.Requests[v1.ResourceCPU]; !got.Equal(resource.MustParse("75m")) {
		t.Errorf("expected app container CPU request 75m, got %s", got.String())
	}
	if orig := pi.GetPod().Spec.Resources.Requests[v1.ResourceCPU]; !orig.Equal(resource.MustParse("200m")) {
		t.Errorf("expected original PodInfo not to be mutated, got %s", orig.String())
	}
}
