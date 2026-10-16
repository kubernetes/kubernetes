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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
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

	t.Run("CalculateResource does not double-count Overhead with Status.Resources", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodLevelResourcesVerticalScaling, true)

		mem100Mi := resource.MustParse("100Mi")
		mem150Mi := resource.MustParse("150Mi")
		mem300Mi := resource.MustParse("300Mi")
		mem50Mi := resource.MustParse("50Mi")

		for _, tc := range []struct {
			name string
			pod  *v1.Pod
		}{
			{
				name: "container-level requests",
				pod: &v1.Pod{
					Spec: v1.PodSpec{
						NodeName: "node1",
						Overhead: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("50m"),
							v1.ResourceMemory: mem50Mi,
						},
						Containers: []v1.Container{{
							Name: "app",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("300m"),
									v1.ResourceMemory: mem300Mi,
								},
							},
						}},
					},
					Status: v1.PodStatus{
						// Kubelet populates pod-level status aggregates with Overhead included.
						AllocatedResources: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("150m"),
							v1.ResourceMemory: mem150Mi,
						},
						Resources: &v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("150m"),
								v1.ResourceMemory: mem150Mi,
							},
						},
						ContainerStatuses: []v1.ContainerStatus{{
							Name: "app",
							AllocatedResources: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("100m"),
								v1.ResourceMemory: mem100Mi,
							},
							Resources: &v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("100m"),
									v1.ResourceMemory: mem100Mi,
								},
							},
						}},
					},
				},
			},
			{
				name: "pod-level requests",
				pod: &v1.Pod{
					Spec: v1.PodSpec{
						NodeName: "node1",
						Overhead: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("50m"),
							v1.ResourceMemory: mem50Mi,
						},
						Resources: &v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("300m"),
								v1.ResourceMemory: mem300Mi,
							},
						},
						Containers: []v1.Container{{Name: "app"}},
					},
					Status: v1.PodStatus{
						AllocatedResources: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("150m"),
							v1.ResourceMemory: mem150Mi,
						},
						Resources: &v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("150m"),
								v1.ResourceMemory: mem150Mi,
							},
						},
					},
				},
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				podInfo, err := framework.NewPodInfo(tc.pod)
				if err != nil {
					t.Fatal(err)
				}
				gotRes := podInfoWithAllocatedResources(podInfo).CalculateResource().Resource
				if gotRes.GetMilliCPU() != 150 {
					t.Errorf("MilliCPU = %d, want 150 (100m allocated + 50m overhead)", gotRes.GetMilliCPU())
				}
				if gotRes.GetMemory() != mem150Mi.Value() {
					t.Errorf("Memory = %d, want %d (100Mi allocated + 50Mi overhead)", gotRes.GetMemory(), mem150Mi.Value())
				}
			})
		}
	})

	t.Run("running container with nil AllocatedResources clears newly added desired requests", func(t *testing.T) {
		mem200Mi := resource.MustParse("200Mi")
		podWithNilRunningAlloc := &v1.Pod{
			Spec: v1.PodSpec{
				NodeName: "node1",
				InitContainers: []v1.Container{{
					Name: "waiting-init",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("50m"),
						},
					},
				}},
				Containers: []v1.Container{
					{
						Name: "burstable",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("100m"),
								v1.ResourceMemory: mem200Mi,
							},
						},
					},
					{
						// Started with no requests; deferred resize added 500m CPU and 500Mi memory.
						Name: "formerly-besteffort",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("500m"),
								v1.ResourceMemory: resource.MustParse("500Mi"),
							},
						},
					},
				},
			},
			Status: v1.PodStatus{
				InitContainerStatuses: []v1.ContainerStatus{{
					Name:  "waiting-init",
					State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}},
				}},
				ContainerStatuses: []v1.ContainerStatus{
					{
						Name:  "burstable",
						State: v1.ContainerState{Running: &v1.ContainerStateRunning{}},
						AllocatedResources: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("100m"),
							v1.ResourceMemory: mem200Mi,
						},
					},
					{
						Name:  "formerly-besteffort",
						State: v1.ContainerState{Running: &v1.ContainerStateRunning{}},
					},
				},
			},
		}

		podInfo, err := framework.NewPodInfo(podWithNilRunningAlloc)
		if err != nil {
			t.Fatal(err)
		}
		gotPI := podInfoWithAllocatedResources(podInfo)
		if got := gotPI.GetPod().Spec.InitContainers[0].Resources.Requests[v1.ResourceCPU]; !got.Equal(resource.MustParse("50m")) {
			t.Errorf("non-running init container CPU = %s, want 50m", got.String())
		}
		gotRes := gotPI.CalculateResource().Resource
		if gotRes.GetMilliCPU() != 100 {
			t.Errorf("MilliCPU = %d, want 100", gotRes.GetMilliCPU())
		}
		if gotRes.GetMemory() != mem200Mi.Value() {
			t.Errorf("Memory = %d, want %d", gotRes.GetMemory(), mem200Mi.Value())
		}
	})
}
