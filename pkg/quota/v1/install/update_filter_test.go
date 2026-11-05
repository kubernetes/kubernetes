/*
Copyright 2024 The Kubernetes Authors.

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

package install

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var (
	containerFoo = v1.Container{

		Name: "foo",
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("2"),
			},
		},
	}
	containerFooInitialStatus = v1.ContainerStatus{
		Name: "foo",
		Resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("2"),
			},
		},
	}
	containerFooChangedStatus = v1.ContainerStatus{
		Name: "foo",
		Resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("3"),
			},
		},
	}
)

func TestHasResourcesChanged(t *testing.T) {

	oldNotChangedPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				containerFoo,
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
			ContainerStatuses: []v1.ContainerStatus{
				containerFooInitialStatus,
			},
		},
	}
	newNotChangedPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				containerFoo,
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
			ContainerStatuses: []v1.ContainerStatus{
				containerFooInitialStatus,
			},
		},
	}

	oldChangedPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				containerFoo,
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
			ContainerStatuses: []v1.ContainerStatus{
				containerFooInitialStatus,
			},
		},
	}
	newChangedPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				containerFoo,
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
			ContainerStatuses: []v1.ContainerStatus{
				containerFooChangedStatus,
			},
		},
	}

	gpuResourceName := v1.ResourceName("nvidia.com/gpu")

	tests := []struct {
		name     string
		oldPod   *v1.Pod
		newPod   *v1.Pod
		expected bool
	}{
		{
			name:     "not-changed-pod",
			oldPod:   oldNotChangedPod,
			newPod:   newNotChangedPod,
			expected: false,
		},
		{
			name:     "changed-pod",
			oldPod:   oldChangedPod,
			newPod:   newChangedPod,
			expected: true,
		},
		{
			name:     "both-pods-have-no-statuses",
			oldPod:   &v1.Pod{},
			newPod:   &v1.Pod{},
			expected: false,
		},
		{
			// This is the critical test case for issue #125134.
			// When a pod is first created, oldPod has no ContainerStatuses.
			// After the Kubelet reports status, newPod has ContainerStatuses
			// with Resources populated. The old code missed this because it
			// iterated over oldPod.Status.ContainerStatuses (which was empty).
			name:   "old-pod-no-statuses-new-pod-has-resources-populated",
			oldPod: &v1.Pod{},
			newPod: &v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							Name: "foo",
							Resources: &v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: resource.MustParse("100m"),
								},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name:   "old-pod-no-statuses-new-pod-has-nil-resources",
			oldPod: &v1.Pod{},
			newPod: &v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							Name:      "foo",
							Resources: nil,
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "old-pod-has-resources-new-pod-nil-resources",
			oldPod: &v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							Name: "foo",
							Resources: &v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: resource.MustParse("100m"),
								},
							},
						},
					},
				},
			},
			newPod: &v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							Name:      "foo",
							Resources: nil,
						},
					},
				},
			},
			expected: true,
		},
		{
			name:   "init-container-old-pod-no-statuses-new-pod-has-resources",
			oldPod: &v1.Pod{},
			newPod: &v1.Pod{
				Status: v1.PodStatus{
					InitContainerStatuses: []v1.ContainerStatus{
						{
							Name: "init-foo",
							Resources: &v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: resource.MustParse("50m"),
								},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			// Extended resource (GPU) newly populated in status — the exact
			// scenario from issue #125134.
			name:   "extended-resource-gpu-newly-populated",
			oldPod: &v1.Pod{},
			newPod: &v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							Name: "gpu-container",
							Resources: &v1.ResourceRequirements{
								Requests: v1.ResourceList{
									gpuResourceName: resource.MustParse("1"),
								},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "extended-resource-gpu-unchanged",
			oldPod: &v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							Name: "gpu-container",
							Resources: &v1.ResourceRequirements{
								Requests: v1.ResourceList{
									gpuResourceName: resource.MustParse("1"),
								},
							},
						},
					},
				},
			},
			newPod: &v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							Name: "gpu-container",
							Resources: &v1.ResourceRequirements{
								Requests: v1.ResourceList{
									gpuResourceName: resource.MustParse("1"),
								},
							},
						},
					},
				},
			},
			expected: false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := hasResourcesChanged(test.oldPod, test.newPod); got != test.expected {
				t.Errorf("TestHasResourcesChanged = %v, expected %v", got, test.expected)
			}
		})
	}
}

