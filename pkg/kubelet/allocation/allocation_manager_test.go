/*
Copyright 2025 The Kubernetes Authors.

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

package allocation

import (
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/kubelet/allocation/state"
)

func TestUpdatePodFromAllocation(t *testing.T) {
	containerRestartPolicyAlways := v1.ContainerRestartPolicyAlways
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345",
			Name:      "test",
			Namespace: "default",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(400, resource.DecimalSI),
						},
					},
				},
				{
					Name: "c2",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(600, resource.DecimalSI),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(700, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(800, resource.DecimalSI),
						},
					},
				},
			},
			InitContainers: []v1.Container{
				{
					Name: "c1-restartable-init",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(300, resource.DecimalSI),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(500, resource.DecimalSI),
						},
					},
					RestartPolicy: &containerRestartPolicyAlways,
				},
				{
					Name: "c1-init",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(600, resource.DecimalSI),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(700, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(800, resource.DecimalSI),
						},
					},
				},
			},
		},
	}

	resizedPod := pod.DeepCopy()
	resizedPod.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = *resource.NewMilliQuantity(200, resource.DecimalSI)
	resizedPod.Spec.InitContainers[0].Resources.Requests[v1.ResourceCPU] = *resource.NewMilliQuantity(300, resource.DecimalSI)

	tests := []struct {
		name         string
		pod          *v1.Pod
		allocs       state.PodResourceAllocation
		expectPod    *v1.Pod
		expectUpdate bool
	}{{
		name: "steady state",
		pod:  pod,
		allocs: state.PodResourceAllocation{
			pod.UID: map[string]v1.ResourceRequirements{
				"c1":                  *pod.Spec.Containers[0].Resources.DeepCopy(),
				"c2":                  *pod.Spec.Containers[1].Resources.DeepCopy(),
				"c1-restartable-init": *pod.Spec.InitContainers[0].Resources.DeepCopy(),
				"c1-init":             *pod.Spec.InitContainers[1].Resources.DeepCopy(),
			},
		},
		expectUpdate: false,
	}, {
		name:         "no allocations",
		pod:          pod,
		allocs:       state.PodResourceAllocation{},
		expectUpdate: false,
	}, {
		name: "missing container allocation",
		pod:  pod,
		allocs: state.PodResourceAllocation{
			pod.UID: map[string]v1.ResourceRequirements{
				"c2": *pod.Spec.Containers[1].Resources.DeepCopy(),
			},
		},
		expectUpdate: false,
	}, {
		name: "resized container",
		pod:  pod,
		allocs: state.PodResourceAllocation{
			pod.UID: map[string]v1.ResourceRequirements{
				"c1":                  *resizedPod.Spec.Containers[0].Resources.DeepCopy(),
				"c2":                  *resizedPod.Spec.Containers[1].Resources.DeepCopy(),
				"c1-restartable-init": *resizedPod.Spec.InitContainers[0].Resources.DeepCopy(),
				"c1-init":             *resizedPod.Spec.InitContainers[1].Resources.DeepCopy(),
			},
		},
		expectUpdate: true,
		expectPod:    resizedPod,
	}}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pod := test.pod.DeepCopy()
			allocatedPod, updated := updatePodFromAllocation(pod, test.allocs)

			if test.expectUpdate {
				assert.True(t, updated, "updated")
				assert.Equal(t, test.expectPod, allocatedPod)
				assert.NotEqual(t, pod, allocatedPod)
			} else {
				assert.False(t, updated, "updated")
				assert.Same(t, pod, allocatedPod)
			}
		})
	}
}
