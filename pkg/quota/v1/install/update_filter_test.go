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
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
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
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := hasResourcesChanged(test.oldPod, test.newPod); got != test.expected {
				t.Errorf("TestHasResourcesChanged = %v, expected %v", got, test.expected)
			}
		})
	}
}

func TestDefaultUpdateFilterExtraNodeAllocatableResources(t *testing.T) {
	extra := []v1.ExtraNodeAllocatableResource{{
		Source:     v1.ExtraNodeAllocatableReference{APIGroup: "resource.k8s.io", Kind: "ResourceClaim", Name: "claim-1"},
		Containers: []string{"foo"},
		Mapping:    []v1.NodeAllocatableMappedResources{{Name: v1.ResourceCPU, Quantity: new(resource.MustParse("1"))}},
	}}
	tests := []struct {
		name                                string
		draNodeAllocatableResourcesDisabled bool
		oldExtraNodeAllocatable             []v1.ExtraNodeAllocatableResource
		newExtraNodeAllocatable             []v1.ExtraNodeAllocatableResource
		expected                            bool
	}{
		{
			name:                    "set",
			newExtraNodeAllocatable: extra,
			expected:                true,
		},
		{
			name:                    "cleared",
			oldExtraNodeAllocatable: extra,
			newExtraNodeAllocatable: []v1.ExtraNodeAllocatableResource{},
			expected:                true,
		},
		{
			name:                                "cleared with the feature disabled",
			draNodeAllocatableResourcesDisabled: true,
			oldExtraNodeAllocatable:             extra,
			newExtraNodeAllocatable:             []v1.ExtraNodeAllocatableResource{},
			expected:                            true,
		},
		{
			name:                    "unchanged",
			oldExtraNodeAllocatable: extra,
			newExtraNodeAllocatable: extra,
			expected:                false,
		},
	}
	filter := DefaultUpdateFilter()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRANodeAllocatableResources, !test.draNodeAllocatableResourcesDisabled)
			oldPod := &v1.Pod{
				Spec:   v1.PodSpec{Containers: []v1.Container{containerFoo}},
				Status: v1.PodStatus{Phase: v1.PodPending, ExtraNodeAllocatableResources: test.oldExtraNodeAllocatable},
			}
			newPod := &v1.Pod{
				Spec:   v1.PodSpec{Containers: []v1.Container{containerFoo}},
				Status: v1.PodStatus{Phase: v1.PodPending, ExtraNodeAllocatableResources: test.newExtraNodeAllocatable},
			}
			if got := filter(schema.GroupVersionResource{Version: "v1", Resource: "pods"}, oldPod, newPod); got != test.expected {
				t.Errorf("DefaultUpdateFilter() = %v, expected %v", got, test.expected)
			}
		})
	}
}
