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

package qos

import (
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestIsContainerQOSGuaranteed(t *testing.T) {
	guaranteedResources := v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("1"),
			v1.ResourceMemory: resource.MustParse("100Mi"),
		},
		Limits: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("1"),
			v1.ResourceMemory: resource.MustParse("100Mi"),
		},
	}
	burstableCPUResources := v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("1"),
			v1.ResourceMemory: resource.MustParse("100Mi"),
		},
		Limits: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("2"),
			v1.ResourceMemory: resource.MustParse("100Mi"),
		},
	}
	burstableMemoryResources := v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("1"),
			v1.ResourceMemory: resource.MustParse("100Mi"),
		},
		Limits: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("1"),
			v1.ResourceMemory: resource.MustParse("200Mi"),
		},
	}
	bestEffortResources := v1.ResourceRequirements{}

	testCases := []struct {
		name     string
		pod      *v1.Pod
		expected bool
	}{
		{
			name: "Guaranteed pod, guaranteed container",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{UID: "pod1"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1", Resources: guaranteedResources}},
				},
				Status: v1.PodStatus{
					QOSClass: v1.PodQOSGuaranteed,
				},
			},
			expected: true,
		},
		{
			name: "Burstable pod, guaranteed container",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{UID: "pod1"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "c1", Resources: guaranteedResources},
					},
				},
				Status: v1.PodStatus{
					QOSClass: v1.PodQOSBurstable,
				},
			},
			expected: true,
		},
		{
			name: "Guaranteed pod, burstable CPU container",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{UID: "pod1"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1", Resources: burstableCPUResources}},
				},
				Status: v1.PodStatus{
					QOSClass: v1.PodQOSGuaranteed,
				},
			},
			expected: false,
		},
		{
			name: "Guaranteed pod, burstable Memory container",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{UID: "pod1"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1", Resources: burstableMemoryResources}},
				},
				Status: v1.PodStatus{
					QOSClass: v1.PodQOSGuaranteed,
				},
			},
			expected: false,
		},
		{
			name: "Guaranteed pod, besteffort container",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{UID: "pod1"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1", Resources: bestEffortResources}},
				},
				Status: v1.PodStatus{
					QOSClass: v1.PodQOSGuaranteed,
				},
			},
			expected: false,
		},
		{
			name: "BestEffort pod, besteffort container",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{UID: "pod1"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1", Resources: bestEffortResources}},
				},
				Status: v1.PodStatus{
					QOSClass: v1.PodQOSBestEffort,
				},
			},
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := IsContainerEquivalentQOSGuaranteed(&tc.pod.Spec.Containers[0])
			assert.Equal(t, tc.expected, result)
		})
	}
}
