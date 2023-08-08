/*
Copyright 2023 The Kubernetes Authors.

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

package noderesources

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"

	"k8s.io/kubernetes/pkg/scheduler/util"
)

func TestResourceAllocationScorerCalculateRequests(t *testing.T) {
	const oneMi = 1048576
	tests := []struct {
		name     string
		pod      v1.Pod
		expected map[v1.ResourceName]int64
	}{
		{
			name: "overhead only",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1"),
						v1.ResourceMemory: resource.MustParse("1Mi"),
					},
				},
			},
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    1000,
				v1.ResourceMemory: oneMi,
			},
		},
		{
			name: "1x requestless container",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{},
					},
				},
			},
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    util.DefaultMilliCPURequest,
				v1.ResourceMemory: util.DefaultMemoryRequest,
			},
		},
		{
			name: "2x requestless container",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{}, {},
					},
				},
			},
			// should accumulate once per container without a request
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    2 * util.DefaultMilliCPURequest,
				v1.ResourceMemory: 2 * util.DefaultMemoryRequest,
			},
		},
		{
			name: "request container + requestless container",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("1Mi"),
								},
							},
						},
						{},
					},
				},
			},
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    1000 + util.DefaultMilliCPURequest,
				v1.ResourceMemory: oneMi + util.DefaultMemoryRequest,
			},
		},
		{
			name: "container + requestless container + overhead",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1"),
						v1.ResourceMemory: resource.MustParse("1Mi"),
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("1Mi"),
								},
							},
						},
						{},
					},
				},
			},
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    2000 + util.DefaultMilliCPURequest,
				v1.ResourceMemory: 2*oneMi + util.DefaultMemoryRequest,
			},
		},
		{
			name: "init container + container + requestless container + overhead",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1"),
						v1.ResourceMemory: resource.MustParse("1Mi"),
					},
					InitContainers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: resource.MustParse("3"),
								},
							},
						},
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("1Mi"),
								},
							},
						},
						{},
					},
				},
			},
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    4000,
				v1.ResourceMemory: 2*oneMi + util.DefaultMemoryRequest,
			},
		},
		{
			name: "requestless init container + small init container + small container ",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{},
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1m"),
									v1.ResourceMemory: resource.MustParse("1"),
								},
							},
						},
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("3m"),
									v1.ResourceMemory: resource.MustParse("3"),
								},
							},
						},
					},
				},
			},
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    util.DefaultMilliCPURequest,
				v1.ResourceMemory: util.DefaultMemoryRequest,
			},
		},
		{
			name: "requestless init container + small init container + small container + requestless container ",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{},
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1m"),
									v1.ResourceMemory: resource.MustParse("1"),
								},
							},
						},
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("3m"),
									v1.ResourceMemory: resource.MustParse("3"),
								},
							},
						},
						{},
					},
				},
			},
			expected: map[v1.ResourceName]int64{
				v1.ResourceCPU:    3 + util.DefaultMilliCPURequest,
				v1.ResourceMemory: 3 + util.DefaultMemoryRequest,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var scorer resourceAllocationScorer
			for n, exp := range tc.expected {
				got := scorer.calculatePodResourceRequest(&tc.pod, n)
				if got != exp {
					t.Errorf("expected %s = %d, got %d", n, exp, got)
				}
			}
		})
	}
}
