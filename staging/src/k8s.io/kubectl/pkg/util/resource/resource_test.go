/*
Copyright 2021 The Kubernetes Authors.

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

package resource

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestPodRequestsAndLimits(t *testing.T) {
	var testCases = []struct {
		name             string
		pod              *v1.Pod
		expectedRequests v1.ResourceList
		expectedLimits   v1.ResourceList
	}{
		{
			name: "no requests, limits, or overhead",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "foobar"},
						{Name: "foobar2"},
					},
				},
			},
			expectedRequests: v1.ResourceList{},
			expectedLimits:   v1.ResourceList{},
		},
		{
			name: "limits only",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foobar",
							Resources: v1.ResourceRequirements{
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("2"),
									v1.ResourceMemory: resource.MustParse("10"),
								},
							},
						},
						{
							Name: "foobar2",
							Resources: v1.ResourceRequirements{
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("8"),
									v1.ResourceMemory: resource.MustParse("24"),
								},
							},
						},
					},
				},
			},
			expectedRequests: v1.ResourceList{},
			expectedLimits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("10"),
				v1.ResourceMemory: resource.MustParse("34"),
			},
		},
		{
			name: "requests only",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foobar",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("5"),
								},
							},
						},
						{
							Name: "foobar2",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("4"),
									v1.ResourceMemory: resource.MustParse("12"),
								},
							},
						},
					},
				},
			},
			expectedRequests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("5"),
				v1.ResourceMemory: resource.MustParse("17"),
			},
			expectedLimits: v1.ResourceList{},
		},
		{
			name: "overhead only",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("3"),
						v1.ResourceMemory: resource.MustParse("8"),
					},
					Containers: []v1.Container{
						{Name: "foobar"},
						{Name: "foobar2"},
					},
				},
			},
			expectedRequests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("3"),
				v1.ResourceMemory: resource.MustParse("8"),
			},
			expectedLimits: v1.ResourceList{},
		},
		{
			name: "two containers with no overhead should just be sum of containers",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foobar",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("5"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("2"),
									v1.ResourceMemory: resource.MustParse("10"),
								},
							},
						},
						{
							Name: "foobar2",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("4"),
									v1.ResourceMemory: resource.MustParse("12"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("8"),
									v1.ResourceMemory: resource.MustParse("24"),
								},
							},
						},
					},
				},
			},
			expectedRequests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("5"),
				v1.ResourceMemory: resource.MustParse("17"),
			},
			expectedLimits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("10"),
				v1.ResourceMemory: resource.MustParse("34"),
			},
		},
		{
			name: "two containers with one of them being unlimited should be unlimited",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foobar",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("5"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("2"),
									v1.ResourceMemory: resource.MustParse("10"),
								},
							},
						},
						{
							Name: "unlimited",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("4"),
									v1.ResourceMemory: resource.MustParse("12"),
								},
							},
						},
					},
				},
			},
			expectedRequests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("5"),
				v1.ResourceMemory: resource.MustParse("17"),
			},
			expectedLimits: v1.ResourceList{},
		},
		{
			name: "two containers with overhead should consider overhead",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("3"),
						v1.ResourceMemory: resource.MustParse("8"),
					},
					Containers: []v1.Container{
						{
							Name: "foobar",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("5"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("2"),
									v1.ResourceMemory: resource.MustParse("10"),
								},
							},
						},
						{
							Name: "foobar2",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("4"),
									v1.ResourceMemory: resource.MustParse("12"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("8"),
									v1.ResourceMemory: resource.MustParse("24"),
								},
							},
						},
					},
				},
			},
			expectedRequests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("8"),
				v1.ResourceMemory: resource.MustParse("25"),
			},
			expectedLimits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("13"),
				v1.ResourceMemory: resource.MustParse("42"),
			},
		},
		{
			name: "two containers with overhead and massive init should be the largest init plus overhead",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("3"),
						v1.ResourceMemory: resource.MustParse("8"),
					},
					Containers: []v1.Container{
						{
							Name: "foobar",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("5"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("2"),
									v1.ResourceMemory: resource.MustParse("10"),
								},
							},
						},
						{
							Name: "foobar2",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("4"),
									v1.ResourceMemory: resource.MustParse("12"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("8"),
									v1.ResourceMemory: resource.MustParse("24"),
								},
							},
						},
					},
					InitContainers: []v1.Container{
						{
							Name: "small-init",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("5"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("5"),
								},
							},
						},
						{
							Name: "big-init",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("40"),
									v1.ResourceMemory: resource.MustParse("120"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("80"),
									v1.ResourceMemory: resource.MustParse("240"),
								},
							},
						},
					},
				},
			},
			expectedRequests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("43"),
				v1.ResourceMemory: resource.MustParse("128"),
			},
			expectedLimits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("83"),
				v1.ResourceMemory: resource.MustParse("248"),
			},
		},
		{
			name: "two containers with unlimited init should be unlimited",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("3"),
						v1.ResourceMemory: resource.MustParse("8"),
					},
					Containers: []v1.Container{
						{
							Name: "foobar",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("5"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("2"),
									v1.ResourceMemory: resource.MustParse("10"),
								},
							},
						},
						{
							Name: "foobar2",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("4"),
									v1.ResourceMemory: resource.MustParse("12"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("8"),
									v1.ResourceMemory: resource.MustParse("24"),
								},
							},
						},
					},
					InitContainers: []v1.Container{
						{
							Name: "small-init",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("5"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("5"),
								},
							},
						},
						{
							Name: "unlimited-init",
						},
					},
				},
			},
			expectedRequests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("8"),
				v1.ResourceMemory: resource.MustParse("25"),
			},
			expectedLimits: v1.ResourceList{},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(tt *testing.T) {
			resRequests, resLimits := PodRequestsAndLimits(tc.pod)

			if !equality.Semantic.DeepEqual(tc.expectedRequests, resRequests) {
				tt.Errorf("requests:\n expected:\t%v\ngot\t\t%v", tc.expectedRequests, resRequests)
			}

			if !equality.Semantic.DeepEqual(tc.expectedLimits, resLimits) {
				tt.Errorf("limits:\n expected:\t%v\ngot\t\t%v", tc.expectedLimits, resLimits)
			}
		})
	}
}
