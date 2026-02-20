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

package resource

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestPodRequestsAndLimits(t *testing.T) {
	tests := []struct {
		name         string
		pod          *corev1.Pod
		wantRequests corev1.ResourceList
		wantLimits   corev1.ResourceList
	}{
		{
			name: "pod with container resources only",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "c1",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU: resource.MustParse("100m"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceCPU: resource.MustParse("200m"),
								},
							},
						},
					},
				},
			},
			wantRequests: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("100m"),
			},
			wantLimits: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("200m"),
			},
		},
		{
			name: "pod with pod-level resources",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "c1",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU: resource.MustParse("100m"),
								},
							},
						},
					},
					Resources: &corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU: resource.MustParse("200m"),
						},
						Limits: corev1.ResourceList{
							corev1.ResourceCPU: resource.MustParse("300m"),
						},
					},
				},
			},
			wantRequests: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("200m"),
			},
			wantLimits: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("300m"),
			},
		},
		{
			name: "pod with only pod-level resources",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "c1",
						},
					},
					Resources: &corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU: resource.MustParse("200m"),
						},
						Limits: corev1.ResourceList{
							corev1.ResourceCPU: resource.MustParse("300m"),
						},
					},
				},
			},
			wantRequests: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("200m"),
			},
			wantLimits: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("300m"),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reqs, limits := PodRequestsAndLimits(tt.pod)
			// DeepEqual might fail on nil vs empty map, so we check content
			if len(reqs) != len(tt.wantRequests) {
				t.Errorf("%s: Requests length = %d, want %d", tt.name, len(reqs), len(tt.wantRequests))
			}
			for k, v := range tt.wantRequests {
				if q, ok := reqs[k]; !ok || q.Cmp(v) != 0 {
					t.Errorf("%s: Request %s = %v, want %v", tt.name, k, q, v)
				}
			}
			if len(limits) != len(tt.wantLimits) {
				t.Errorf("%s: Limits length = %d, want %d", tt.name, len(limits), len(tt.wantLimits))
			}
			for k, v := range tt.wantLimits {
				if q, ok := limits[k]; !ok || q.Cmp(v) != 0 {
					t.Errorf("%s: Limit %s = %v, want %v", tt.name, k, q, v)
				}
			}
		})
	}
}
