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

package resource

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func BenchmarkPodRequests(b *testing.B) {
	restartAlways := v1.ContainerRestartPolicyAlways

	cases := []struct {
		name string
		pod  *v1.Pod
	}{
		{
			name: "SimplePod",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "container-1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("100m"),
									v1.ResourceMemory: resource.MustParse("100Mi"),
								},
							},
						},
					},
				},
			},
		},
		{
			name: "ManyContainers",
			pod: func() *v1.Pod {
				pod := &v1.Pod{}
				for i := 0; i < 20; i++ {
					pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
						Name: fmt.Sprintf("container-%d", i),
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("100m"),
								v1.ResourceMemory: resource.MustParse("100Mi"),
							},
						},
					})
					pod.Status.ContainerStatuses = append(pod.Status.ContainerStatuses, v1.ContainerStatus{
						Name: fmt.Sprintf("container-%d", i),
						Resources: &v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("120m"),
								v1.ResourceMemory: resource.MustParse("120Mi"),
							},
						},
					})
				}
				return pod
			}(),
		},
		{
			name: "InitContainers",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "container-1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("100m"),
									v1.ResourceMemory: resource.MustParse("100Mi"),
								},
							},
						},
					},
					InitContainers: []v1.Container{
						{
							Name: "init-1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("200m"),
									v1.ResourceMemory: resource.MustParse("200Mi"),
								},
							},
						},
						{
							Name: "init-2",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("300m"),
									v1.ResourceMemory: resource.MustParse("300Mi"),
								},
							},
						},
					},
				},
			},
		},
		{
			name: "Sidecars",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "container-1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("100m"),
									v1.ResourceMemory: resource.MustParse("100Mi"),
								},
							},
						},
					},
					InitContainers: []v1.Container{
						{
							Name:          "sidecar-1",
							RestartPolicy: &restartAlways,
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("50m"),
									v1.ResourceMemory: resource.MustParse("50Mi"),
								},
							},
						},
					},
				},
			},
		},
		{
			name: "PodLevelResources",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("1"),
							v1.ResourceMemory: resource.MustParse("1Gi"),
						},
					},
					Containers: []v1.Container{
						{
							Name: "container-1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("100m"),
									v1.ResourceMemory: resource.MustParse("100Mi"),
								},
							},
						},
					},
				},
			},
		},
		{
			name: "WithStatus",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "container-1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("100m"),
									v1.ResourceMemory: resource.MustParse("100Mi"),
								},
							},
						},
					},
				},
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							Name: "container-1",
							Resources: &v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("150m"),
									v1.ResourceMemory: resource.MustParse("150Mi"),
								},
							},
						},
					},
				},
			},
		},
		{
			name: "FullHouse",
			pod: func() *v1.Pod {
				pod := &v1.Pod{
					Spec: v1.PodSpec{
						Resources: &v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("2"),
								v1.ResourceMemory: resource.MustParse("2Gi"),
							},
						},
						Overhead: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("100m"),
							v1.ResourceMemory: resource.MustParse("100Mi"),
						},
					},
					Status: v1.PodStatus{
						Conditions: []v1.PodCondition{
							{
								Type:   v1.PodResizePending,
								Reason: v1.PodReasonInfeasible,
							},
						},
						Resources: &v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1.5"),
								v1.ResourceMemory: resource.MustParse("1.5Gi"),
							},
						},
					},
				}
				for i := 0; i < 5; i++ {
					pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
						Name: fmt.Sprintf("container-%d", i),
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("100m"),
								v1.ResourceMemory: resource.MustParse("100Mi"),
							},
						},
					})
					pod.Status.ContainerStatuses = append(pod.Status.ContainerStatuses, v1.ContainerStatus{
						Name: fmt.Sprintf("container-%d", i),
						Resources: &v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("120m"),
								v1.ResourceMemory: resource.MustParse("120Mi"),
							},
						},
					})
				}
				for i := 0; i < 3; i++ {
					pod.Spec.InitContainers = append(pod.Spec.InitContainers, v1.Container{
						Name:          fmt.Sprintf("sidecar-%d", i),
						RestartPolicy: &restartAlways,
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("50m"),
								v1.ResourceMemory: resource.MustParse("50Mi"),
							},
						},
					})
					pod.Status.InitContainerStatuses = append(pod.Status.InitContainerStatuses, v1.ContainerStatus{
						Name: fmt.Sprintf("sidecar-%d", i),
						Resources: &v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("60m"),
								v1.ResourceMemory: resource.MustParse("60Mi"),
							},
						},
					})
				}
				return pod
			}(),
		},
	}

	options := []struct {
		name string
		opts PodResourcesOptions
	}{
		{
			name: "Default",
			opts: PodResourcesOptions{},
		},
		{
			name: "WithStatus",
			opts: PodResourcesOptions{
				UseStatusResources: true,
				InPlacePodLevelResourcesVerticalScalingEnabled: true,
			},
		},
		{
			name: "WithReuse",
			opts: PodResourcesOptions{
				Reuse: v1.ResourceList{},
			},
		},
		{
			name: "WithNonMissing",
			opts: PodResourcesOptions{
				NonMissingContainerRequests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
			},
		},
		{
			name: "WithAll",
			opts: PodResourcesOptions{
				UseStatusResources:                              true,
				InPlacePodLevelResourcesVerticalScalingEnabled: true,
				NonMissingContainerRequests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
				Reuse: v1.ResourceList{},
			},
		},

	}

	for _, tc := range cases {
		for _, opt := range options {
			b.Run(fmt.Sprintf("func=PodRequests/pod=%s/opts=%s", tc.name, opt.name), func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					_ = PodRequests(tc.pod, opt.opts)
				}
			})
			b.Run(fmt.Sprintf("func=AggregateContainerRequests/pod=%s/opts=%s", tc.name, opt.name), func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					_ = AggregateContainerRequests(tc.pod, opt.opts)
				}
			})
		}
	}
}
