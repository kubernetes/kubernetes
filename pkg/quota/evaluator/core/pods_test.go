/*
Copyright 2016 The Kubernetes Authors.

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

package core

import (
	"testing"

	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/fake"
	"k8s.io/kubernetes/pkg/quota"
)

func TestPodConstraintsFunc(t *testing.T) {
	testCases := map[string]struct {
		pod      *v1.Pod
		required []v1.ResourceName
		err      string
	}{
		"init container resource invalid": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("2m")},
							Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("1m")},
						},
					}},
				},
			},
			err: `spec.initContainers[0].resources.limits: Invalid value: "1m": must be greater than or equal to cpu request`,
		},
		"container resource invalid": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("2m")},
							Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("1m")},
						},
					}},
				},
			},
			err: `spec.containers[0].resources.limits: Invalid value: "1m": must be greater than or equal to cpu request`,
		},
		"init container resource missing": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1m")},
							Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("2m")},
						},
					}},
				},
			},
			required: []v1.ResourceName{v1.ResourceMemory},
			err:      `must specify memory`,
		},
		"container resource missing": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1m")},
							Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("2m")},
						},
					}},
				},
			},
			required: []v1.ResourceName{v1.ResourceMemory},
			err:      `must specify memory`,
		},
	}
	for testName, test := range testCases {
		err := PodConstraintsFunc(test.required, test.pod)
		switch {
		case err != nil && len(test.err) == 0,
			err == nil && len(test.err) != 0,
			err != nil && test.err != err.Error():
			t.Errorf("%s unexpected error: %v", testName, err)
		}
	}
}

func TestPodEvaluatorUsage(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()
	evaluator := NewPodEvaluator(kubeClient, nil)
	testCases := map[string]struct {
		pod   *v1.Pod
		usage v1.ResourceList
	}{
		"init container CPU": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1m")},
							Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("2m")},
						},
					}},
				},
			},
			usage: v1.ResourceList{
				v1.ResourceRequestsCPU: resource.MustParse("1m"),
				v1.ResourceLimitsCPU:   resource.MustParse("2m"),
				v1.ResourcePods:        resource.MustParse("1"),
				v1.ResourceCPU:         resource.MustParse("1m"),
			},
		},
		"init container MEM": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("1m")},
							Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("2m")},
						},
					}},
				},
			},
			usage: v1.ResourceList{
				v1.ResourceRequestsMemory: resource.MustParse("1m"),
				v1.ResourceLimitsMemory:   resource.MustParse("2m"),
				v1.ResourcePods:           resource.MustParse("1"),
				v1.ResourceMemory:         resource.MustParse("1m"),
			},
		},
		"container CPU": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1m")},
							Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("2m")},
						},
					}},
				},
			},
			usage: v1.ResourceList{
				v1.ResourceRequestsCPU: resource.MustParse("1m"),
				v1.ResourceLimitsCPU:   resource.MustParse("2m"),
				v1.ResourcePods:        resource.MustParse("1"),
				v1.ResourceCPU:         resource.MustParse("1m"),
			},
		},
		"container MEM": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("1m")},
							Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("2m")},
						},
					}},
				},
			},
			usage: v1.ResourceList{
				v1.ResourceRequestsMemory: resource.MustParse("1m"),
				v1.ResourceLimitsMemory:   resource.MustParse("2m"),
				v1.ResourcePods:           resource.MustParse("1"),
				v1.ResourceMemory:         resource.MustParse("1m"),
			},
		},
		"init container maximums override sum of containers": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("4"),
									v1.ResourceMemory: resource.MustParse("100M"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("8"),
									v1.ResourceMemory: resource.MustParse("200M"),
								},
							},
						},
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("50M"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("2"),
									v1.ResourceMemory: resource.MustParse("100M"),
								},
							},
						},
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("50M"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("2"),
									v1.ResourceMemory: resource.MustParse("100M"),
								},
							},
						},
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("2"),
									v1.ResourceMemory: resource.MustParse("25M"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("5"),
									v1.ResourceMemory: resource.MustParse("50M"),
								},
							},
						},
					},
				},
			},
			usage: v1.ResourceList{
				v1.ResourceRequestsCPU:    resource.MustParse("4"),
				v1.ResourceRequestsMemory: resource.MustParse("100M"),
				v1.ResourceLimitsCPU:      resource.MustParse("8"),
				v1.ResourceLimitsMemory:   resource.MustParse("200M"),
				v1.ResourcePods:           resource.MustParse("1"),
				v1.ResourceCPU:            resource.MustParse("4"),
				v1.ResourceMemory:         resource.MustParse("100M"),
			},
		},
	}
	for testName, testCase := range testCases {
		actual := evaluator.Usage(testCase.pod)
		if !quota.Equals(testCase.usage, actual) {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.usage, actual)
		}
	}
}
