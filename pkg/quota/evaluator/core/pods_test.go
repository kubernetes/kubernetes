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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/quota"
)

func TestPodConstraintsFunc(t *testing.T) {
	testCases := map[string]struct {
		pod      *api.Pod
		required []api.ResourceName
		err      string
	}{
		"init container resource invalid": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					InitContainers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("2m")},
							Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("1m")},
						},
					}},
				},
			},
			err: `spec.initContainers[0].resources.limits: Invalid value: "1m": must be greater than or equal to cpu request`,
		},
		"container resource invalid": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("2m")},
							Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("1m")},
						},
					}},
				},
			},
			err: `spec.containers[0].resources.limits: Invalid value: "1m": must be greater than or equal to cpu request`,
		},
		"init container resource missing": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					InitContainers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("1m")},
							Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("2m")},
						},
					}},
				},
			},
			required: []api.ResourceName{api.ResourceMemory},
			err:      `must specify memory`,
		},
		"container resource missing": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("1m")},
							Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("2m")},
						},
					}},
				},
			},
			required: []api.ResourceName{api.ResourceMemory},
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
	evaluator := NewPodEvaluator(kubeClient)
	testCases := map[string]struct {
		pod   *api.Pod
		usage api.ResourceList
	}{
		"init container CPU": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					InitContainers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("1m")},
							Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("2m")},
						},
					}},
				},
			},
			usage: api.ResourceList{
				api.ResourceRequestsCPU: resource.MustParse("1m"),
				api.ResourceLimitsCPU:   resource.MustParse("2m"),
				api.ResourcePods:        resource.MustParse("1"),
				api.ResourceCPU:         resource.MustParse("1m"),
			},
		},
		"init container MEM": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					InitContainers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceMemory: resource.MustParse("1m")},
							Limits:   api.ResourceList{api.ResourceMemory: resource.MustParse("2m")},
						},
					}},
				},
			},
			usage: api.ResourceList{
				api.ResourceRequestsMemory: resource.MustParse("1m"),
				api.ResourceLimitsMemory:   resource.MustParse("2m"),
				api.ResourcePods:           resource.MustParse("1"),
				api.ResourceMemory:         resource.MustParse("1m"),
			},
		},
		"container CPU": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("1m")},
							Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("2m")},
						},
					}},
				},
			},
			usage: api.ResourceList{
				api.ResourceRequestsCPU: resource.MustParse("1m"),
				api.ResourceLimitsCPU:   resource.MustParse("2m"),
				api.ResourcePods:        resource.MustParse("1"),
				api.ResourceCPU:         resource.MustParse("1m"),
			},
		},
		"container MEM": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceMemory: resource.MustParse("1m")},
							Limits:   api.ResourceList{api.ResourceMemory: resource.MustParse("2m")},
						},
					}},
				},
			},
			usage: api.ResourceList{
				api.ResourceRequestsMemory: resource.MustParse("1m"),
				api.ResourceLimitsMemory:   resource.MustParse("2m"),
				api.ResourcePods:           resource.MustParse("1"),
				api.ResourceMemory:         resource.MustParse("1m"),
			},
		},
		"init container maximums override sum of containers": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					InitContainers: []api.Container{
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU:    resource.MustParse("4"),
									api.ResourceMemory: resource.MustParse("100M"),
								},
								Limits: api.ResourceList{
									api.ResourceCPU:    resource.MustParse("8"),
									api.ResourceMemory: resource.MustParse("200M"),
								},
							},
						},
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU:    resource.MustParse("1"),
									api.ResourceMemory: resource.MustParse("50M"),
								},
								Limits: api.ResourceList{
									api.ResourceCPU:    resource.MustParse("2"),
									api.ResourceMemory: resource.MustParse("100M"),
								},
							},
						},
					},
					Containers: []api.Container{
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU:    resource.MustParse("1"),
									api.ResourceMemory: resource.MustParse("50M"),
								},
								Limits: api.ResourceList{
									api.ResourceCPU:    resource.MustParse("2"),
									api.ResourceMemory: resource.MustParse("100M"),
								},
							},
						},
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU:    resource.MustParse("2"),
									api.ResourceMemory: resource.MustParse("25M"),
								},
								Limits: api.ResourceList{
									api.ResourceCPU:    resource.MustParse("5"),
									api.ResourceMemory: resource.MustParse("50M"),
								},
							},
						},
					},
				},
			},
			usage: api.ResourceList{
				api.ResourceRequestsCPU:    resource.MustParse("4"),
				api.ResourceRequestsMemory: resource.MustParse("100M"),
				api.ResourceLimitsCPU:      resource.MustParse("8"),
				api.ResourceLimitsMemory:   resource.MustParse("200M"),
				api.ResourcePods:           resource.MustParse("1"),
				api.ResourceCPU:            resource.MustParse("4"),
				api.ResourceMemory:         resource.MustParse("100M"),
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
