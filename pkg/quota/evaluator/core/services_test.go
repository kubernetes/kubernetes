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

func TestServiceEvaluatorMatchesResources(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()
	evaluator := NewServiceEvaluator(kubeClient)
	expected := quota.ToSet([]v1.ResourceName{
		v1.ResourceServices,
		v1.ResourceServicesNodePorts,
		v1.ResourceServicesLoadBalancers,
	})
	actual := quota.ToSet(evaluator.MatchesResources())
	if !expected.Equal(actual) {
		t.Errorf("expected: %v, actual: %v", expected, actual)
	}
}

func TestServiceEvaluatorUsage(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()
	evaluator := NewServiceEvaluator(kubeClient)
	testCases := map[string]struct {
		service *v1.Service
		usage   v1.ResourceList
	}{
		"loadbalancer": {
			service: &v1.Service{
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeLoadBalancer,
				},
			},
			usage: v1.ResourceList{
				v1.ResourceServicesNodePorts:     resource.MustParse("0"),
				v1.ResourceServicesLoadBalancers: resource.MustParse("1"),
				v1.ResourceServices:              resource.MustParse("1"),
			},
		},
		"clusterip": {
			service: &v1.Service{
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeClusterIP,
				},
			},
			usage: v1.ResourceList{
				v1.ResourceServices:              resource.MustParse("1"),
				v1.ResourceServicesNodePorts:     resource.MustParse("0"),
				v1.ResourceServicesLoadBalancers: resource.MustParse("0"),
			},
		},
		"nodeports": {
			service: &v1.Service{
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeNodePort,
					Ports: []v1.ServicePort{
						{
							Port: 27443,
						},
					},
				},
			},
			usage: v1.ResourceList{
				v1.ResourceServices:              resource.MustParse("1"),
				v1.ResourceServicesNodePorts:     resource.MustParse("1"),
				v1.ResourceServicesLoadBalancers: resource.MustParse("0"),
			},
		},
		"multi-nodeports": {
			service: &v1.Service{
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeNodePort,
					Ports: []v1.ServicePort{
						{
							Port: 27443,
						},
						{
							Port: 27444,
						},
					},
				},
			},
			usage: v1.ResourceList{
				v1.ResourceServices:              resource.MustParse("1"),
				v1.ResourceServicesNodePorts:     resource.MustParse("2"),
				v1.ResourceServicesLoadBalancers: resource.MustParse("0"),
			},
		},
	}
	for testName, testCase := range testCases {
		actual := evaluator.Usage(testCase.service)
		if !quota.Equals(testCase.usage, actual) {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.usage, actual)
		}
	}
}

func TestServiceConstraintsFunc(t *testing.T) {
	testCases := map[string]struct {
		service  *v1.Service
		required []v1.ResourceName
		err      string
	}{
		"loadbalancer": {
			service: &v1.Service{
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeLoadBalancer,
				},
			},
			required: []v1.ResourceName{v1.ResourceServicesLoadBalancers},
		},
		"clusterip": {
			service: &v1.Service{
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeClusterIP,
				},
			},
			required: []v1.ResourceName{v1.ResourceServicesLoadBalancers, v1.ResourceServices},
		},
		"nodeports": {
			service: &v1.Service{
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeNodePort,
					Ports: []v1.ServicePort{
						{
							Port: 27443,
						},
					},
				},
			},
			required: []v1.ResourceName{v1.ResourceServicesNodePorts},
		},
		"multi-nodeports": {
			service: &v1.Service{
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeNodePort,
					Ports: []v1.ServicePort{
						{
							Port: 27443,
						},
						{
							Port: 27444,
						},
					},
				},
			},
			required: []v1.ResourceName{v1.ResourceServicesNodePorts},
		},
	}
	for testName, test := range testCases {
		err := ServiceConstraintsFunc(test.required, test.service)
		switch {
		case err != nil && len(test.err) == 0,
			err == nil && len(test.err) != 0,
			err != nil && test.err != err.Error():
			t.Errorf("%s unexpected error: %v", testName, err)
		}
	}
}
