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
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/fake"
	"k8s.io/kubernetes/pkg/quota"
)

func TestServiceEvaluatorMatchesResources(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()
	evaluator := NewServiceEvaluator(kubeClient)
	expected := quota.ToSet([]api.ResourceName{
		api.ResourceServices,
		api.ResourceServicesNodePorts,
		api.ResourceServicesLoadBalancers,
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
		service *api.Service
		usage   api.ResourceList
	}{
		"loadbalancer": {
			service: &api.Service{
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeLoadBalancer,
				},
			},
			usage: api.ResourceList{
				api.ResourceServicesNodePorts:     resource.MustParse("0"),
				api.ResourceServicesLoadBalancers: resource.MustParse("1"),
				api.ResourceServices:              resource.MustParse("1"),
			},
		},
		"clusterip": {
			service: &api.Service{
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeClusterIP,
				},
			},
			usage: api.ResourceList{
				api.ResourceServices:              resource.MustParse("1"),
				api.ResourceServicesNodePorts:     resource.MustParse("0"),
				api.ResourceServicesLoadBalancers: resource.MustParse("0"),
			},
		},
		"nodeports": {
			service: &api.Service{
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Port: 27443,
						},
					},
				},
			},
			usage: api.ResourceList{
				api.ResourceServices:              resource.MustParse("1"),
				api.ResourceServicesNodePorts:     resource.MustParse("1"),
				api.ResourceServicesLoadBalancers: resource.MustParse("0"),
			},
		},
		"multi-nodeports": {
			service: &api.Service{
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Port: 27443,
						},
						{
							Port: 27444,
						},
					},
				},
			},
			usage: api.ResourceList{
				api.ResourceServices:              resource.MustParse("1"),
				api.ResourceServicesNodePorts:     resource.MustParse("2"),
				api.ResourceServicesLoadBalancers: resource.MustParse("0"),
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
		service  *api.Service
		required []api.ResourceName
		err      string
	}{
		"loadbalancer": {
			service: &api.Service{
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeLoadBalancer,
				},
			},
			required: []api.ResourceName{api.ResourceServicesLoadBalancers},
		},
		"clusterip": {
			service: &api.Service{
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeClusterIP,
				},
			},
			required: []api.ResourceName{api.ResourceServicesLoadBalancers, api.ResourceServices},
		},
		"nodeports": {
			service: &api.Service{
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Port: 27443,
						},
					},
				},
			},
			required: []api.ResourceName{api.ResourceServicesNodePorts},
		},
		"multi-nodeports": {
			service: &api.Service{
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Port: 27443,
						},
						{
							Port: 27444,
						},
					},
				},
			},
			required: []api.ResourceName{api.ResourceServicesNodePorts},
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
