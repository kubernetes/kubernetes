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

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime/schema"
	quota "k8s.io/apiserver/pkg/quota/v1"
	"k8s.io/apiserver/pkg/quota/v1/generic"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestServiceEvaluatorMatchesResources(t *testing.T) {
	evaluator := NewServiceEvaluator(nil)
	// we give a lot of resources
	input := []corev1.ResourceName{
		corev1.ResourceConfigMaps,
		corev1.ResourceCPU,
		corev1.ResourceServices,
		corev1.ResourceServicesNodePorts,
		corev1.ResourceServicesLoadBalancers,
	}
	// but we only match these...
	expected := quota.ToSet([]corev1.ResourceName{
		corev1.ResourceServices,
		corev1.ResourceServicesNodePorts,
		corev1.ResourceServicesLoadBalancers,
	})
	actual := quota.ToSet(evaluator.MatchingResources(input))
	if !expected.Equal(actual) {
		t.Errorf("expected: %v, actual: %v", expected, actual)
	}
}

func TestServiceEvaluatorUsage(t *testing.T) {
	evaluator := NewServiceEvaluator(nil)
	testCases := map[string]struct {
		service *api.Service
		usage   corev1.ResourceList
	}{
		"loadbalancer": {
			service: &api.Service{
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeLoadBalancer,
				},
			},
			usage: corev1.ResourceList{
				corev1.ResourceServicesNodePorts:     resource.MustParse("0"),
				corev1.ResourceServicesLoadBalancers: resource.MustParse("1"),
				corev1.ResourceServices:              resource.MustParse("1"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "services"}): resource.MustParse("1"),
			},
		},
		"loadbalancer_ports": {
			service: &api.Service{
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeLoadBalancer,
					Ports: []api.ServicePort{
						{
							Port: 27443,
						},
					},
				},
			},
			usage: corev1.ResourceList{
				corev1.ResourceServicesNodePorts:     resource.MustParse("1"),
				corev1.ResourceServicesLoadBalancers: resource.MustParse("1"),
				corev1.ResourceServices:              resource.MustParse("1"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "services"}): resource.MustParse("1"),
			},
		},
		"clusterip": {
			service: &api.Service{
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeClusterIP,
				},
			},
			usage: corev1.ResourceList{
				corev1.ResourceServices:              resource.MustParse("1"),
				corev1.ResourceServicesNodePorts:     resource.MustParse("0"),
				corev1.ResourceServicesLoadBalancers: resource.MustParse("0"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "services"}): resource.MustParse("1"),
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
			usage: corev1.ResourceList{
				corev1.ResourceServices:              resource.MustParse("1"),
				corev1.ResourceServicesNodePorts:     resource.MustParse("1"),
				corev1.ResourceServicesLoadBalancers: resource.MustParse("0"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "services"}): resource.MustParse("1"),
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
			usage: corev1.ResourceList{
				corev1.ResourceServices:              resource.MustParse("1"),
				corev1.ResourceServicesNodePorts:     resource.MustParse("2"),
				corev1.ResourceServicesLoadBalancers: resource.MustParse("0"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "services"}): resource.MustParse("1"),
			},
		},
	}
	for testName, testCase := range testCases {
		actual, err := evaluator.Usage(testCase.service)
		if err != nil {
			t.Errorf("%s unexpected error: %v", testName, err)
		}
		if !quota.Equals(testCase.usage, actual) {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.usage, actual)
		}
	}
}

func TestServiceConstraintsFunc(t *testing.T) {
	testCases := map[string]struct {
		service  *api.Service
		required []corev1.ResourceName
		err      string
	}{
		"loadbalancer": {
			service: &api.Service{
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeLoadBalancer,
				},
			},
			required: []corev1.ResourceName{corev1.ResourceServicesLoadBalancers},
		},
		"clusterip": {
			service: &api.Service{
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeClusterIP,
				},
			},
			required: []corev1.ResourceName{corev1.ResourceServicesLoadBalancers, corev1.ResourceServices},
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
			required: []corev1.ResourceName{corev1.ResourceServicesNodePorts},
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
			required: []corev1.ResourceName{corev1.ResourceServicesNodePorts},
		},
	}

	evaluator := NewServiceEvaluator(nil)
	for testName, test := range testCases {
		err := evaluator.Constraints(test.required, test.service)
		switch {
		case err != nil && len(test.err) == 0,
			err == nil && len(test.err) != 0,
			err != nil && test.err != err.Error():
			t.Errorf("%s unexpected error: %v", testName, err)
		}
	}
}
