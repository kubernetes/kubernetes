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

package apiserver

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	v1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/cache"

	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
)

func TestGetDestinationHost(t *testing.T) {
	tests := []struct {
		name       string
		services   []*v1.Service
		apiService *apiregistration.APIService

		expected string
	}{
		{
			name: "cluster ip without ports",
			services: []*v1.Service{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "alfa"},
					Spec: v1.ServiceSpec{
						Type:      v1.ServiceTypeClusterIP,
						ClusterIP: "hit",
					},
				},
			},
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1."},
				Spec: apiregistration.APIServiceSpec{
					Service: &apiregistration.ServiceReference{
						Namespace: "one",
						Name:      "alfa",
					},
				},
			},

			expected: "hit:443",
		},
		{
			name: "cluster ip",
			services: []*v1.Service{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "alfa"},
					Spec: v1.ServiceSpec{
						Type:      v1.ServiceTypeClusterIP,
						ClusterIP: "hit",
						Ports: []v1.ServicePort{
							{Port: 8443, TargetPort: intstr.FromInt(1443)},
							{Port: 1234, TargetPort: intstr.FromInt(1234)},
						},
					},
				},
			},
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1."},
				Spec: apiregistration.APIServiceSpec{
					Service: &apiregistration.ServiceReference{
						Namespace: "one",
						Name:      "alfa",
					},
				},
			},

			expected: "hit:8443",
		},
		{
			name: "loadbalancer",
			services: []*v1.Service{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "alfa"},
					Spec: v1.ServiceSpec{
						Type:      v1.ServiceTypeLoadBalancer,
						ClusterIP: "lb",
						Ports: []v1.ServicePort{
							{Port: 8443, TargetPort: intstr.FromInt(1443)},
							{Port: 1234, TargetPort: intstr.FromInt(1234)},
						},
					},
				},
			},
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1."},
				Spec: apiregistration.APIServiceSpec{
					Service: &apiregistration.ServiceReference{
						Namespace: "one",
						Name:      "alfa",
					},
				},
			},

			expected: "lb:8443",
		},
		{
			name: "node port",
			services: []*v1.Service{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "alfa"},
					Spec: v1.ServiceSpec{
						Type:      v1.ServiceTypeNodePort,
						ClusterIP: "np",
						Ports: []v1.ServicePort{
							{Port: 8443, TargetPort: intstr.FromInt(1443)},
							{Port: 1234, TargetPort: intstr.FromInt(1234)},
						},
					},
				},
			},
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1."},
				Spec: apiregistration.APIServiceSpec{
					Service: &apiregistration.ServiceReference{
						Namespace: "one",
						Name:      "alfa",
					},
				},
			},

			expected: "np:8443",
		},
		{
			name: "external name",
			services: []*v1.Service{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "alfa"},
					Spec: v1.ServiceSpec{
						Type:         v1.ServiceTypeExternalName,
						ExternalName: "foo.bar.com",
						Ports: []v1.ServicePort{
							{Port: 8443, TargetPort: intstr.FromInt(1443)},
							{Port: 1234, TargetPort: intstr.FromInt(1234)},
						},
					},
				},
			},
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1."},
				Spec: apiregistration.APIServiceSpec{
					Service: &apiregistration.ServiceReference{
						Namespace: "one",
						Name:      "alfa",
					},
				},
			},

			expected: "foo.bar.com:8443",
		},
		{
			name: "missing service",
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1."},
				Spec: apiregistration.APIServiceSpec{
					Service: &apiregistration.ServiceReference{
						Namespace: "one",
						Name:      "alfa",
					},
				},
			},

			expected: "alfa.one.svc", // defaulting to 443 due to https:// prefix
		},
	}

	for _, test := range tests {
		serviceCache := cache.NewIndexer(cache.DeletionHandlingMetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
		serviceLister := v1listers.NewServiceLister(serviceCache)
		c := &APIServiceRegistrationController{
			serviceLister: serviceLister,
		}
		for i := range test.services {
			serviceCache.Add(test.services[i])
		}

		actual := c.getDestinationHost(test.apiService)
		if actual != test.expected {
			t.Errorf("%s expected %v, got %v", test.name, test.expected, actual)
		}

	}
}
