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
			name: "cluster ip",
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

			expected: "hit",
		},
		{
			name: "loadbalancer",
			services: []*v1.Service{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "alfa"},
					Spec: v1.ServiceSpec{
						Type:      v1.ServiceTypeLoadBalancer,
						ClusterIP: "lb",
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

			expected: "lb",
		},
		{
			name: "node port",
			services: []*v1.Service{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "alfa"},
					Spec: v1.ServiceSpec{
						Type:      v1.ServiceTypeNodePort,
						ClusterIP: "np",
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

			expected: "np",
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

			expected: "alfa.one.svc",
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
