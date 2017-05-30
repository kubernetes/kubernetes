/*
Copyright 2017 The Kubernetes Authors.

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

package proxy

import (
	"net/url"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	v1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/cache"

	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
)

func TestResolve(t *testing.T) {
	endpoints := []*v1.Endpoints{{
		ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "alfa"},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{Hostname: "dummy-host", IP: "127.0.0.1"}},
			Ports:     []v1.EndpointPort{{Port: 443}},
		}},
	}}

	type expectation struct {
		url   string
		error bool
	}

	tests := []struct {
		name       string
		services   []*v1.Service
		endpoints  []*v1.Endpoints
		apiService *apiregistration.APIService

		clusterMode  expectation
		endpointMode expectation
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
			endpoints: endpoints, // TODO: do we have endpoints without ports?
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1."},
				Spec: apiregistration.APIServiceSpec{
					Service: &apiregistration.ServiceReference{
						Namespace: "one",
						Name:      "alfa",
					},
				},
			},

			clusterMode:  expectation{url: "https://hit"}, // TODO: this should be an error as well
			endpointMode: expectation{error: true},
		},
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
			endpoints: endpoints,
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1."},
				Spec: apiregistration.APIServiceSpec{
					Service: &apiregistration.ServiceReference{
						Namespace: "one",
						Name:      "alfa",
					},
				},
			},

			clusterMode:  expectation{url: "https://hit"},
			endpointMode: expectation{url: "https://127.0.0.1"},
		},
		{
			name: "cluster ip without endpoints",
			services: []*v1.Service{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "alfa"},
					Spec: v1.ServiceSpec{
						Type:      v1.ServiceTypeClusterIP,
						ClusterIP: "hit",
					},
				},
			},
			endpoints: nil,
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1."},
				Spec: apiregistration.APIServiceSpec{
					Service: &apiregistration.ServiceReference{
						Namespace: "one",
						Name:      "alfa",
					},
				},
			},

			clusterMode:  expectation{url: "https://hit"},
			endpointMode: expectation{error: true},
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
			endpoints: nil,
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1."},
				Spec: apiregistration.APIServiceSpec{
					Service: &apiregistration.ServiceReference{
						Namespace: "one",
						Name:      "alfa",
					},
				},
			},

			clusterMode:  expectation{url: "https://lb"},
			endpointMode: expectation{error: true},
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
			endpoints: nil,
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1."},
				Spec: apiregistration.APIServiceSpec{
					Service: &apiregistration.ServiceReference{
						Namespace: "one",
						Name:      "alfa",
					},
				},
			},

			clusterMode:  expectation{url: "https://np"},
			endpointMode: expectation{error: true},
		},
		{
			name:      "missing service",
			services:  nil,
			endpoints: nil,
			apiService: &apiregistration.APIService{
				ObjectMeta: metav1.ObjectMeta{Name: "v1."},
				Spec: apiregistration.APIServiceSpec{
					Service: &apiregistration.ServiceReference{
						Namespace: "one",
						Name:      "alfa",
					},
				},
			},

			clusterMode:  expectation{url: "https://alfa.one.svc"}, // defaulting to 443 due to https:// prefix
			endpointMode: expectation{error: true},
		},
	}

	for _, test := range tests {
		serviceCache := cache.NewIndexer(cache.DeletionHandlingMetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
		serviceLister := v1listers.NewServiceLister(serviceCache)
		for i := range test.services {
			serviceCache.Add(test.services[i])
		}

		endpointCache := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
		endpointLister := v1listers.NewEndpointsLister(endpointCache)
		for i := range test.endpoints {
			endpointCache.Add(test.endpoints[i])
		}

		check := func(mode string, expected expectation, url *url.URL, err error) {
			switch {
			case err == nil && expected.error:
				t.Errorf("%s in %s mode expected error, got none", test.name, mode)
			case err != nil && expected.error:
				// ignore
			case err != nil:
				t.Errorf("%s in %s mode unexpected error: %v", test.name, mode, err)
			case url.String() != expected.url:
				t.Errorf("%s in %s mode expected url %q, got %q", test.name, mode, expected.url, url.String())
			}
		}

		clusterURL, err := ResolveCluster(serviceLister, "one", "alfa")
		check("cluster", test.clusterMode, clusterURL, err)

		endpointURL, err := ResolveEndpoint(serviceLister, endpointLister, "one", "alfa")
		check("endpoint", test.endpointMode, endpointURL, err)
	}
}
