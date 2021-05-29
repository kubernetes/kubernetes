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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	v1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
)

func TestResolve(t *testing.T) {
	matchingEndpoints := func(svc *v1.Service) []*v1.Endpoints {
		ports := []v1.EndpointPort{}
		for _, p := range svc.Spec.Ports {
			if p.TargetPort.Type != intstr.Int {
				continue
			}
			ports = append(ports, v1.EndpointPort{Name: p.Name, Port: p.TargetPort.IntVal})
		}

		return []*v1.Endpoints{{
			ObjectMeta: metav1.ObjectMeta{Namespace: svc.Namespace, Name: svc.Name},
			Subsets: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{Hostname: "dummy-host", IP: "127.0.0.1"}},
				Ports:     ports,
			}},
		}}
	}

	type expectation struct {
		url   string
		error bool
	}

	tests := []struct {
		name      string
		services  []*v1.Service
		endpoints func(svc *v1.Service) []*v1.Endpoints

		clusterMode  expectation
		endpointMode expectation
	}{
		{
			name: "cluster ip without 443 port",
			services: []*v1.Service{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "alfa"},
					Spec: v1.ServiceSpec{
						Type:      v1.ServiceTypeClusterIP,
						ClusterIP: "hit",
						Ports: []v1.ServicePort{
							{Port: 1234, TargetPort: intstr.FromInt(1234)},
						},
					},
				},
			},
			endpoints: matchingEndpoints,

			clusterMode:  expectation{error: true},
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
						Ports: []v1.ServicePort{
							{Name: "https", Port: 443, TargetPort: intstr.FromInt(1443)},
							{Port: 1234, TargetPort: intstr.FromInt(1234)},
						},
					},
				},
			},
			endpoints: matchingEndpoints,

			clusterMode:  expectation{url: "https://hit:443"},
			endpointMode: expectation{url: "https://127.0.0.1:1443"},
		},
		{
			name: "cluster ip without endpoints",
			services: []*v1.Service{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "alfa"},
					Spec: v1.ServiceSpec{
						Type:      v1.ServiceTypeClusterIP,
						ClusterIP: "hit",
						Ports: []v1.ServicePort{
							{Name: "https", Port: 443, TargetPort: intstr.FromInt(1443)},
							{Port: 1234, TargetPort: intstr.FromInt(1234)},
						},
					},
				},
			},
			endpoints: nil,

			clusterMode:  expectation{url: "https://hit:443"},
			endpointMode: expectation{error: true},
		},
		{
			name: "none cluster ip",
			services: []*v1.Service{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "alfa"},
					Spec: v1.ServiceSpec{
						Type:      v1.ServiceTypeClusterIP,
						ClusterIP: v1.ClusterIPNone,
					},
				},
			},
			endpoints: nil,

			clusterMode:  expectation{error: true},
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
						Ports: []v1.ServicePort{
							{Name: "https", Port: 443, TargetPort: intstr.FromInt(1443)},
							{Port: 1234, TargetPort: intstr.FromInt(1234)},
						},
					},
				},
			},
			endpoints: matchingEndpoints,

			clusterMode:  expectation{url: "https://lb:443"},
			endpointMode: expectation{url: "https://127.0.0.1:1443"},
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
							{Name: "https", Port: 443, TargetPort: intstr.FromInt(1443)},
							{Port: 1234, TargetPort: intstr.FromInt(1234)},
						},
					},
				},
			},
			endpoints: matchingEndpoints,

			clusterMode:  expectation{url: "https://np:443"},
			endpointMode: expectation{url: "https://127.0.0.1:1443"},
		},
		{
			name: "external name",
			services: []*v1.Service{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "alfa"},
					Spec: v1.ServiceSpec{
						Type:         v1.ServiceTypeExternalName,
						ExternalName: "foo.bar.com",
					},
				},
			},
			endpoints: nil,

			clusterMode:  expectation{url: "https://foo.bar.com:443"},
			endpointMode: expectation{error: true},
		},
		{
			name:      "missing service",
			services:  nil,
			endpoints: nil,

			clusterMode:  expectation{error: true},
			endpointMode: expectation{error: true},
		},
	}

	for _, test := range tests {
		serviceCache := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
		serviceLister := v1listers.NewServiceLister(serviceCache)
		for i := range test.services {
			if err := serviceCache.Add(test.services[i]); err != nil {
				t.Fatalf("%s unexpected service add error: %v", test.name, err)
			}
		}

		endpointCache := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
		endpointLister := v1listers.NewEndpointsLister(endpointCache)
		if test.endpoints != nil {
			for _, svc := range test.services {
				for _, ep := range test.endpoints(svc) {
					if err := endpointCache.Add(ep); err != nil {
						t.Fatalf("%s unexpected endpoint add error: %v", test.name, err)
					}
				}
			}
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

		clusterURL, err := ResolveCluster(serviceLister, "one", "alfa", 443)
		check("cluster", test.clusterMode, clusterURL, err)

		endpointURL, err := ResolveEndpoint(serviceLister, endpointLister, "one", "alfa", 443)
		check("endpoint", test.endpointMode, endpointURL, err)
	}
}
