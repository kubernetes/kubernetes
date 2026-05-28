//go:build linux

/*
Copyright The Kubernetes Authors.

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

package nftables

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/utils/ptr"
)

func generateServicesAndEndpoints(numServices, numEndpointsPerService int) ([]*v1.Service, []*discovery.EndpointSlice) {
	services := make([]*v1.Service, numServices)
	endpointSlices := make([]*discovery.EndpointSlice, numServices)
	for i := 0; i < numServices; i++ {
		svcName := fmt.Sprintf("svc-%d", i)
		clusterIP := fmt.Sprintf("172.30.%d.%d", (i/250)+1, (i%250)+1)
		services[i] = makeTestService("ns", svcName, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = clusterIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     "p80",
				Port:     80,
				Protocol: v1.ProtocolTCP,
			}}
		})

		eps := makeTestEndpointSlice("ns", svcName, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To("p80"),
				Port:     ptr.To[int32](80),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
			for j := 0; j < numEndpointsPerService; j++ {
				ip := fmt.Sprintf("10.%d.%d.%d", (j/250)+1, ((i*numEndpointsPerService+j)/250)%250+1, ((i*numEndpointsPerService+j)%250)+1)
				eps.Endpoints = append(eps.Endpoints, discovery.Endpoint{
					Addresses:  []string{ip},
					Conditions: discovery.EndpointConditions{Ready: ptr.To(true)},
					NodeName:   ptr.To(testNodeName),
				})
			}
		})
		endpointSlices[i] = eps
	}
	return services, endpointSlices
}

func runSyncBenchmark(b *testing.B, numServices, numEndpointsPerService int) {
	services, endpointSlices := generateServicesAndEndpoints(numServices, numEndpointsPerService)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		_, fp := NewFakeProxier(v1.IPv4Protocol)
		// Register all services and endpoints
		for _, svc := range services {
			fp.OnServiceAdd(svc)
		}
		for _, eps := range endpointSlices {
			fp.OnEndpointSliceAdd(eps)
		}
		b.StartTimer()

		// Run cold start ruleset generation
		_ = fp.syncProxyRules()
	}
}

func BenchmarkSyncProxyRules_100_Services(b *testing.B) {
	runSyncBenchmark(b, 100, 10)
}

func BenchmarkSyncProxyRules_1000_Services(b *testing.B) {
	runSyncBenchmark(b, 1000, 10)
}

func BenchmarkSyncProxyRules_5000_Services(b *testing.B) {
	runSyncBenchmark(b, 5000, 10)
}
