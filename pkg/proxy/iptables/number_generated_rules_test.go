/*
Copyright 2022 The Kubernetes Authors.

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

package iptables

import (
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	iptablestest "k8s.io/kubernetes/pkg/util/iptables/testing"
	netutils "k8s.io/utils/net"
	"k8s.io/utils/pointer"
)

// kube-proxy generates iptables rules to forward traffic from Services to Endpoints
// kube-proxy uses iptables-restore to configure the rules atomically, however,
// this has the downside that large number of rules take a long time to be processed,
// causing disruption.
// There are different parameters than influence the number of rules generated:
// - ServiceType
// - Number of Services
// - Number of Endpoints per Service
// This test will fail when the number of rules change, so the person
// that is modifying the code can have feedback about the performance impact
// on their changes. It also runs multiple number of rules test cases to check
// if the number of rules grows linearly.
func TestNumberIptablesRules(t *testing.T) {
	testCases := []struct {
		name                string
		epsFunc             func(eps *discovery.EndpointSlice)
		svcFunc             func(svc *v1.Service)
		services            int
		epPerService        int
		expectedFilterRules int
		expectedNatRules    int
	}{
		{
			name:                "0 Services 0 EndpointsPerService - ClusterIP",
			services:            0,
			epPerService:        0,
			expectedFilterRules: 3,
			expectedNatRules:    5,
		},
		{
			name:                "1 Services 0 EndpointPerService - ClusterIP",
			services:            1,
			epPerService:        0,
			expectedFilterRules: 4,
			expectedNatRules:    5,
		},
		{
			name:                "1 Services 1 EndpointPerService - ClusterIP",
			services:            1,
			epPerService:        1,
			expectedFilterRules: 3,
			expectedNatRules:    10,
		},
		{
			name:                "1 Services 2 EndpointPerService - ClusterIP",
			services:            1,
			epPerService:        2,
			expectedFilterRules: 3,
			expectedNatRules:    13,
		},
		{
			name:                "1 Services 10 EndpointPerService - ClusterIP",
			services:            1,
			epPerService:        10,
			expectedFilterRules: 3,
			expectedNatRules:    37,
		},
		{
			name:                "10 Services 0 EndpointsPerService - ClusterIP",
			services:            10,
			epPerService:        0,
			expectedFilterRules: 13,
			expectedNatRules:    5,
		},
		{
			name:                "10 Services 1 EndpointPerService - ClusterIP",
			services:            10,
			epPerService:        1,
			expectedFilterRules: 3,
			expectedNatRules:    55,
		},
		{
			name:                "10 Services 2 EndpointPerService - ClusterIP",
			services:            10,
			epPerService:        2,
			expectedFilterRules: 3,
			expectedNatRules:    85,
		},
		{
			name:                "10 Services 10 EndpointPerService - ClusterIP",
			services:            10,
			epPerService:        10,
			expectedFilterRules: 3,
			expectedNatRules:    325,
		},

		{
			name: "0 Services 0 EndpointsPerService - LoadBalancer",
			svcFunc: func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeLoadBalancer
				svc.Spec.ExternalIPs = []string{"1.2.3.4"}
				svc.Spec.LoadBalancerSourceRanges = []string{" 1.2.3.4/28"}
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
					IP: "1.2.3.4",
				}}
			},
			services:            0,
			epPerService:        0,
			expectedFilterRules: 3,
			expectedNatRules:    5,
		},
		{
			name: "1 Services 0 EndpointPerService - LoadBalancer",
			svcFunc: func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeLoadBalancer
				svc.Spec.ExternalIPs = []string{"1.2.3.4"}
				svc.Spec.LoadBalancerSourceRanges = []string{" 1.2.3.4/28"}
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
					IP: "1.2.3.4",
				}}
			},
			services:            1,
			epPerService:        0,
			expectedFilterRules: 7,
			expectedNatRules:    5,
		},
		{
			name: "1 Services 1 EndpointPerService - LoadBalancer",
			svcFunc: func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeLoadBalancer
				svc.Spec.ExternalIPs = []string{"1.2.3.4"}
				svc.Spec.LoadBalancerSourceRanges = []string{" 1.2.3.4/28"}
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
					IP: "1.2.3.4",
				}}
			},
			services:            1,
			epPerService:        1,
			expectedFilterRules: 4,
			expectedNatRules:    16,
		},
		{
			name: "1 Services 2 EndpointPerService - LoadBalancer",
			svcFunc: func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeLoadBalancer
				svc.Spec.ExternalIPs = []string{"1.2.3.4"}
				svc.Spec.LoadBalancerSourceRanges = []string{" 1.2.3.4/28"}
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
					IP: "1.2.3.4",
				}}
			},
			services:            1,
			epPerService:        2,
			expectedFilterRules: 4,
			expectedNatRules:    19,
		},
		{
			name: "1 Services 10 EndpointPerService - LoadBalancer",
			svcFunc: func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeLoadBalancer
				svc.Spec.ExternalIPs = []string{"1.2.3.4"}
				svc.Spec.LoadBalancerSourceRanges = []string{" 1.2.3.4/28"}
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
					IP: "1.2.3.4",
				}}
			},
			services:            1,
			epPerService:        10,
			expectedFilterRules: 4,
			expectedNatRules:    43,
		},
		{
			name: "10 Services 0 EndpointsPerService - LoadBalancer",
			svcFunc: func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeLoadBalancer
				svc.Spec.ExternalIPs = []string{"1.2.3.4"}
				svc.Spec.LoadBalancerSourceRanges = []string{" 1.2.3.4/28"}
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
					IP: "1.2.3.4",
				}}
			},
			services:            10,
			epPerService:        0,
			expectedFilterRules: 43,
			expectedNatRules:    5,
		},
		{
			name: "10 Services 1 EndpointPerService - LoadBalancer",
			svcFunc: func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeLoadBalancer
				svc.Spec.ExternalIPs = []string{"1.2.3.4"}
				svc.Spec.LoadBalancerSourceRanges = []string{" 1.2.3.4/28"}
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
					IP: "1.2.3.4",
				}}
			},
			services:            10,
			epPerService:        1,
			expectedFilterRules: 13,
			expectedNatRules:    115,
		},
		{
			name: "10 Services 2 EndpointPerService - LoadBalancer",
			svcFunc: func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeLoadBalancer
				svc.Spec.ExternalIPs = []string{"1.2.3.4"}
				svc.Spec.LoadBalancerSourceRanges = []string{" 1.2.3.4/28"}
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
					IP: "1.2.3.4",
				}}
			},
			services:            10,
			epPerService:        2,
			expectedFilterRules: 13,
			expectedNatRules:    145,
		},
		{
			name: "10 Services 10 EndpointPerService - LoadBalancer",
			svcFunc: func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeLoadBalancer
				svc.Spec.ExternalIPs = []string{"1.2.3.4"}
				svc.Spec.LoadBalancerSourceRanges = []string{" 1.2.3.4/28"}
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
					IP: "1.2.3.4",
				}}
			},
			services:            10,
			epPerService:        10,
			expectedFilterRules: 13,
			expectedNatRules:    385,
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			ipt := iptablestest.NewFake()
			fp := NewFakeProxier(ipt)

			svcs, eps := generateServiceEndpoints(test.services, test.epPerService, test.epsFunc, test.svcFunc)

			makeServiceMap(fp, svcs...)
			populateEndpointSlices(fp, eps...)

			now := time.Now()
			fp.syncProxyRules()
			t.Logf("time to sync rule: %v", time.Since(now))
			t.Logf("iptables data size: %d bytes", fp.iptablesData.Len())

			if fp.filterRules.Lines() != test.expectedFilterRules {
				t.Errorf("expected number of Filter rules: %d, got: %d", test.expectedFilterRules, fp.filterRules.Lines())
			}

			if fp.natRules.Lines() != test.expectedNatRules {
				t.Errorf("expected number of NAT rules: %d, got: %d", test.expectedNatRules, fp.natRules.Lines())
			}

			// print generated iptables data
			// t.Logf("Generated rules:\n %s", fp.iptablesData.String())
		})
	}
}

func Test_generateServiceEndpoints(t *testing.T) {
	testCases := []struct {
		name         string
		services     int
		epPerService int
		svcType      v1.ServiceType
	}{
		{
			name:         "Generate 10 Services with 10 Endpoints per Service and LoadBalancer Type",
			services:     10,
			epPerService: 10,
			svcType:      v1.ServiceTypeLoadBalancer,
		},
		{
			name:         "Generate 10 Services with 20 Endpoints per Service and NodePort Type",
			services:     10,
			epPerService: 20,
			svcType:      v1.ServiceTypeNodePort,
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			// test the function to mutate services
			svcFunc := func(svc *v1.Service) {
				svc.Spec.Type = test.svcType
			}
			// test the function to mutate endpoint slices
			epsFunc := func(eps *discovery.EndpointSlice) {
				for i := range eps.Endpoints {
					nodeName := fmt.Sprintf("node-%d", i)
					eps.Endpoints[i].NodeName = &nodeName
				}
			}

			svcs, eps := generateServiceEndpoints(test.services, test.epPerService, epsFunc, svcFunc)

			if len(svcs) != test.services {
				t.Fatalf("expected %d service, received %d", test.services, len(svcs))
			}
			if len(eps) != test.services {
				t.Fatalf("expected %d endpoint slice , received %d", test.services, len(eps))
			}

			for i := 0; i < test.services; i++ {
				if svcs[i].Spec.Type != test.svcType {
					t.Fatalf("expected Service Type %s, got %s", test.svcType, svcs[i].Spec.Type)
				}
				if eps[i].ObjectMeta.Labels[discovery.LabelServiceName] != svcs[i].Name {
					t.Fatalf("endpoint slice reference %s instead of Service %s", eps[i].ObjectMeta.Labels[discovery.LabelServiceName], svcs[i].Name)
				}
				if len(eps[i].Endpoints) != test.epPerService {
					t.Fatalf("expected %d endpoints per slice , received %d", test.epPerService, len(eps[i].Endpoints))
				}
				for j := 0; j < test.epPerService; j++ {
					nodeName := fmt.Sprintf("node-%d", j)
					if *eps[i].Endpoints[j].NodeName != nodeName {
						t.Errorf("Endpoint %d on EndpointSlice %d expected Nodename %s, got %s", j, i, nodeName, *eps[i].Endpoints[j].NodeName)
					}
				}
			}
		})
	}

}

// generateServiceEndpoints generate Services with the Type specified and it creates N Endpoints per Service
func generateServiceEndpoints(nServices, nEndpoints int, epsFunc func(eps *discovery.EndpointSlice), svcFunc func(svc *v1.Service)) ([]*v1.Service, []*discovery.EndpointSlice) {
	services := make([]*v1.Service, nServices)
	endpointSlices := make([]*discovery.EndpointSlice, nServices)

	// base parameters
	basePort := 80
	base := netutils.BigForIP(netutils.ParseIPSloppy("10.0.0.1"))

	// generate a base endpoint slice object
	baseEp := netutils.BigForIP(netutils.ParseIPSloppy("172.16.0.1"))
	epPort := 8080

	tcpProtocol := v1.ProtocolTCP

	eps := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "ep",
			Namespace: "namespace",
		},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints:   []discovery.Endpoint{},
		Ports: []discovery.EndpointPort{{
			Name:     pointer.String(fmt.Sprintf("%d", epPort)),
			Port:     pointer.Int32(int32(epPort)),
			Protocol: &tcpProtocol,
		}},
	}

	for j := 0; j < nEndpoints; j++ {
		ipEp := netutils.AddIPOffset(baseEp, j)
		eps.Endpoints = append(eps.Endpoints, discovery.Endpoint{
			Addresses: []string{ipEp.String()},
		})
	}

	if epsFunc != nil {
		epsFunc(eps)
	}

	// generate a base service object
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "svc",
			Namespace: "namespace",
		},
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeClusterIP,
		},
	}

	if svcFunc != nil {
		svcFunc(svc)
	}

	// Create the Services and associate and endpoint slice object to each one
	for i := 0; i < nServices; i++ {
		ip := netutils.AddIPOffset(base, i)
		services[i] = svc.DeepCopy()
		services[i].Name = fmt.Sprintf("svc%d", i)
		services[i].Spec.ClusterIP = ip.String()
		services[i].Spec.Ports = []v1.ServicePort{
			{
				Name:       fmt.Sprintf("%d", epPort),
				Protocol:   v1.ProtocolTCP,
				Port:       int32(basePort + i),
				TargetPort: intstr.FromInt(epPort),
			},
		}

		if svc.Spec.Type == v1.ServiceTypeNodePort || svc.Spec.Type == v1.ServiceTypeLoadBalancer {
			services[i].Spec.Ports[0].NodePort = int32(30000 + i)

		}
		if svc.Spec.Type == v1.ServiceTypeLoadBalancer {
			services[i].Spec.HealthCheckNodePort = int32(32000 + nServices + i)
		}

		endpointSlices[i] = eps.DeepCopy()
		endpointSlices[i].Name = services[i].Name
		endpointSlices[i].ObjectMeta.Labels = map[string]string{
			discovery.LabelServiceName: services[i].Name,
		}

	}

	return services, endpointSlices
}
