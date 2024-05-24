//go:build linux
// +build linux

/*
Copyright 2023 The Kubernetes Authors.

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

package conntrack

import (
	"net"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/proxy"
)

const (
	testClusterIP      = "172.30.1.1"
	testExternalIP     = "192.168.99.100"
	testLoadBalancerIP = "1.2.3.4"

	testEndpointIP = "10.240.0.4"

	testPort         = 53
	testNodePort     = 5353
	testEndpointPort = "5300"
)

func TestCleanStaleEntries(t *testing.T) {
	// We need to construct a proxy.ServicePortMap to pass to CleanStaleEntries.
	// ServicePortMap is just map[string]proxy.ServicePort, but there are no public
	// constructors for any implementation of proxy.ServicePort, so we have to either
	// provide our own implementation of that interface, or else use a
	// proxy.ServiceChangeTracker to construct them and fill in the map for us.

	sct := proxy.NewServiceChangeTracker(nil, v1.IPv4Protocol, nil, nil)
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "cleanup-test",
			Namespace: "test",
		},
		Spec: v1.ServiceSpec{
			ClusterIP:   testClusterIP,
			ExternalIPs: []string{testExternalIP},
			Ports: []v1.ServicePort{
				{
					Name:     "dns-tcp",
					Port:     testPort,
					Protocol: v1.ProtocolTCP,
				},
				{
					Name:     "dns-udp",
					Port:     testPort,
					NodePort: testNodePort,
					Protocol: v1.ProtocolUDP,
				},
			},
		},
		Status: v1.ServiceStatus{
			LoadBalancer: v1.LoadBalancerStatus{
				Ingress: []v1.LoadBalancerIngress{{
					IP: testLoadBalancerIP,
				}},
			},
		},
	}
	sct.Update(nil, svc)

	svcPortMap := make(proxy.ServicePortMap)
	_ = svcPortMap.Update(sct)

	// (At this point we are done with sct, and in particular, we don't use sct to
	// construct UpdateServiceMapResults, because pkg/proxy already has its own tests
	// for that. Also, svcPortMap is read-only from this point on.)

	tcpPortName := proxy.ServicePortName{
		NamespacedName: types.NamespacedName{
			Namespace: svc.Namespace,
			Name:      svc.Name,
		},
		Port:     svc.Spec.Ports[0].Name,
		Protocol: svc.Spec.Ports[0].Protocol,
	}

	udpPortName := proxy.ServicePortName{
		NamespacedName: types.NamespacedName{
			Namespace: svc.Namespace,
			Name:      svc.Name,
		},
		Port:     svc.Spec.Ports[1].Name,
		Protocol: svc.Spec.Ports[1].Protocol,
	}

	unknownPortName := udpPortName
	unknownPortName.Namespace = "unknown"

	// Sanity-check to make sure we constructed the map correctly
	if len(svcPortMap) != 2 {
		t.Fatalf("expected svcPortMap to have 2 entries, got %+v", svcPortMap)
	}
	servicePort := svcPortMap[tcpPortName]
	if servicePort == nil || servicePort.String() != "172.30.1.1:53/TCP" {
		t.Fatalf("expected svcPortMap[%q] to be \"172.30.1.1:53/TCP\", got %q", tcpPortName.String(), servicePort.String())
	}
	servicePort = svcPortMap[udpPortName]
	if servicePort == nil || servicePort.String() != "172.30.1.1:53/UDP" {
		t.Fatalf("expected svcPortMap[%q] to be \"172.30.1.1:53/UDP\", got %q", udpPortName.String(), servicePort.String())
	}

	testCases := []struct {
		description string

		serviceUpdates   proxy.UpdateServiceMapResult
		endpointsUpdates proxy.UpdateEndpointsMapResult

		result FakeInterface
	}{
		{
			description: "DeletedUDPClusterIPs clears entries for given clusterIPs (only)",

			serviceUpdates: proxy.UpdateServiceMapResult{
				// Note: this isn't testClusterIP; it's the IP of some
				// unknown (because deleted) service.
				DeletedUDPClusterIPs: sets.New("172.30.99.99"),
			},
			endpointsUpdates: proxy.UpdateEndpointsMapResult{},

			result: FakeInterface{
				ClearedIPs: sets.New("172.30.99.99"),

				ClearedPorts:    sets.New[int](),
				ClearedNATs:     map[string]string{},
				ClearedPortNATs: map[int]string{},
			},
		},
		{
			description: "DeletedUDPEndpoints clears NAT entries for all IPs and NodePorts",

			serviceUpdates: proxy.UpdateServiceMapResult{
				DeletedUDPClusterIPs: sets.New[string](),
			},
			endpointsUpdates: proxy.UpdateEndpointsMapResult{
				DeletedUDPEndpoints: []proxy.ServiceEndpoint{{
					Endpoint:        net.JoinHostPort(testEndpointIP, testEndpointPort),
					ServicePortName: udpPortName,
				}},
			},

			result: FakeInterface{
				ClearedIPs:   sets.New[string](),
				ClearedPorts: sets.New[int](),

				ClearedNATs: map[string]string{
					testClusterIP:      testEndpointIP,
					testExternalIP:     testEndpointIP,
					testLoadBalancerIP: testEndpointIP,
				},
				ClearedPortNATs: map[int]string{
					testNodePort: testEndpointIP,
				},
			},
		},
		{
			description: "NewlyActiveUDPServices clears entries for all IPs and NodePorts",

			serviceUpdates: proxy.UpdateServiceMapResult{
				DeletedUDPClusterIPs: sets.New[string](),
			},
			endpointsUpdates: proxy.UpdateEndpointsMapResult{
				DeletedUDPEndpoints: []proxy.ServiceEndpoint{},
				NewlyActiveUDPServices: []proxy.ServicePortName{
					udpPortName,
				},
			},

			result: FakeInterface{
				ClearedIPs:   sets.New(testClusterIP, testExternalIP, testLoadBalancerIP),
				ClearedPorts: sets.New(testNodePort),

				ClearedNATs:     map[string]string{},
				ClearedPortNATs: map[int]string{},
			},
		},

		{
			description: "DeletedUDPEndpoints for unknown Service has no effect",

			serviceUpdates: proxy.UpdateServiceMapResult{
				DeletedUDPClusterIPs: sets.New[string](),
			},
			endpointsUpdates: proxy.UpdateEndpointsMapResult{
				DeletedUDPEndpoints: []proxy.ServiceEndpoint{{
					Endpoint:        "10.240.0.4:80",
					ServicePortName: unknownPortName,
				}},
				NewlyActiveUDPServices: []proxy.ServicePortName{},
			},

			result: FakeInterface{
				ClearedIPs:      sets.New[string](),
				ClearedPorts:    sets.New[int](),
				ClearedNATs:     map[string]string{},
				ClearedPortNATs: map[int]string{},
			},
		},
		{
			description: "NewlyActiveUDPServices for unknown Service has no effect",

			serviceUpdates: proxy.UpdateServiceMapResult{
				DeletedUDPClusterIPs: sets.New[string](),
			},
			endpointsUpdates: proxy.UpdateEndpointsMapResult{
				DeletedUDPEndpoints: []proxy.ServiceEndpoint{},
				NewlyActiveUDPServices: []proxy.ServicePortName{
					unknownPortName,
				},
			},

			result: FakeInterface{
				ClearedIPs:      sets.New[string](),
				ClearedPorts:    sets.New[int](),
				ClearedNATs:     map[string]string{},
				ClearedPortNATs: map[int]string{},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			fake := NewFake()
			CleanStaleEntries(fake, svcPortMap, tc.serviceUpdates, tc.endpointsUpdates)
			if !fake.ClearedIPs.Equal(tc.result.ClearedIPs) {
				t.Errorf("Expected ClearedIPs=%v, got %v", tc.result.ClearedIPs, fake.ClearedIPs)
			}
			if !fake.ClearedPorts.Equal(tc.result.ClearedPorts) {
				t.Errorf("Expected ClearedPorts=%v, got %v", tc.result.ClearedPorts, fake.ClearedPorts)
			}
			if !reflect.DeepEqual(fake.ClearedNATs, tc.result.ClearedNATs) {
				t.Errorf("Expected ClearedNATs=%v, got %v", tc.result.ClearedNATs, fake.ClearedNATs)
			}
			if !reflect.DeepEqual(fake.ClearedPortNATs, tc.result.ClearedPortNATs) {
				t.Errorf("Expected ClearedPortNATs=%v, got %v", tc.result.ClearedPortNATs, fake.ClearedPortNATs)
			}
		})
	}
}
