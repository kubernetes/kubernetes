/*
Copyright 2019 The Kubernetes Authors.

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

package endpointslice

import (
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilpointer "k8s.io/utils/pointer"
)

func TestGetEndpointsByPort(t *testing.T) {
	ns := "foo"
	endpoints, ports, endpointSlices := setupTestData(ns)

	testCases := map[string]struct {
		endpointSlices          []*discovery.EndpointSlice
		expectedEndpointsByPort EndpointsByPort
	}{
		"1 endpoint slice, multiple ports": {
			endpointSlices: endpointSlices[0:1],
			expectedEndpointsByPort: EndpointsByPort{
				ports[0]: endpoints[0:3],
				ports[1]: endpoints[0:3],
			},
		},
		"2 endpoint slices, multiple ports": {
			endpointSlices: endpointSlices[0:2],
			expectedEndpointsByPort: EndpointsByPort{
				ports[0]: endpoints[0:5],
				ports[1]: endpoints[0:3],
				ports[3]: endpoints[3:5],
			},
		},
		"1 endpoint slice, 1 port": {
			endpointSlices: endpointSlices[2:3],
			expectedEndpointsByPort: EndpointsByPort{
				ports[2]: endpoints[5:],
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			endpointsByPort := GetEndpointsByPort(tc.endpointSlices)

			if len(tc.expectedEndpointsByPort) != len(endpointsByPort) {
				t.Fatalf("Expected %d ports, got %d", len(tc.expectedEndpointsByPort), len(endpointsByPort))
			}

			for port, endpoints := range endpointsByPort {
				expectedEndpoints, ok := tc.expectedEndpointsByPort[port]
				if !ok {
					t.Fatalf("Got unexpected port: %v", port)
				}
				if !endpointsEqual(expectedEndpoints, endpoints) {
					t.Errorf("Expected endpoints: %v, got: %v", expectedEndpoints, endpoints)
				}
			}
		})
	}
}

func TestGetEndpointsForPort(t *testing.T) {
	ns := "foo"
	endpoints, ports, endpointSlices := setupTestData(ns)

	testCases := map[string]struct {
		portName                string
		expectedEndpointsByPort EndpointsByPort
	}{
		"http": {
			portName: "http",
			expectedEndpointsByPort: EndpointsByPort{
				ports[1]: endpoints[0:3],
			},
		},
		"https": {
			portName: "https",
			expectedEndpointsByPort: EndpointsByPort{
				ports[0]: endpoints[0:5],
			},
		},
		"dns": {
			portName: "dns",
			expectedEndpointsByPort: EndpointsByPort{
				ports[2]: endpoints[5:],
				ports[3]: endpoints[3:5],
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			endpointsByPort := GetEndpointsForPort(tc.portName, endpointSlices)

			if len(tc.expectedEndpointsByPort) != len(endpointsByPort) {
				t.Fatalf("Expected %d ports, got %d", len(tc.expectedEndpointsByPort), len(endpointsByPort))
			}

			for port, endpoints := range endpointsByPort {
				expectedEndpoints, ok := tc.expectedEndpointsByPort[port]
				if !ok {
					t.Fatalf("Got unexpected port: %v", port)
				}
				if !endpointsEqual(expectedEndpoints, endpoints) {
					t.Errorf("Expected endpoints: %v, got: %v", expectedEndpoints, endpoints)
				}
			}
		})
	}
}

func setupTestData(ns string) ([]discovery.Endpoint, []discovery.EndpointPort, []*discovery.EndpointSlice) {
	tcpProtocol := corev1.ProtocolTCP
	udpProtocol := corev1.ProtocolUDP

	serviceNames := []string{"svc1", "svc2"}

	ports := []discovery.EndpointPort{{
		Name:     utilpointer.StringPtr("https"),
		Port:     utilpointer.Int32Ptr(443),
		Protocol: &tcpProtocol,
	}, {
		Name:     utilpointer.StringPtr("http"),
		Port:     utilpointer.Int32Ptr(80),
		Protocol: &tcpProtocol,
	}, {
		Name:     utilpointer.StringPtr("dns"),
		Port:     utilpointer.Int32Ptr(53),
		Protocol: &udpProtocol,
	}, {
		Name:     utilpointer.StringPtr("dns"),
		Port:     utilpointer.Int32Ptr(3053),
		Protocol: &udpProtocol,
	}}

	endpoints := []discovery.Endpoint{{
		Addresses: []string{"10.0.0.1"},
	}, {
		Addresses: []string{"10.0.0.2"},
	}, {
		Addresses: []string{"10.0.0.3"},
	}, {
		Addresses: []string{"10.0.0.4"},
	}, {
		Addresses: []string{"10.0.0.5"},
	}, {
		Addresses: []string{"10.0.0.6"},
	}}

	endpointSlices := []*discovery.EndpointSlice{{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "epslice1",
			Namespace: ns,
			Labels: map[string]string{
				discovery.LabelServiceName: serviceNames[0],
				"sliceName":                "epslice1",
			},
		},
		Ports:       []discovery.EndpointPort{ports[0], ports[1]},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints:   endpoints[0:3],
	}, {
		ObjectMeta: metav1.ObjectMeta{
			Name:      "epslice2",
			Namespace: ns,
			Labels: map[string]string{
				discovery.LabelServiceName: serviceNames[0],
				"sliceName":                "epslice2",
			},
		},
		Ports:       []discovery.EndpointPort{ports[0], ports[3]},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints:   endpoints[3:5],
	}, {
		ObjectMeta: metav1.ObjectMeta{
			Name:      "epslice3",
			Namespace: ns,
			Labels: map[string]string{
				discovery.LabelServiceName: serviceNames[1],
				"sliceName":                "epslice3",
			},
		},
		Ports:       []discovery.EndpointPort{ports[2]},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints:   endpoints[5:],
	}}

	return endpoints, ports, endpointSlices
}

func endpointsEqual(endpoints1, endpoints2 []discovery.Endpoint) bool {
	if len(endpoints1) != len(endpoints2) {
		return false
	}

	for _, ep1 := range endpoints1 {
		if !endpointIn(ep1, endpoints2) {
			return false
		}
	}

	return true
}

func endpointIn(endpoint discovery.Endpoint, endpoints []discovery.Endpoint) bool {
	for _, ep := range endpoints {
		if reflect.DeepEqual(ep, endpoint) {
			return true
		}
	}

	return false
}
