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

package validation

import (
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/discovery"
	utilpointer "k8s.io/utils/pointer"
)

func TestValidateEndpointSlice(t *testing.T) {
	standardMeta := metav1.ObjectMeta{
		Name:      "hello",
		Namespace: "world",
	}

	testCases := map[string]struct {
		expectedErrors int
		endpointSlice  *discovery.EndpointSlice
	}{
		"good-slice": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateAddresses(1),
					Hostname:  utilpointer.StringPtr("valid-123"),
				}},
			},
		},
		"all-protocols": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("tcp"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}, {
					Name:     utilpointer.StringPtr("udp"),
					Protocol: protocolPtr(api.ProtocolUDP),
				}, {
					Name:     utilpointer.StringPtr("sctp"),
					Protocol: protocolPtr(api.ProtocolSCTP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateAddresses(1),
					Hostname:  utilpointer.StringPtr("valid-123"),
				}},
			},
		},
		"empty-port-name": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr(""),
					Protocol: protocolPtr(api.ProtocolTCP),
				}, {
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateAddresses(1),
				}},
			},
		},
		"empty-ports-and-endpoints": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports:       []discovery.EndpointPort{},
				Endpoints:   []discovery.Endpoint{},
			},
		},
		"max-endpoints": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports:       generatePorts(1),
				Endpoints:   generateEndpoints(maxEndpoints),
			},
		},
		"max-ports": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports:       generatePorts(maxPorts),
			},
		},
		"max-addresses": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateAddresses(maxAddresses),
				}},
			},
		},
		"max-topology-keys": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateAddresses(1),
					Topology:  generateTopology(maxTopologyLabels),
				}},
			},
		},

		// expected failures
		"duplicate-port-name": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr(""),
					Protocol: protocolPtr(api.ProtocolTCP),
				}, {
					Name:     utilpointer.StringPtr(""),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{},
			},
		},
		"bad-port-name-caps": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("aCapital"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{},
			},
		},
		"bad-port-name-chars": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("almost_valid"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{},
			},
		},
		"invalid-port-protocol": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.Protocol("foo")),
				}},
			},
		},
		"too-many-ports": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports:       generatePorts(maxPorts + 1),
			},
		},
		"too-many-endpoints": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports:       generatePorts(1),
				Endpoints:   generateEndpoints(maxEndpoints + 1),
			},
		},
		"no-endpoint-addresses": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateAddresses(0),
				}},
			},
		},
		"too-many-addresses": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateAddresses(maxAddresses + 1),
				}},
			},
		},
		"bad-address-type": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressType("other")),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateAddresses(1),
				}},
			},
		},
		"bad-topology-key": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateAddresses(1),
					Topology:  map[string]string{"--INVALID": "example"},
				}},
			},
		},
		"too-many-topology-keys": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateAddresses(1),
					Topology:  generateTopology(maxTopologyLabels + 1),
				}},
			},
		},
		"bad-hostname": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateAddresses(1),
					Hostname:  utilpointer.StringPtr("--INVALID"),
				}},
			},
		},
		"bad-meta": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "*&^",
					Namespace: "foo",
				},
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateAddresses(1),
					Hostname:  utilpointer.StringPtr("valid-123"),
				}},
			},
		},
		"empty-everything": {
			expectedErrors: 3,
			endpointSlice:  &discovery.EndpointSlice{},
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			errs := ValidateEndpointSlice(testCase.endpointSlice)
			if len(errs) != testCase.expectedErrors {
				t.Errorf("Expected %d errors, got %d errors: %v", testCase.expectedErrors, len(errs), errs)
			}
		})
	}
}

func TestValidateEndpointSliceUpdate(t *testing.T) {
	standardMeta := metav1.ObjectMeta{Name: "es1", Namespace: "test"}

	testCases := map[string]struct {
		expectedErrors   int
		newEndpointSlice *discovery.EndpointSlice
		oldEndpointSlice *discovery.EndpointSlice
	}{
		"valid and identical slices": {
			newEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
			},
			oldEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
			},
			expectedErrors: 0,
		},
		"valid and identical slices with different address types": {
			newEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
			},
			oldEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressType("other")),
			},
			expectedErrors: 1,
		},
		"invalid slices with valid address types": {
			newEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr(""),
					Protocol: protocolPtr(api.Protocol("invalid")),
				}},
			},
			oldEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: addressTypePtr(discovery.AddressTypeIP),
			},
			expectedErrors: 1,
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			errs := ValidateEndpointSliceUpdate(testCase.newEndpointSlice, testCase.oldEndpointSlice)
			if len(errs) != testCase.expectedErrors {
				t.Errorf("Expected %d errors, got %d errors: %v", testCase.expectedErrors, len(errs), errs)
			}
		})
	}
}

// Test helpers

func protocolPtr(protocol api.Protocol) *api.Protocol {
	return &protocol
}

func addressTypePtr(addressType discovery.AddressType) *discovery.AddressType {
	return &addressType
}

func generatePorts(n int) []discovery.EndpointPort {
	ports := []discovery.EndpointPort{}
	for i := 0; i < n; i++ {
		ports = append(ports, discovery.EndpointPort{
			Name:     utilpointer.StringPtr(fmt.Sprintf("http-%d", i)),
			Protocol: protocolPtr(api.ProtocolTCP),
		})
	}
	return ports
}

func generateEndpoints(n int) []discovery.Endpoint {
	endpoints := []discovery.Endpoint{}
	for i := 0; i < n; i++ {
		endpoints = append(endpoints, discovery.Endpoint{
			Addresses: []string{fmt.Sprintf("10.1.2.%d", i%255)},
		})
	}
	return endpoints
}

func generateAddresses(n int) []string {
	addresses := []string{}
	for i := 0; i < n; i++ {
		addresses = append(addresses, fmt.Sprintf("10.1.2.%d", i%255))
	}
	return addresses
}

func generateTopology(n int) map[string]string {
	topology := map[string]string{}
	for i := 0; i < n; i++ {
		topology[fmt.Sprintf("topology-%d", i)] = "example"
	}
	return topology
}
