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
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/discovery"
	"k8s.io/kubernetes/pkg/features"
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
		enableTopology bool
	}{
		"good-slice": {
			enableTopology: true,
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hostname:  utilpointer.StringPtr("valid-123"),
				}},
			},
		},
		"good-fqdns": {
			enableTopology: true,
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeFQDN,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: []string{"foo.example.com", "example.com", "example.com.", "hyphens-are-good.example.com"},
					Hostname:  utilpointer.StringPtr("valid-123"),
				}},
			},
		},
		"all-protocols": {
			enableTopology: true,
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
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
					Addresses: generateIPAddresses(1),
					Hostname:  utilpointer.StringPtr("valid-123"),
				}},
			},
		},
		"app-protocols": {
			enableTopology: true,
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:        utilpointer.StringPtr("one"),
					Protocol:    protocolPtr(api.ProtocolTCP),
					AppProtocol: utilpointer.StringPtr("HTTP"),
				}, {
					Name:        utilpointer.StringPtr("two"),
					Protocol:    protocolPtr(api.ProtocolTCP),
					AppProtocol: utilpointer.StringPtr("https"),
				}, {
					Name:        utilpointer.StringPtr("three"),
					Protocol:    protocolPtr(api.ProtocolTCP),
					AppProtocol: utilpointer.StringPtr("my-protocol"),
				}, {
					Name:        utilpointer.StringPtr("four"),
					Protocol:    protocolPtr(api.ProtocolTCP),
					AppProtocol: utilpointer.StringPtr("example.com/custom-protocol"),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hostname:  utilpointer.StringPtr("valid-123"),
				}},
			},
		},
		"empty-port-name": {
			enableTopology: true,
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr(""),
					Protocol: protocolPtr(api.ProtocolTCP),
				}, {
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
				}},
			},
		},
		"long-port-name": {
			enableTopology: true,
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr(strings.Repeat("a", 63)),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
				}},
			},
		},
		"empty-ports-and-endpoints": {
			enableTopology: true,
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports:       []discovery.EndpointPort{},
				Endpoints:   []discovery.Endpoint{},
			},
		},
		"max-endpoints": {
			enableTopology: true,
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports:       generatePorts(1),
				Endpoints:   generateEndpoints(maxEndpoints),
			},
		},
		"max-ports": {
			enableTopology: true,
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports:       generatePorts(maxPorts),
			},
		},
		"max-addresses": {
			enableTopology: true,
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(maxAddresses),
				}},
			},
		},
		"max-topology-keys": {
			enableTopology: true,
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Topology:  generateTopology(maxTopologyLabels),
				}},
			},
		},

		// expected failures
		"duplicate-port-name": {
			enableTopology: true,
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
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
			enableTopology: true,
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("aCapital"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{},
			},
		},
		"bad-port-name-chars": {
			enableTopology: true,
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("almost_valid"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{},
			},
		},
		"bad-port-name-length": {
			enableTopology: true,
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr(strings.Repeat("a", 64)),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{},
			},
		},
		"invalid-port-protocol": {
			enableTopology: true,
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.Protocol("foo")),
				}},
			},
		},
		"too-many-ports": {
			enableTopology: true,
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports:       generatePorts(maxPorts + 1),
			},
		},
		"too-many-endpoints": {
			enableTopology: true,
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports:       generatePorts(1),
				Endpoints:   generateEndpoints(maxEndpoints + 1),
			},
		},
		"no-endpoint-addresses": {
			enableTopology: true,
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(0),
				}},
			},
		},
		"too-many-addresses": {
			enableTopology: true,
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(maxAddresses + 1),
				}},
			},
		},
		"bad-topology-key": {
			enableTopology: true,
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Topology:  map[string]string{"--INVALID": "example"},
				}},
			},
		},
		"topology-disabled": {
			enableTopology: false,
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Topology:  generateTopology(1),
				}},
			},
		},
		"too-many-topology-keys": {
			enableTopology: true,
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Topology:  generateTopology(maxTopologyLabels + 1),
				}},
			},
		},
		"bad-hostname": {
			enableTopology: true,
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hostname:  utilpointer.StringPtr("--INVALID"),
				}},
			},
		},
		"bad-meta": {
			enableTopology: true,
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "*&^",
					Namespace: "foo",
				},
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hostname:  utilpointer.StringPtr("valid-123"),
				}},
			},
		},
		"bad-ip": {
			enableTopology: true,
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: []string{"123.456.789.012"},
					Hostname:  utilpointer.StringPtr("valid-123"),
				}},
			},
		},
		"bad-ipv4": {
			enableTopology: true,
			expectedErrors: 2,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: []string{"123.456.789.012", "2001:4860:4860::8888"},
					Hostname:  utilpointer.StringPtr("valid-123"),
				}},
			},
		},
		"bad-ipv6": {
			enableTopology: true,
			expectedErrors: 2,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv6,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: []string{"123.456.789.012", "2001:4860:4860:defg"},
					Hostname:  utilpointer.StringPtr("valid-123"),
				}},
			},
		},
		"bad-fqdns": {
			enableTopology: true,
			expectedErrors: 4,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeFQDN,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: []string{"foo.*", "FOO.example.com", "underscores_are_bad.example.com", "*.example.com"},
					Hostname:  utilpointer.StringPtr("valid-123"),
				}},
			},
		},
		"bad-app-protocol": {
			enableTopology: true,
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:        utilpointer.StringPtr("http"),
					Protocol:    protocolPtr(api.ProtocolTCP),
					AppProtocol: utilpointer.StringPtr("--"),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hostname:  utilpointer.StringPtr("valid-123"),
				}},
			},
		},
		"empty-everything": {
			enableTopology: true,
			expectedErrors: 3,
			endpointSlice:  &discovery.EndpointSlice{},
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EndpointSliceTopology, testCase.enableTopology)()

			errs := ValidateEndpointSlice(testCase.endpointSlice)
			if len(errs) != testCase.expectedErrors {
				t.Errorf("Expected %d errors, got %d errors: %v", testCase.expectedErrors, len(errs), errs)
			}
		})
	}
}

func TestValidateEndpointSliceCreate(t *testing.T) {
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
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hostname:  utilpointer.StringPtr("valid-123"),
				}},
			},
		},

		// expected failures
		"deprecated-address-type": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressType("IP"),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
				}},
			},
		},
		"bad-address-type": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressType("other"),
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr("http"),
					Protocol: protocolPtr(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
				}},
			},
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			errs := ValidateEndpointSliceCreate(testCase.endpointSlice)
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
				AddressType: discovery.AddressTypeIPv6,
			},
			oldEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv6,
			},
			expectedErrors: 0,
		},
		"deprecated address type": {
			expectedErrors: 1,
			newEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressType("IP"),
			},
			oldEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressType("IP"),
			},
		},
		"valid and identical slices with different address types": {
			newEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
			},
			oldEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressType("other"),
			},
			expectedErrors: 1,
		},
		"invalid slices with valid address types": {
			newEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     utilpointer.StringPtr(""),
					Protocol: protocolPtr(api.Protocol("invalid")),
				}},
			},
			oldEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
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

func generateIPAddresses(n int) []string {
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
