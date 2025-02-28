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

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/discovery"
	"k8s.io/utils/ptr"
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
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hostname:  ptr.To("valid-123"),
				}},
			},
		},
		"good-ipv6": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv6,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: []string{"a00:100::4"},
					Hostname:  ptr.To("valid-123"),
				}},
			},
		},
		"good-fqdns": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeFQDN,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: []string{"foo.example.com", "example.com", "example.com.", "hyphens-are-good.example.com"},
					Hostname:  ptr.To("valid-123"),
				}},
			},
		},
		"all-protocols": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("tcp"),
					Protocol: ptr.To(api.ProtocolTCP),
				}, {
					Name:     ptr.To("udp"),
					Protocol: ptr.To(api.ProtocolUDP),
				}, {
					Name:     ptr.To("sctp"),
					Protocol: ptr.To(api.ProtocolSCTP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hostname:  ptr.To("valid-123"),
				}},
			},
		},
		"app-protocols": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:        ptr.To("one"),
					Protocol:    ptr.To(api.ProtocolTCP),
					AppProtocol: ptr.To("HTTP"),
				}, {
					Name:        ptr.To("two"),
					Protocol:    ptr.To(api.ProtocolTCP),
					AppProtocol: ptr.To("https"),
				}, {
					Name:        ptr.To("three"),
					Protocol:    ptr.To(api.ProtocolTCP),
					AppProtocol: ptr.To("my-protocol"),
				}, {
					Name:        ptr.To("four"),
					Protocol:    ptr.To(api.ProtocolTCP),
					AppProtocol: ptr.To("example.com/custom-protocol"),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hostname:  ptr.To("valid-123"),
				}},
			},
		},
		"empty-port-name": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To(""),
					Protocol: ptr.To(api.ProtocolTCP),
				}, {
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
				}},
			},
		},
		"long-port-name": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To(strings.Repeat("a", 63)),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
				}},
			},
		},
		"empty-ports-and-endpoints": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports:       []discovery.EndpointPort{},
				Endpoints:   []discovery.Endpoint{},
			},
		},
		"max-endpoints": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports:       generatePorts(1),
				Endpoints:   generateEndpoints(maxEndpoints),
			},
		},
		"max-ports": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports:       generatePorts(maxPorts),
			},
		},
		"max-addresses": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(maxAddresses),
				}},
			},
		},
		"max-topology-keys": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses:          generateIPAddresses(1),
					DeprecatedTopology: generateTopology(maxTopologyLabels),
				}},
			},
		},
		"valid-hints": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hints: &discovery.EndpointHints{
						ForZones: []discovery.ForZone{{Name: "zone-a"}},
					},
				}},
			},
		},

		// expected failures
		"duplicate-port-name": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To(""),
					Protocol: ptr.To(api.ProtocolTCP),
				}, {
					Name:     ptr.To(""),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{},
			},
		},
		"bad-port-name-caps": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("aCapital"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{},
			},
		},
		"bad-port-name-chars": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("almost_valid"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{},
			},
		},
		"bad-port-name-length": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To(strings.Repeat("a", 64)),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{},
			},
		},
		"invalid-port-protocol": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.Protocol("foo")),
				}},
			},
		},
		"too-many-ports": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports:       generatePorts(maxPorts + 1),
			},
		},
		"too-many-endpoints": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports:       generatePorts(1),
				Endpoints:   generateEndpoints(maxEndpoints + 1),
			},
		},
		"no-endpoint-addresses": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(0),
				}},
			},
		},
		"too-many-addresses": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(maxAddresses + 1),
				}},
			},
		},
		"bad-topology-key": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses:          generateIPAddresses(1),
					DeprecatedTopology: map[string]string{"--INVALID": "example"},
				}},
			},
		},
		"too-many-topology-keys": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses:          generateIPAddresses(1),
					DeprecatedTopology: generateTopology(maxTopologyLabels + 1),
				}},
			},
		},
		"bad-hostname": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hostname:  ptr.To("--INVALID"),
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
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hostname:  ptr.To("valid-123"),
				}},
			},
		},
		"bad-ip": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: []string{"123.456.789.012"},
					Hostname:  ptr.To("valid-123"),
				}},
			},
		},
		"bad-ipv4": {
			expectedErrors: 2,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: []string{"123.456.789.012", "2001:4860:4860::8888"},
					Hostname:  ptr.To("valid-123"),
				}},
			},
		},
		"bad-ipv6": {
			expectedErrors: 2,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv6,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: []string{"123.456.789.012", "2001:4860:4860:defg"},
					Hostname:  ptr.To("valid-123"),
				}},
			},
		},
		"bad-fqdns": {
			expectedErrors: 4,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeFQDN,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: []string{"foo.*", "FOO.example.com", "underscores_are_bad.example.com", "*.example.com"},
					Hostname:  ptr.To("valid-123"),
				}},
			},
		},
		"bad-app-protocol": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:        ptr.To("http"),
					Protocol:    ptr.To(api.ProtocolTCP),
					AppProtocol: ptr.To("--"),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hostname:  ptr.To("valid-123"),
				}},
			},
		},
		"invalid-hints": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hints: &discovery.EndpointHints{
						ForZones: []discovery.ForZone{{Name: "inv@lid"}},
					},
				}},
			},
		},
		"overlapping-hints": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hints: &discovery.EndpointHints{
						ForZones: []discovery.ForZone{
							{Name: "zone-a"},
							{Name: "zone-b"},
							{Name: "zone-a"},
						},
					},
				}},
			},
		},
		"too-many-hints": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hints: &discovery.EndpointHints{
						ForZones: []discovery.ForZone{
							{Name: "zone-a"},
							{Name: "zone-b"},
							{Name: "zone-c"},
							{Name: "zone-d"},
							{Name: "zone-e"},
							{Name: "zone-f"},
							{Name: "zone-g"},
							{Name: "zone-h"},
							{Name: "zone-i"},
						},
					},
				}},
			},
		},
		"empty-everything": {
			expectedErrors: 3,
			endpointSlice:  &discovery.EndpointSlice{},
		},
		"zone-key-topology": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses:          generateIPAddresses(1),
					DeprecatedTopology: map[string]string{corev1.LabelTopologyZone: "zone1"},
				}},
			},
		},
		"special-ipv4": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: []string{"127.0.0.1"},
					Hostname:  ptr.To("valid-123"),
				}},
			},
		},
		"special-ipv6": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv6,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: []string{"fe80::9656:d028:8652:66b6"},
					Hostname:  ptr.To("valid-123"),
				}},
			},
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

func TestValidateEndpointSliceCreate(t *testing.T) {
	standardMeta := metav1.ObjectMeta{
		Name:      "hello",
		Namespace: "world",
	}

	testCases := map[string]struct {
		expectedErrors      int
		endpointSlice       *discovery.EndpointSlice
		nodeNameGateEnabled bool
	}{
		"good-slice": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hostname:  ptr.To("valid-123"),
				}},
			},
		},
		"good-slice-node-name": {
			expectedErrors: 0,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hostname:  ptr.To("valid-123"),
					NodeName:  ptr.To("valid-node-name"),
				}},
			},
		},

		// expected failures
		"bad-node-name": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
				}},
				Endpoints: []discovery.Endpoint{{
					Addresses: generateIPAddresses(1),
					Hostname:  ptr.To("valid-123"),
					NodeName:  ptr.To("INvalid-node-name"),
				}},
			},
		},
		"deprecated-address-type": {
			expectedErrors: 1,
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressType("IP"),
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
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
					Name:     ptr.To("http"),
					Protocol: ptr.To(api.ProtocolTCP),
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
		expectedErrors      int
		nodeNameGateEnabled bool
		oldEndpointSlice    *discovery.EndpointSlice
		newEndpointSlice    *discovery.EndpointSlice
	}{
		"valid and identical slices": {
			oldEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv6,
			},
			newEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv6,
			},
			expectedErrors: 0,
		},

		// expected errors
		"invalid node name set": {
			oldEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Endpoints: []discovery.Endpoint{{
					Addresses: []string{"10.1.2.3"},
				}},
			},
			newEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Endpoints: []discovery.Endpoint{{
					Addresses: []string{"10.1.2.3"},
					NodeName:  ptr.To("INVALID foo"),
				}},
			},
			expectedErrors: 1,
		},

		"deprecated address type": {
			expectedErrors: 1,
			oldEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressType("IP"),
			},
			newEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressType("IP"),
			},
		},
		"valid and identical slices with different address types": {
			oldEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressType("other"),
			},
			newEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
			},
			expectedErrors: 1,
		},
		"invalid slices with valid address types": {
			oldEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
			},
			newEndpointSlice: &discovery.EndpointSlice{
				ObjectMeta:  standardMeta,
				AddressType: discovery.AddressTypeIPv4,
				Ports: []discovery.EndpointPort{{
					Name:     ptr.To(""),
					Protocol: ptr.To(api.Protocol("invalid")),
				}},
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

func generatePorts(n int) []discovery.EndpointPort {
	ports := []discovery.EndpointPort{}
	for i := 0; i < n; i++ {
		ports = append(ports, discovery.EndpointPort{
			Name:     ptr.To(fmt.Sprintf("http-%d", i)),
			Protocol: ptr.To(api.ProtocolTCP),
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
