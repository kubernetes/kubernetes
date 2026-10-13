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

package helpers

import (
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
)

const (
	testKubeletHostname = "hostname"
)

func TestAddToNodeAddresses(t *testing.T) {
	testCases := []struct {
		name     string
		existing []v1.NodeAddress
		toAdd    []v1.NodeAddress
		expected []v1.NodeAddress
	}{
		{
			name:     "add no addresses to empty node addresses",
			existing: []v1.NodeAddress{},
			toAdd:    []v1.NodeAddress{},
			expected: []v1.NodeAddress{},
		},
		{
			name:     "add new node addresses to empty existing addresses",
			existing: []v1.NodeAddress{},
			toAdd: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeHostName, Address: "localhost"},
			},
			expected: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeHostName, Address: "localhost"},
			},
		},
		{
			name:     "add a duplicate node address",
			existing: []v1.NodeAddress{},
			toAdd: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
			},
			expected: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
			},
		},
		{
			name: "add 1 new and 1 duplicate address",
			existing: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
			},
			toAdd: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeHostName, Address: "localhost"},
			},
			expected: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeHostName, Address: "localhost"},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			AddToNodeAddresses(&tc.existing, tc.toAdd...)
			if !apiequality.Semantic.DeepEqual(tc.expected, tc.existing) {
				t.Errorf("expected: %v, got: %v", tc.expected, tc.existing)
			}
		})
	}
}

func TestGetNodeAddressesFromNodeIP(t *testing.T) {
	cases := []struct {
		name              string
		nodeIP            string
		nodeAddresses     []v1.NodeAddress
		expectedAddresses []v1.NodeAddress
		shouldError       bool
	}{
		{
			name:   "A single InternalIP",
			nodeIP: "10.1.1.1",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			expectedAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			shouldError: false,
		},
		{
			name:   "NodeIP is external",
			nodeIP: "55.55.55.55",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeExternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			expectedAddresses: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			shouldError: false,
		},
		{
			// Accommodating #45201 and #49202
			name:   "InternalIP and ExternalIP are the same",
			nodeIP: "55.55.55.55",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "44.44.44.44"},
				{Type: v1.NodeExternalIP, Address: "44.44.44.44"},
				{Type: v1.NodeInternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeExternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			expectedAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeExternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			shouldError: false,
		},
		{
			name:   "An Internal/ExternalIP, an Internal/ExternalDNS",
			nodeIP: "10.1.1.1",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeExternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeInternalDNS, Address: "ip-10-1-1-1.us-west-2.compute.internal"},
				{Type: v1.NodeExternalDNS, Address: "ec2-55-55-55-55.us-west-2.compute.amazonaws.com"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			expectedAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeExternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeInternalDNS, Address: "ip-10-1-1-1.us-west-2.compute.internal"},
				{Type: v1.NodeExternalDNS, Address: "ec2-55-55-55-55.us-west-2.compute.amazonaws.com"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			shouldError: false,
		},
		{
			name:   "An Internal with multiple internal IPs",
			nodeIP: "10.1.1.1",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "10.2.2.2"},
				{Type: v1.NodeInternalIP, Address: "10.3.3.3"},
				{Type: v1.NodeExternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			expectedAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeExternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			shouldError: false,
		},
		{
			name:   "An InternalIP that isn't valid: should error",
			nodeIP: "10.2.2.2",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeExternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			expectedAddresses: nil,
			shouldError:       true,
		},
		{
			name:   "Dual-stack cloud, with nodeIP, different IPv6 formats",
			nodeIP: "2600:1f14:1d4:d101::ba3d",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "2600:1f14:1d4:d101:0:0:0:ba3d"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			expectedAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "2600:1f14:1d4:d101:0:0:0:ba3d"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			shouldError: false,
		},
		{
			name:   "Single-stack cloud, dual-stack request",
			nodeIP: "10.1.1.1,fc01:1234::5678",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			shouldError: true,
		},
		{
			name:   "Dual-stack cloud, IPv4 first, IPv4-primary request",
			nodeIP: "10.1.1.1,fc01:1234::5678",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "fc01:1234::5678"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			expectedAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "fc01:1234::5678"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			shouldError: false,
		},
		{
			name:   "Dual-stack cloud, IPv6 first, IPv4-primary request",
			nodeIP: "10.1.1.1,fc01:1234::5678",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "fc01:1234::5678"},
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			expectedAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "fc01:1234::5678"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			shouldError: false,
		},
		{
			name:   "Dual-stack cloud, dual-stack request, multiple IPs",
			nodeIP: "10.1.1.1,fc01:1234::5678",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "10.1.1.2"},
				{Type: v1.NodeInternalIP, Address: "fc01:1234::1234"},
				{Type: v1.NodeInternalIP, Address: "fc01:1234::5678"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			// additional IPs of the same type are removed, as in the
			// single-stack case.
			expectedAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "fc01:1234::5678"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			shouldError: false,
		},
		{
			name:   "Dual-stack cloud, dual-stack request, extra ExternalIP",
			nodeIP: "10.1.1.1,fc01:1234::5678",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeExternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeInternalIP, Address: "fc01:1234::1234"},
				{Type: v1.NodeInternalIP, Address: "fc01:1234::5678"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			// The ExternalIP is preserved, since no ExternalIP was matched
			// by --node-ip.
			expectedAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "fc01:1234::5678"},
				{Type: v1.NodeExternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			shouldError: false,
		},
		{
			name:   "Dual-stack cloud, dual-stack request, multiple ExternalIPs",
			nodeIP: "fc01:1234::5678,10.1.1.1",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeExternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeInternalIP, Address: "fc01:1234::5678"},
				{Type: v1.NodeExternalIP, Address: "2001:db1::1"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			// The ExternalIPs are preserved, since no ExternalIP was matched
			// by --node-ip.
			expectedAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "fc01:1234::5678"},
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeExternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeExternalIP, Address: "2001:db1::1"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			shouldError: false,
		},
		{
			name:   "Dual-stack cloud, dual-stack request, mixed InternalIP/ExternalIP match",
			nodeIP: "55.55.55.55,fc01:1234::5678",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeExternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeInternalIP, Address: "fc01:1234::5678"},
				{Type: v1.NodeExternalIP, Address: "2001:db1::1"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			// Since the IPv4 --node-ip value matched an ExternalIP, that
			// filters out the IPv6 ExternalIP. Since the IPv6 --node-ip value
			// matched in InternalIP, that filters out the IPv4 InternalIP
			// value.
			expectedAddresses: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeInternalIP, Address: "fc01:1234::5678"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			shouldError: false,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetNodeAddressesFromNodeIP(tt.nodeIP, tt.nodeAddresses)
			if (err != nil) != tt.shouldError {
				t.Errorf("GetNodeAddressesFromNodeIP() error = %v, wantErr %v", err, tt.shouldError)
				return
			}
			if !reflect.DeepEqual(got, tt.expectedAddresses) {
				t.Errorf("GetNodeAddressesFromNodeIP() = %v, want %v", got, tt.expectedAddresses)
			}
		})
	}
}
