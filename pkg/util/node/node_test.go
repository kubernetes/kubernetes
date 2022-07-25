/*
Copyright 2016 The Kubernetes Authors.

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

package node

import (
	"net"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	netutils "k8s.io/utils/net"
)

func TestGetPreferredAddress(t *testing.T) {
	testcases := map[string]struct {
		Labels      map[string]string
		Addresses   []v1.NodeAddress
		Preferences []v1.NodeAddressType

		ExpectErr     string
		ExpectAddress string
	}{
		"no addresses": {
			ExpectErr: "no preferred addresses found; known addresses: []",
		},
		"missing address": {
			Addresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "1.2.3.4"},
			},
			Preferences: []v1.NodeAddressType{v1.NodeHostName},
			ExpectErr:   "no preferred addresses found; known addresses: [{InternalIP 1.2.3.4}]",
		},
		"found address": {
			Addresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "1.2.3.4"},
				{Type: v1.NodeExternalIP, Address: "1.2.3.5"},
				{Type: v1.NodeExternalIP, Address: "1.2.3.7"},
			},
			Preferences:   []v1.NodeAddressType{v1.NodeHostName, v1.NodeExternalIP},
			ExpectAddress: "1.2.3.5",
		},
		"found hostname address": {
			Labels: map[string]string{v1.LabelHostname: "label-hostname"},
			Addresses: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.2.3.5"},
				{Type: v1.NodeHostName, Address: "status-hostname"},
			},
			Preferences:   []v1.NodeAddressType{v1.NodeHostName, v1.NodeExternalIP},
			ExpectAddress: "status-hostname",
		},
		"label address ignored": {
			Labels: map[string]string{v1.LabelHostname: "label-hostname"},
			Addresses: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.2.3.5"},
			},
			Preferences:   []v1.NodeAddressType{v1.NodeHostName, v1.NodeExternalIP},
			ExpectAddress: "1.2.3.5",
		},
	}

	for k, tc := range testcases {
		node := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{Labels: tc.Labels},
			Status:     v1.NodeStatus{Addresses: tc.Addresses},
		}
		address, err := GetPreferredNodeAddress(node, tc.Preferences)
		errString := ""
		if err != nil {
			errString = err.Error()
		}
		if errString != tc.ExpectErr {
			t.Errorf("%s: expected err=%q, got %q", k, tc.ExpectErr, errString)
		}
		if address != tc.ExpectAddress {
			t.Errorf("%s: expected address=%q, got %q", k, tc.ExpectAddress, address)
		}
	}
}

func TestGetNodeHostIPs(t *testing.T) {
	testcases := []struct {
		name      string
		addresses []v1.NodeAddress

		expectIPs []net.IP
	}{
		{
			name:      "no addresses",
			expectIPs: nil,
		},
		{
			name: "no InternalIP/ExternalIP",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeHostName, Address: "example.com"},
			},
			expectIPs: nil,
		},
		{
			name: "IPv4-only, simple",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "1.2.3.4"},
				{Type: v1.NodeExternalIP, Address: "4.3.2.1"},
				{Type: v1.NodeExternalIP, Address: "4.3.2.2"},
			},
			expectIPs: []net.IP{netutils.ParseIPSloppy("1.2.3.4")},
		},
		{
			name: "IPv4-only, external-first",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "4.3.2.1"},
				{Type: v1.NodeExternalIP, Address: "4.3.2.2"},
				{Type: v1.NodeInternalIP, Address: "1.2.3.4"},
			},
			expectIPs: []net.IP{netutils.ParseIPSloppy("1.2.3.4")},
		},
		{
			name: "IPv4-only, no internal",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "4.3.2.1"},
				{Type: v1.NodeExternalIP, Address: "4.3.2.2"},
			},
			expectIPs: []net.IP{netutils.ParseIPSloppy("4.3.2.1")},
		},
		{
			name: "dual-stack node",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "1.2.3.4"},
				{Type: v1.NodeExternalIP, Address: "4.3.2.1"},
				{Type: v1.NodeExternalIP, Address: "4.3.2.2"},
				{Type: v1.NodeInternalIP, Address: "a:b::c:d"},
				{Type: v1.NodeExternalIP, Address: "d:c::b:a"},
			},
			expectIPs: []net.IP{netutils.ParseIPSloppy("1.2.3.4"), netutils.ParseIPSloppy("a:b::c:d")},
		},
		{
			name: "dual-stack node, different order",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "1.2.3.4"},
				{Type: v1.NodeInternalIP, Address: "a:b::c:d"},
				{Type: v1.NodeExternalIP, Address: "4.3.2.1"},
				{Type: v1.NodeExternalIP, Address: "4.3.2.2"},
				{Type: v1.NodeExternalIP, Address: "d:c::b:a"},
			},
			expectIPs: []net.IP{netutils.ParseIPSloppy("1.2.3.4"), netutils.ParseIPSloppy("a:b::c:d")},
		},
		{
			name: "dual-stack node, IPv6-first, no internal IPv4, dual-stack cluster",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "a:b::c:d"},
				{Type: v1.NodeExternalIP, Address: "d:c::b:a"},
				{Type: v1.NodeExternalIP, Address: "4.3.2.1"},
				{Type: v1.NodeExternalIP, Address: "4.3.2.2"},
			},
			expectIPs: []net.IP{netutils.ParseIPSloppy("a:b::c:d"), netutils.ParseIPSloppy("4.3.2.1")},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			node := &v1.Node{
				Status: v1.NodeStatus{Addresses: tc.addresses},
			}
			nodeIPs, err := GetNodeHostIPs(node)
			nodeIP, err2 := GetNodeHostIP(node)

			if (err == nil && err2 != nil) || (err != nil && err2 == nil) {
				t.Errorf("GetNodeHostIPs() returned error=%q but GetNodeHostIP() returned error=%q", err, err2)
			}
			if err != nil {
				if tc.expectIPs != nil {
					t.Errorf("expected %v, got error (%v)", tc.expectIPs, err)
				}
			} else if tc.expectIPs == nil {
				t.Errorf("expected error, got %v", nodeIPs)
			} else if !reflect.DeepEqual(nodeIPs, tc.expectIPs) {
				t.Errorf("expected %v, got %v", tc.expectIPs, nodeIPs)
			} else if !nodeIP.Equal(nodeIPs[0]) {
				t.Errorf("GetNodeHostIP did not return same primary (%s) as GetNodeHostIPs (%s)", nodeIP.String(), nodeIPs[0].String())
			}
		})
	}
}

func TestGetHostname(t *testing.T) {
	testCases := []struct {
		hostName         string
		expectedHostName string
		expectError      bool
	}{
		{
			hostName:    "   ",
			expectError: true,
		},
		{
			hostName:         " abc  ",
			expectedHostName: "abc",
			expectError:      false,
		},
	}

	for idx, test := range testCases {
		hostName, err := GetHostname(test.hostName)
		if err != nil && !test.expectError {
			t.Errorf("[%d]: unexpected error: %s", idx, err)
		}
		if err == nil && test.expectError {
			t.Errorf("[%d]: expected error, got none", idx)
		}
		if test.expectedHostName != hostName {
			t.Errorf("[%d]: expected output %q, got %q", idx, test.expectedHostName, hostName)
		}

	}
}

func TestIsNodeReady(t *testing.T) {
	testCases := []struct {
		name   string
		Node   *v1.Node
		expect bool
	}{
		{
			name: "case that returns true",
			Node: &v1.Node{
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionTrue,
						},
					},
				},
			},
			expect: true,
		},
		{
			name: "case that returns false",
			Node: &v1.Node{
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionFalse,
						},
					},
				},
			},
			expect: false,
		},
		{
			name: "case that returns false",
			Node: &v1.Node{
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:   v1.NodeMemoryPressure,
							Status: v1.ConditionFalse,
						},
					},
				},
			},
			expect: false,
		},
	}
	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			result := IsNodeReady(test.Node)
			assert.Equal(t, test.expect, result)
		})
	}
}
