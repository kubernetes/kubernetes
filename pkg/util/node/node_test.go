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
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

func Test_GetZoneKey(t *testing.T) {
	tests := []struct {
		name string
		node *v1.Node
		zone string
	}{
		{
			name: "has no zone or region keys",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{},
				},
			},
			zone: "",
		},
		{
			name: "has beta zone and region keys",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						v1.LabelZoneFailureDomain: "zone1",
						v1.LabelZoneRegion:        "region1",
					},
				},
			},
			zone: "region1:\x00:zone1",
		},
		{
			name: "has GA zone and region keys",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						v1.LabelZoneFailureDomainStable: "zone1",
						v1.LabelZoneRegionStable:        "region1",
					},
				},
			},
			zone: "region1:\x00:zone1",
		},
		{
			name: "has both beta and GA zone and region keys",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						v1.LabelZoneFailureDomainStable: "zone1",
						v1.LabelZoneRegionStable:        "region1",
						v1.LabelZoneFailureDomain:       "zone1",
						v1.LabelZoneRegion:              "region1",
					},
				},
			},
			zone: "region1:\x00:zone1",
		},
		{
			name: "has both beta and GA zone and region keys, beta labels take precedent",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						v1.LabelZoneFailureDomainStable: "zone1",
						v1.LabelZoneRegionStable:        "region1",
						v1.LabelZoneFailureDomain:       "zone2",
						v1.LabelZoneRegion:              "region2",
					},
				},
			},
			zone: "region2:\x00:zone2",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			zone := GetZoneKey(test.node)
			if zone != test.zone {
				t.Logf("actual zone key: %q", zone)
				t.Logf("expected zone key: %q", test.zone)
				t.Errorf("unexpected zone key")
			}
		})
	}
}

func TestGetNodeHostIP(t *testing.T) {
	testcases := map[string]struct {
		Addresses []v1.NodeAddress

		ExpectErr string
		ExpectIP  string
	}{
		"no addresses": {
			ExpectErr: "host IP unknown; known addresses: []",
		},
		"both internal and external IPs": {
			Addresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "1.2.3.4"},
				{Type: v1.NodeExternalIP, Address: "5.6.7.8"},
			},
			ExpectIP: "1.2.3.4",
		},
		"only external IP": {
			Addresses: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "5.6.7.8"},
			},
			ExpectIP: "5.6.7.8",
		},
		"multiple internal IPs": {
			Addresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "1.2.3.4"},
				{Type: v1.NodeInternalIP, Address: "2.3.4.5"},
			},
			ExpectIP: "1.2.3.4",
		},
		"multiple external IPs": {
			Addresses: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "5.6.7.8"},
				{Type: v1.NodeExternalIP, Address: "6.7.8.9"},
			},
			ExpectIP: "5.6.7.8",
		},
		"IPv6 IPs": {
			Addresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "2607:f0d0:1002:51::4"},
			},
			ExpectIP: "2607:f0d0:1002:51::4",
		},
		"both valid and invalid internal IPs": {
			Addresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "invalid"},
				{Type: v1.NodeInternalIP, Address: "2.3.4.5"},
			},
			ExpectIP: "2.3.4.5",
		},
		"both valid and invalid external IPs": {
			Addresses: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "invalid"},
				{Type: v1.NodeExternalIP, Address: "6.7.8.9"},
			},
			ExpectIP: "6.7.8.9",
		},
		"all invalid IPs": {
			Addresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "invalid"},
				{Type: v1.NodeExternalIP, Address: "invalid"},
			},
			ExpectErr: "host IP unknown; known addresses: [{InternalIP invalid} {ExternalIP invalid}]",
		},
	}

	for name, test := range testcases {
		node := &v1.Node{
			Status: v1.NodeStatus{Addresses: test.Addresses},
		}
		hostIP, err := GetNodeHostIP(node)

		if err != nil {
			if err.Error() != test.ExpectErr {
				t.Errorf("test %s fails, expected err=%q but got %q", name, test.ExpectErr, err.Error())
				return
			}
		} else {
			if hostIP.String() != test.ExpectIP {
				t.Errorf("test %s fails, expected hostIP=%s but got %s", name, test.ExpectIP, hostIP.String())
				return
			}
		}
	}
}
