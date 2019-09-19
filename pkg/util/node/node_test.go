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

func TestGetNodeHostIP(t *testing.T) {
	testCases := []struct {
		description    string
		addresses      []v1.NodeAddress
		expectedHostIP net.IP
		expectError    bool
	}{
		{
			description: "empty node addresses",
			addresses:   []v1.NodeAddress{},
			expectError: true,
		},
		{
			description: "with external address only",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.2.3.4"},
			},
			expectedHostIP: net.ParseIP("1.2.3.4"),
			expectError:    false,
		},
		{
			description: "with external and internal address",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.2.3.4"},
				{Type: v1.NodeInternalIP, Address: "5.6.7.8"},
			},
			expectedHostIP: net.ParseIP("5.6.7.8"),
			expectError:    false,
		},
		{
			description: "with external and multiple internal addresses",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.2.3.4"},
				{Type: v1.NodeInternalIP, Address: "5.6.7.8"},
				{Type: v1.NodeInternalIP, Address: "3.4.5.6"},
			},
			expectedHostIP: net.ParseIP("5.6.7.8"),
			expectError:    false,
		},
		{
			description: "with external and internal ipv6 addresses",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "FE80:CD00:0000:0CDE:1257:0000:211E:729C"},
				{Type: v1.NodeInternalIP, Address: "FE80:CD00:0000:0CDE:1257:0000:211E:809C"},
				{Type: v1.NodeInternalIP, Address: "FE80:CD00:0000:0CDE:1257:0000:211E:729C"},
			},
			expectedHostIP: net.ParseIP("FE80:CD00:0000:0CDE:1257:0000:211E:809C"),
			expectError:    false,
		},
	}

	for idx, test := range testCases {
		node := &v1.Node{
			Status: v1.NodeStatus{Addresses: test.addresses},
		}
		hostIP, err := GetNodeHostIP(node)
		if err != nil && !test.expectError {
			t.Errorf("[%d]: unexpected error: %s", idx, err)
		}
		if err == nil && test.expectError {
			t.Errorf("[%d]: expected error, got none", idx)
		}
		if !test.expectedHostIP.Equal(hostIP) {
			t.Errorf("[%d]: for test case: %q expected output %q, got %q", idx, test.description, test.expectedHostIP, hostIP)
		}
	}
}
