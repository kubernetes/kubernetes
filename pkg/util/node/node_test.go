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
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
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
			Labels: map[string]string{kubeletapis.LabelHostname: "label-hostname"},
			Addresses: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.2.3.5"},
				{Type: v1.NodeHostName, Address: "status-hostname"},
			},
			Preferences:   []v1.NodeAddressType{v1.NodeHostName, v1.NodeExternalIP},
			ExpectAddress: "status-hostname",
		},
		"label address ignored": {
			Labels: map[string]string{kubeletapis.LabelHostname: "label-hostname"},
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
