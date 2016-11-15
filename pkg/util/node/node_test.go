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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

func TestGetPreferredAddress(t *testing.T) {
	testcases := map[string]struct {
		Labels      map[string]string
		Addresses   []api.NodeAddress
		Preferences []api.NodeAddressType

		ExpectErr     string
		ExpectAddress string
	}{
		"no addresses": {
			ExpectErr: "no preferred addresses found; known addresses: []",
		},
		"missing address": {
			Addresses: []api.NodeAddress{
				{Type: api.NodeInternalIP, Address: "1.2.3.4"},
			},
			Preferences: []api.NodeAddressType{api.NodeHostName},
			ExpectErr:   "no preferred addresses found; known addresses: [{InternalIP 1.2.3.4}]",
		},
		"found address": {
			Addresses: []api.NodeAddress{
				{Type: api.NodeInternalIP, Address: "1.2.3.4"},
				{Type: api.NodeExternalIP, Address: "1.2.3.5"},
				{Type: api.NodeExternalIP, Address: "1.2.3.7"},
			},
			Preferences:   []api.NodeAddressType{api.NodeHostName, api.NodeExternalIP},
			ExpectAddress: "1.2.3.5",
		},
		"found hostname address": {
			Labels: map[string]string{unversioned.LabelHostname: "label-hostname"},
			Addresses: []api.NodeAddress{
				{Type: api.NodeExternalIP, Address: "1.2.3.5"},
				{Type: api.NodeHostName, Address: "status-hostname"},
			},
			Preferences:   []api.NodeAddressType{api.NodeHostName, api.NodeExternalIP},
			ExpectAddress: "status-hostname",
		},
		"found label address": {
			Labels: map[string]string{unversioned.LabelHostname: "label-hostname"},
			Addresses: []api.NodeAddress{
				{Type: api.NodeExternalIP, Address: "1.2.3.5"},
			},
			Preferences:   []api.NodeAddressType{api.NodeHostName, api.NodeExternalIP},
			ExpectAddress: "label-hostname",
		},
	}

	for k, tc := range testcases {
		node := &api.Node{
			ObjectMeta: api.ObjectMeta{Labels: tc.Labels},
			Status:     api.NodeStatus{Addresses: tc.Addresses},
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
