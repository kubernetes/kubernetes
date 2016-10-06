package node

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

func TestGetPreferredNodeAddress(t *testing.T) {
	testcases := map[string]struct {
		addresses       []api.NodeAddress
		preference      []api.NodeAddressType
		expectedAddress string
		expectedError   bool
	}{
		"empty preferred addresses": {
			addresses:       []api.NodeAddress{{Type: api.NodeInternalIP, Address: "1.2.3.4"}},
			preference:      []api.NodeAddressType{},
			expectedAddress: "",
			expectedError:   true,
		},
		"empty node addresses": {
			addresses:       []api.NodeAddress{},
			preference:      []api.NodeAddressType{api.NodeInternalIP},
			expectedAddress: "",
			expectedError:   true,
		},
		"exact match": {
			addresses:       []api.NodeAddress{{Type: api.NodeInternalIP, Address: "1.2.3.4"}},
			preference:      []api.NodeAddressType{api.NodeInternalIP},
			expectedAddress: "1.2.3.4",
			expectedError:   false,
		},
		"fallback": {
			addresses:       []api.NodeAddress{{Type: api.NodeInternalIP, Address: "1.2.3.4"}},
			preference:      []api.NodeAddressType{api.NodeExternalIP, api.NodeInternalIP},
			expectedAddress: "1.2.3.4",
			expectedError:   false,
		},
		"select first": {
			addresses: []api.NodeAddress{
				{Type: api.NodeInternalIP, Address: "1.2.3.4"},
				{Type: api.NodeInternalIP, Address: "2.3.4.5"},
				{Type: api.NodeExternalIP, Address: "3.4.5.6"},
				{Type: api.NodeExternalIP, Address: "4.5.6.7"},
			},
			preference:      []api.NodeAddressType{api.NodeExternalIP, api.NodeInternalIP},
			expectedAddress: "3.4.5.6",
			expectedError:   false,
		},
	}

	for k, tc := range testcases {
		address, err := GetPreferredNodeAddress(&api.Node{Status: api.NodeStatus{Addresses: tc.addresses}}, tc.preference)
		if tc.expectedError != (err != nil) {
			t.Errorf("%s: expected error %v, got %v", k, tc.expectedError, err)
		}
		if address != tc.expectedAddress {
			t.Errorf("%s: expected address %s, got %s", k, tc.expectedAddress, address)
		}
	}
}
