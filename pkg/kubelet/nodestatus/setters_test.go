/*
Copyright 2018 The Kubernetes Authors.

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

package nodestatus

import (
	"net"
	"sort"
	"testing"

	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"

	"github.com/stretchr/testify/assert"
)

const (
	testKubeletHostname = "127.0.0.1"
)

// TODO(mtaufen): below is ported from the old kubelet_node_status_test.go code, potentially add more test coverage for NodeAddress setter in future
func TestNodeAddress(t *testing.T) {
	cases := []struct {
		name              string
		nodeIP            net.IP
		nodeAddresses     []v1.NodeAddress
		expectedAddresses []v1.NodeAddress
		shouldError       bool
	}{
		{
			name:   "A single InternalIP",
			nodeIP: net.ParseIP("10.1.1.1"),
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
			nodeIP: net.ParseIP("55.55.55.55"),
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
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
			// Accommodating #45201 and #49202
			name:   "InternalIP and ExternalIP are the same",
			nodeIP: net.ParseIP("55.55.55.55"),
			nodeAddresses: []v1.NodeAddress{
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
			nodeIP: net.ParseIP("10.1.1.1"),
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
			nodeIP: net.ParseIP("10.1.1.1"),
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
			nodeIP: net.ParseIP("10.2.2.2"),
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeExternalIP, Address: "55.55.55.55"},
				{Type: v1.NodeHostName, Address: testKubeletHostname},
			},
			expectedAddresses: nil,
			shouldError:       true,
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			// testCase setup
			existingNode := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: testKubeletHostname, Annotations: make(map[string]string)},
				Spec:       v1.NodeSpec{},
			}

			nodeIP := testCase.nodeIP
			nodeIPValidator := func(nodeIP net.IP) error {
				return nil
			}
			hostname := testKubeletHostname
			externalCloudProvider := false
			cloud := &fakecloud.FakeCloud{
				Addresses: testCase.nodeAddresses,
				Err:       nil,
			}
			nodeAddressesFunc := func() ([]v1.NodeAddress, error) {
				return testCase.nodeAddresses, nil
			}

			// construct setter
			setter := NodeAddress(nodeIP,
				nodeIPValidator,
				hostname,
				externalCloudProvider,
				cloud,
				nodeAddressesFunc)

			// call setter on existing node
			err := setter(existingNode)
			if err != nil && !testCase.shouldError {
				t.Fatalf("unexpected error: %v", err)
			} else if err != nil && testCase.shouldError {
				// expected an error, and got one, so just return early here
				return
			}

			// Sort both sets for consistent equality
			sortNodeAddresses(testCase.expectedAddresses)
			sortNodeAddresses(existingNode.Status.Addresses)

			assert.True(t, apiequality.Semantic.DeepEqual(testCase.expectedAddresses, existingNode.Status.Addresses),
				"Diff: %s", diff.ObjectDiff(testCase.expectedAddresses, existingNode.Status.Addresses))
		})
	}
}

// Test Helpers:

// sortableNodeAddress is a type for sorting []v1.NodeAddress
type sortableNodeAddress []v1.NodeAddress

func (s sortableNodeAddress) Len() int { return len(s) }
func (s sortableNodeAddress) Less(i, j int) bool {
	return (string(s[i].Type) + s[i].Address) < (string(s[j].Type) + s[j].Address)
}
func (s sortableNodeAddress) Swap(i, j int) { s[j], s[i] = s[i], s[j] }

func sortNodeAddresses(addrs sortableNodeAddress) {
	sort.Sort(addrs)
}
