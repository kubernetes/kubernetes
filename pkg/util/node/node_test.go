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
	"context"
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
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

func Test_PatchNodeStatus(t *testing.T) {
	tests := []struct {
		name               string
		oldNode            *v1.Node
		newNode            *v1.Node
		expectedNode       *v1.Node
		expectedPatchBytes []byte
		expectedErr        error
	}{
		{
			name: "spec ignored in patch bytes",
			oldNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-node",
				},
				Spec: v1.NodeSpec{
					PodCIDR: "172.17.0.1/24",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionFalse,
							Reason: "KubeletNotReady",
						},
					},
				},
			},
			newNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-node",
				},
				Spec: v1.NodeSpec{
					PodCIDR: "172.17.0.2/24",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionTrue,
							Reason: "KubeletReady",
						},
					},
				},
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-node",
				},
				Spec: v1.NodeSpec{
					PodCIDR: "172.17.0.1/24",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionTrue,
							Reason: "KubeletReady",
						},
					},
				},
			},
			expectedPatchBytes: []byte(`{"status":{"$setElementOrder/conditions":[{"type":"Ready"}],"conditions":[{"reason":"KubeletReady","status":"True","type":"Ready"}]}}`),
			expectedErr:        nil,
		},
		{
			name: "patch without node addresses",
			oldNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-node",
				},
				Spec: v1.NodeSpec{
					PodCIDR: "172.17.0.1/24",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionTrue,
							Reason: "KubeletReady",
						},
					},
					Images: []v1.ContainerImage{
						{
							Names: []string{"foobar:latest"},
						},
					},
				},
			},
			newNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-node",
				},
				Spec: v1.NodeSpec{
					PodCIDR: "172.17.0.2/24",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionTrue,
							Reason: "KubeletReady",
						},
						{
							Type:   v1.NodeMemoryPressure,
							Status: v1.ConditionTrue,
							Reason: "KubeletHasSufficientMemory",
						},
					},
					Images: []v1.ContainerImage{
						{
							Names: []string{"foobar:v1"},
						},
					},
				},
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-node",
				},
				Spec: v1.NodeSpec{
					PodCIDR: "172.17.0.1/24",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionTrue,
							Reason: "KubeletReady",
						},
						{
							Type:   v1.NodeMemoryPressure,
							Status: v1.ConditionTrue,
							Reason: "KubeletHasSufficientMemory",
						},
					},
					Images: []v1.ContainerImage{
						{
							Names: []string{"foobar:v1"},
						},
					},
				},
			},
			expectedPatchBytes: []byte(`{"status":{"$setElementOrder/conditions":[{"type":"Ready"},{"type":"MemoryPressure"}],"conditions":[{"lastHeartbeatTime":null,"lastTransitionTime":null,"reason":"KubeletHasSufficientMemory","status":"True","type":"MemoryPressure"}],"images":[{"names":["foobar:v1"]}]}}`),
			expectedErr:        nil,
		},
		{
			name: "patch with node addresses using replace strategy instead of merge",
			oldNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-node",
				},
				Spec: v1.NodeSpec{
					PodCIDR: "172.17.0.1/24",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionTrue,
							Reason: "KubeletReady",
						},
					},
					Addresses: []v1.NodeAddress{
						{
							Type:    v1.NodeInternalIP,
							Address: "10.0.0.1",
						},
						{
							Type:    v1.NodeExternalIP,
							Address: "1.1.1.1",
						},
					},
				},
			},
			newNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-node",
				},
				Spec: v1.NodeSpec{
					PodCIDR: "172.17.0.2/24",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionTrue,
							Reason: "KubeletReady",
						},
					},
					Addresses: []v1.NodeAddress{
						{
							Type:    v1.NodeInternalIP,
							Address: "10.0.0.1",
						},
						{
							Type:    v1.NodeInternalIP,
							Address: "10.0.0.2",
						},
					},
				},
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-node",
				},
				Spec: v1.NodeSpec{
					PodCIDR: "172.17.0.1/24",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionTrue,
							Reason: "KubeletReady",
						},
					},
					Addresses: []v1.NodeAddress{
						{
							Type:    v1.NodeInternalIP,
							Address: "10.0.0.1",
						},
						{
							Type:    v1.NodeInternalIP,
							Address: "10.0.0.2",
						},
					},
				},
			},
			expectedPatchBytes: []byte(`{"status":{"addresses":[{"address":"10.0.0.1","type":"InternalIP"},{"address":"10.0.0.2","type":"InternalIP"},{"$patch":"replace"}]}}`),
			expectedErr:        nil,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			clientset := fake.NewSimpleClientset()
			_, err := clientset.CoreV1().Nodes().Create(context.TODO(), test.oldNode, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create initial node: %v", err)
			}
			node, patchBytes, err := PatchNodeStatus(clientset.CoreV1(), types.NodeName(test.oldNode.Name), test.oldNode, test.newNode)
			if !reflect.DeepEqual(node, test.expectedNode) {
				t.Logf("actual patched node: %v", node)
				t.Logf("expected patched node: %v", test.expectedNode)
				t.Error("unexpectd node")
			}

			if !reflect.DeepEqual(patchBytes, test.expectedPatchBytes) {
				t.Logf("actual patch bytes: %v", string(patchBytes))
				t.Logf("expected patch bytes: %v", string(test.expectedPatchBytes))
				t.Error("unexpected patch bytes")
			}

			if err != test.expectedErr {
				t.Logf("actual err: %v", err)
				t.Logf("expected err: %v", test.expectedErr)
				t.Error("unexpected err")
			}
		})
	}
}
