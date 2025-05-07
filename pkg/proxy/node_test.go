/*
Copyright 2022 The Kubernetes Authors.

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

package proxy

import (
	"context"
	"net"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/utils/ktesting"
	netutils "k8s.io/utils/net"
	"k8s.io/utils/ptr"
)

const (
	testNodeName = "test-node"
)

func TestNodePodCIDRHandlerAdd(t *testing.T) {
	oldKlogOsExit := klog.OsExit
	defer func() {
		klog.OsExit = oldKlogOsExit
	}()
	klog.OsExit = customExit

	tests := []struct {
		name            string
		oldNodePodCIDRs []string
		newNodePodCIDRs []string
		expectPanic     bool
	}{
		{
			name: "both empty",
		},
		{
			name:            "initialized correctly",
			newNodePodCIDRs: []string{"192.168.1.0/24", "fd00:1:2:3::/64"},
		},
		{
			name:            "already initialized and same node",
			oldNodePodCIDRs: []string{"10.0.0.0/24", "fd00:3:2:1::/64"},
			newNodePodCIDRs: []string{"10.0.0.0/24", "fd00:3:2:1::/64"},
		},
		{
			name:            "already initialized and different node",
			oldNodePodCIDRs: []string{"192.168.1.0/24", "fd00:1:2:3::/64"},
			newNodePodCIDRs: []string{"10.0.0.0/24", "fd00:3:2:1::/64"},
			expectPanic:     true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &NodePodCIDRHandler{
				podCIDRs: tt.oldNodePodCIDRs,
			}
			node := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-node",
					ResourceVersion: "1",
				},
				Spec: v1.NodeSpec{
					PodCIDRs: tt.newNodePodCIDRs,
				},
			}
			defer func() {
				r := recover()
				if r == nil && tt.expectPanic {
					t.Errorf("The code did not panic")
				} else if r != nil && !tt.expectPanic {
					t.Errorf("The code did panic")
				}
			}()

			n.OnNodeAdd(node)
		})
	}
}

func TestNodePodCIDRHandlerUpdate(t *testing.T) {
	oldKlogOsExit := klog.OsExit
	defer func() {
		klog.OsExit = oldKlogOsExit
	}()
	klog.OsExit = customExit

	tests := []struct {
		name            string
		oldNodePodCIDRs []string
		newNodePodCIDRs []string
		expectPanic     bool
	}{
		{
			name: "both empty",
		},
		{
			name:            "initialize",
			newNodePodCIDRs: []string{"192.168.1.0/24", "fd00:1:2:3::/64"},
		},
		{
			name:            "same node",
			oldNodePodCIDRs: []string{"192.168.1.0/24", "fd00:1:2:3::/64"},
			newNodePodCIDRs: []string{"192.168.1.0/24", "fd00:1:2:3::/64"},
		},
		{
			name:            "different nodes",
			oldNodePodCIDRs: []string{"192.168.1.0/24", "fd00:1:2:3::/64"},
			newNodePodCIDRs: []string{"10.0.0.0/24", "fd00:3:2:1::/64"},
			expectPanic:     true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &NodePodCIDRHandler{
				podCIDRs: tt.oldNodePodCIDRs,
			}
			oldNode := &v1.Node{}
			node := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-node",
					ResourceVersion: "1",
				},
				Spec: v1.NodeSpec{
					PodCIDRs: tt.newNodePodCIDRs,
				},
			}
			defer func() {
				r := recover()
				if r == nil && tt.expectPanic {
					t.Errorf("The code did not panic")
				} else if r != nil && !tt.expectPanic {
					t.Errorf("The code did panic")
				}
			}()

			n.OnNodeUpdate(oldNode, node)
		})
	}
}

func customExit(exitCode int) {
	panic(strconv.Itoa(exitCode))
}

type nodeTweak func(n *v1.Node)

func makeNode(tweaks ...nodeTweak) *v1.Node {
	n := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testNodeName,
		},
	}
	for _, tw := range tweaks {
		tw(n)
	}
	return n
}

func tweakNodeIPs(nodeIPs ...string) nodeTweak {
	return func(n *v1.Node) {
		for _, ip := range nodeIPs {
			n.Status.Addresses = append(n.Status.Addresses, v1.NodeAddress{Type: v1.NodeInternalIP, Address: ip})
		}
	}
}

func TestNewNodeManager(t *testing.T) {
	testCases := []struct {
		name            string
		nodeUpdates     []func(context.Context, clientset.Interface)
		expectedNodeIPs []net.IP
		expectedError   string
	}{
		{
			name: "node object doesn't exist",
			// assert on error thrown by node lister
			expectedError: "node \"test-node\" not found",
		},
		{
			name: "node object exist without NodeIP",
			nodeUpdates: []func(ctx context.Context, client clientset.Interface){
				func(ctx context.Context, client clientset.Interface) {
					// node object doesn't exist initially
				},

				func(ctx context.Context, client clientset.Interface) {
					// node object now exists but without NodeIP
					_, _ = client.CoreV1().Nodes().Create(ctx, makeNode(), metav1.CreateOptions{})
				},
			},
			// assert on error thrown by GetNodeHostIPs()
			expectedError: "host IP unknown; known addresses: []",
		},
		{
			name: "node object exist with NodeIP",
			nodeUpdates: []func(ctx context.Context, client clientset.Interface){
				func(ctx context.Context, client clientset.Interface) {
					// node object doesn't exist initially
				},

				func(ctx context.Context, client clientset.Interface) {
					// node object now exists but without NodeIP
					_, _ = client.CoreV1().Nodes().Create(ctx, makeNode(), metav1.CreateOptions{})
				},

				func(ctx context.Context, client clientset.Interface) {
					// node object got updated with NodeIPs
					_, _ = client.CoreV1().Nodes().Update(ctx, makeNode(
						tweakNodeIPs("192.168.1.10"),
					), metav1.UpdateOptions{})
				},
			},
			expectedNodeIPs: []net.IP{netutils.ParseIPSloppy("192.168.1.10")},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			client := clientsetfake.NewClientset()

			// call the node update functions in go routine, we add 15ms sleep in between
			// each update function to wait for the 10ms poll interval to finish
			go func() {
				// wait for node manager setup
				time.Sleep(100 * time.Millisecond)

				for _, update := range tc.nodeUpdates {
					update(ctx, client)
					// wait for 15 ms for 10ms poll interval to finish
					time.Sleep(15 * time.Millisecond)
				}
			}()
			// initialize the node manager with 10ms poll interval and 1s poll timeout
			nodeManager, err := newNodeManager(ctx, client, time.Second, testNodeName, func(i int) {}, 10*time.Millisecond, time.Second)
			if len(tc.expectedError) > 0 {
				require.Nil(t, nodeManager)
				require.ErrorContains(t, err, tc.expectedError)
			} else {
				require.NoError(t, err)
				require.Equal(t, tc.expectedNodeIPs, nodeManager.NodeIPs())
			}
		})
	}
}

func TestNodeManagerOnNodeChange(t *testing.T) {
	tests := []struct {
		name             string
		initialNodeIPs   []string
		updatedNodeIPs   []string
		expectedExitCode *int
	}{
		{
			name:             "node updated with same NodeIPs",
			initialNodeIPs:   []string{"192.168.1.1", "fd00:1:2:3::1"},
			updatedNodeIPs:   []string{"192.168.1.1", "fd00:1:2:3::1"},
			expectedExitCode: nil,
		},
		{
			name:             "node updated with different NodeIPs",
			initialNodeIPs:   []string{"192.168.1.1", "fd00:1:2:3::1"},
			updatedNodeIPs:   []string{"10.0.1.1", "fd00:3:2:1::2"},
			expectedExitCode: ptr.To(1),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			var exitCode *int
			exitFunc := func(code int) {
				exitCode = &code
			}

			client := clientsetfake.NewClientset()
			_, err := client.CoreV1().Nodes().Create(ctx, makeNode(tweakNodeIPs(tc.initialNodeIPs...)), metav1.CreateOptions{})
			require.NoError(t, err)

			nodeManager, err := newNodeManager(ctx, client, 30*time.Second, testNodeName, exitFunc, 10*time.Millisecond, time.Second)
			require.NoError(t, err)

			nodeManager.onNodeChange(makeNode(tweakNodeIPs(tc.updatedNodeIPs...)))
			require.Equal(t, tc.expectedExitCode, exitCode)
		})
	}
}

func TestNodeManagerOnNodeDelete(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	var exitCode *int
	exitFunc := func(code int) {
		exitCode = &code
	}
	client := clientsetfake.NewClientset()
	_, _ = client.CoreV1().Nodes().Create(ctx, makeNode(tweakNodeIPs("192.168.1.1")), metav1.CreateOptions{})
	nodeManager, err := newNodeManager(ctx, client, 30*time.Second, testNodeName, exitFunc, 10*time.Millisecond, time.Second)
	require.NoError(t, err)

	nodeManager.OnNodeDelete(makeNode())
	require.Equal(t, ptr.To(1), exitCode)
}
