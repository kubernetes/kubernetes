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
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/test/utils/ktesting"
	netutils "k8s.io/utils/net"
	"k8s.io/utils/ptr"
)

const (
	testNodeName = "test-node"
)

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

func tweakPodCIDRs(podCIDRs ...string) nodeTweak {
	return func(n *v1.Node) {
		n.Spec.PodCIDRs = append(n.Spec.PodCIDRs, podCIDRs...)
	}
}

func TestNewNodeManager(t *testing.T) {
	testCases := []struct {
		name             string
		watchPodCIDRs    bool
		storeUpdateFuncs []func(ctx context.Context, client clientset.Interface)
		expectedNodeIPs  []net.IP
		expectedPodCIDRs []string
		expectError      bool
	}{
		{
			name: "without watch pod cidrs",
			storeUpdateFuncs: []func(ctx context.Context, client clientset.Interface){
				func(ctx context.Context, client clientset.Interface) {
					// node object doesn't exist
				},
				func(ctx context.Context, client clientset.Interface) {
					// node initially has no IP
					_, _ = client.CoreV1().Nodes().Create(ctx, makeNode(), metav1.CreateOptions{})
				},
				func(ctx context.Context, client clientset.Interface) {
					// node updated with NodeIPs
					_, _ = client.CoreV1().Nodes().Update(ctx, makeNode(
						tweakNodeIPs("192.168.1.1"),
					), metav1.UpdateOptions{})
				},
				func(ctx context.Context, client clientset.Interface) {
					// node updated with NodeIPs and PodCIDRs
					_, _ = client.CoreV1().Nodes().Update(ctx, makeNode(
						tweakNodeIPs("192.168.1.1"),
						tweakPodCIDRs("10.0.0.0/24"),
					), metav1.UpdateOptions{})

				},
			},
			watchPodCIDRs:    false,
			expectedNodeIPs:  []net.IP{netutils.ParseIPSloppy("192.168.1.1")},
			expectedPodCIDRs: nil,
			expectError:      false,
		},
		{
			name: "with watch pod cidrs",
			storeUpdateFuncs: []func(ctx context.Context, client clientset.Interface){
				func(ctx context.Context, client clientset.Interface) {
					// node object doesn't exist
				},
				func(ctx context.Context, client clientset.Interface) {
					// node initially has no IP
					_, _ = client.CoreV1().Nodes().Create(ctx, makeNode(), metav1.CreateOptions{})
				},
				func(ctx context.Context, client clientset.Interface) {
					// node updated with PodCIDRs
					_, _ = client.CoreV1().Nodes().Update(ctx, makeNode(
						tweakPodCIDRs("10.0.0.0/24"),
					), metav1.UpdateOptions{})
				},
				func(ctx context.Context, client clientset.Interface) {
					// node updated with NodeIPs and PodCIDRs
					_, _ = client.CoreV1().Nodes().Update(ctx, makeNode(
						tweakPodCIDRs("10.0.0.0/24"),
						tweakNodeIPs("192.168.1.1"),
					), metav1.UpdateOptions{})
				},
			},
			watchPodCIDRs:    true,
			expectedNodeIPs:  []net.IP{netutils.ParseIPSloppy("192.168.1.1")},
			expectedPodCIDRs: []string{"10.0.0.0/24"},
			expectError:      false,
		},
		{
			name: "with watch pod cidrs and without node update for PodCIDRs",
			storeUpdateFuncs: []func(ctx context.Context, client clientset.Interface){
				func(ctx context.Context, client clientset.Interface) {
					// node object doesn't exist
				},
				func(ctx context.Context, client clientset.Interface) {
					// node initially has no IP
					_, _ = client.CoreV1().Nodes().Create(ctx, makeNode(), metav1.CreateOptions{})
				},
				func(ctx context.Context, client clientset.Interface) {
					// node updated with NodeIPs
					_, _ = client.CoreV1().Nodes().Update(ctx, makeNode(
						tweakNodeIPs("192.168.1.1"),
					), metav1.UpdateOptions{})
				},
			},
			watchPodCIDRs:    true,
			expectedNodeIPs:  nil,
			expectedPodCIDRs: nil,
			expectError:      true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			client := clientsetfake.NewClientset()

			// call the store update functions in go routine, we add 1.5µs sleep in between
			// each update function to wait for the 1µs poll interval to finish
			go func() {
				// wait for node manager setup
				time.Sleep(time.Millisecond)

				for _, update := range tc.storeUpdateFuncs {
					update(ctx, client)
					// wait for 1.5µs for 1µs poll interval to finish
					time.Sleep(1500 * time.Nanosecond)
				}
			}()

			nodeManager, err := newNodeManager(ctx, client, time.Second, testNodeName, tc.watchPodCIDRs, func(i int) {}, time.Microsecond, time.Second)
			if tc.expectError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				require.Equal(t, tc.expectedNodeIPs, nodeManager.NodeIPs())
				require.Equal(t, tc.expectedPodCIDRs, nodeManager.PodCIDRs())
			}
		})
	}
}

func TestNodeManagerOnNodeChange(t *testing.T) {
	tests := []struct {
		name             string
		initialNodeIPs   []string
		initialPodCIDRs  []string
		updatedNodeIPs   []string
		updatedPodCIDRs  []string
		watchPodCIDRs    bool
		expectedExitCode *int
	}{
		{
			name:             "no initial NodeIPs and node updated without NodeIPs",
			initialPodCIDRs:  []string{"10.0.0.0/8", "fd01:2345::/64"},
			updatedPodCIDRs:  []string{"10.0.0.0/8", "fd01:2345::/64"},
			expectedExitCode: nil,
		},
		{
			name:             "no initial NodeIPs and node updated with NodeIPs",
			initialPodCIDRs:  []string{"10.0.0.0/8", "fd01:2345::/64"},
			updatedPodCIDRs:  []string{"10.0.0.0/8", "fd01:2345::/64"},
			updatedNodeIPs:   []string{"192.168.1.1", "fd00:1:2:3::1"},
			expectedExitCode: ptr.To(1),
		},
		{
			name:             "without watchPodCIDR and node updated with same NodeIPs and PodCIDRs",
			initialNodeIPs:   []string{"192.168.1.1", "fd00:1:2:3::1"},
			initialPodCIDRs:  []string{"10.0.0.0/8", "fd01:2345::/64"},
			updatedNodeIPs:   []string{"192.168.1.1", "fd00:1:2:3::1"},
			updatedPodCIDRs:  []string{"10.0.0.0/8", "fd01:2345::/64"},
			watchPodCIDRs:    false,
			expectedExitCode: nil,
		},
		{
			name:             "with watchPodCIDR and node updated with same NodeIPs and PodCIDRs",
			initialNodeIPs:   []string{"192.168.1.1", "fd00:1:2:3::1"},
			initialPodCIDRs:  []string{"10.0.0.0/8", "fd01:2345::/64"},
			updatedNodeIPs:   []string{"192.168.1.1", "fd00:1:2:3::1"},
			updatedPodCIDRs:  []string{"10.0.0.0/8", "fd01:2345::/64"},
			watchPodCIDRs:    true,
			expectedExitCode: nil,
		},
		{
			name:             "without watchPodCIDR and node updated with different NodeIPs and same PodCIDRs",
			initialNodeIPs:   []string{"192.168.1.1", "fd00:1:2:3::1"},
			initialPodCIDRs:  []string{"10.0.0.0/8", "fd01:2345::/64"},
			updatedNodeIPs:   []string{"172.16.10.10", "fd00:3:2:1::1"},
			updatedPodCIDRs:  []string{"10.0.0.0/8", "fd01:2345::/64"},
			watchPodCIDRs:    false,
			expectedExitCode: ptr.To(1),
		},
		{
			name:             "with watchPodCIDR and node updated with different NodeIPs and same PodCIDRs",
			initialNodeIPs:   []string{"192.168.1.1", "fd00:1:2:3::1"},
			initialPodCIDRs:  []string{"10.0.0.0/8", "fd01:2345::/64"},
			updatedNodeIPs:   []string{"172.16.10.10", "fd00:3:2:1::1"},
			updatedPodCIDRs:  []string{"10.0.0.0/8", "fd01:2345::/64"},
			watchPodCIDRs:    true,
			expectedExitCode: ptr.To(1),
		},
		{

			name:             "without watchPodCIDR and node updated with same NodeIPs and different PodCIDRs",
			initialNodeIPs:   []string{"192.168.1.1", "fd00:1:2:3::1"},
			initialPodCIDRs:  []string{"10.0.0.0/8", "fd01:2345::/64"},
			updatedNodeIPs:   []string{"192.168.1.1", "fd00:1:2:3::1"},
			updatedPodCIDRs:  []string{"172.16.10.0/24", "fd01:5422::/64"},
			watchPodCIDRs:    false,
			expectedExitCode: nil,
		},
		{
			name:             "with watchPodCIDR and node updated with same NodeIPs and different PodCIDRs",
			initialNodeIPs:   []string{"192.168.1.1", "fd00:1:2:3::1"},
			initialPodCIDRs:  []string{"10.0.0.0/8", "fd01:2345::/64"},
			updatedNodeIPs:   []string{"192.168.1.1", "fd00:1:2:3::1"},
			updatedPodCIDRs:  []string{"172.16.10.0/24", "fd01:5422::/64"},
			watchPodCIDRs:    true,
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
			_, err := client.CoreV1().Nodes().Create(ctx, makeNode(
				tweakNodeIPs(tc.initialNodeIPs...),
				tweakPodCIDRs(tc.initialPodCIDRs...),
			), metav1.CreateOptions{})
			require.NoError(t, err)

			nodeManager, err := newNodeManager(ctx, client, 30*time.Second, testNodeName, tc.watchPodCIDRs, exitFunc, time.Nanosecond, time.Nanosecond)
			require.NoError(t, err)

			nodeManager.onNodeChange(makeNode(tweakNodeIPs(tc.updatedNodeIPs...), tweakPodCIDRs(tc.updatedPodCIDRs...)))
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
	nodeManager, err := newNodeManager(ctx, client, 30*time.Second, testNodeName, false, exitFunc, time.Nanosecond, time.Nanosecond)
	require.NoError(t, err)

	nodeManager.OnNodeDelete(makeNode())
	require.Equal(t, ptr.To(1), exitCode)
}
