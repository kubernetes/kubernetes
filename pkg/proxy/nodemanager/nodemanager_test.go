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

package nodemanager

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

func tweakResourceVersion(resourceVersion string) nodeTweak {
	return func(n *v1.Node) {
		n.ResourceVersion = resourceVersion
	}
}

func TestNewNodeManager(t *testing.T) {
	testCases := []struct {
		name             string
		watchPodCIDRs    bool
		nodeUpdates      []func(context.Context, clientset.Interface)
		expectedNodeIPs  []net.IP
		expectedPodCIDRs []string
		expectedError    string
	}{
		{
			name: "node object doesn't exist",
			// times out and ignores the error
			expectedNodeIPs: nil,
		},
		{
			name:          "node object doesn't exist, with watchPodCIDRs",
			watchPodCIDRs: true,
			// assert on error thrown by newNodeManager()
			expectedError: "timeout waiting for node \"test-node\" to exist",
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
			// times out and ignores the error
			expectedNodeIPs: nil,
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
		{
			name:          "watchPodCIDRs and node object exist without PodCIDRs",
			watchPodCIDRs: true,
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
			// assert on error thrown by newNodeManager()
			expectedError: "timeout waiting for PodCIDR allocation on node \"test-node\"",
		},
		{
			name:          "watchPodCIDRs and node object exist with NodeIP and PodCIDR",
			watchPodCIDRs: true,
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
				func(ctx context.Context, client clientset.Interface) {
					// node updated with PodCIDRs
					_, _ = client.CoreV1().Nodes().Update(ctx, makeNode(
						tweakNodeIPs("192.168.1.1"),
						tweakPodCIDRs("10.0.0.0/24"),
					), metav1.UpdateOptions{})
				},
			},
			expectedNodeIPs:  []net.IP{netutils.ParseIPSloppy("192.168.1.1")},
			expectedPodCIDRs: []string{"10.0.0.0/24"},
		},
		{
			name:          "watchPodCIDRs and node object exist without NodeIP and with PodCIDR",
			watchPodCIDRs: true,
			nodeUpdates: []func(ctx context.Context, client clientset.Interface){
				func(ctx context.Context, client clientset.Interface) {
					// node object doesn't exist initially
				},

				func(ctx context.Context, client clientset.Interface) {
					// node object now exists but without NodeIP
					_, _ = client.CoreV1().Nodes().Create(ctx, makeNode(), metav1.CreateOptions{})
				},
				func(ctx context.Context, client clientset.Interface) {
					// node updated with PodCIDRs
					_, _ = client.CoreV1().Nodes().Update(ctx, makeNode(
						tweakPodCIDRs("10.0.0.0/24"),
					), metav1.UpdateOptions{})
				},
			},
			// times out and ignores the error
			expectedNodeIPs:  nil,
			expectedPodCIDRs: []string{"10.0.0.0/24"},
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
			nodeManager, err := newNodeManager(ctx, client, time.Second, testNodeName, tc.watchPodCIDRs, func(i int) {}, 10*time.Millisecond, time.Second, time.Second)
			if len(tc.expectedError) > 0 {
				require.Nil(t, nodeManager)
				require.ErrorContains(t, err, tc.expectedError)
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
		{
			name:             "watchPodCIDR and node updated with same PodCIDRs",
			initialNodeIPs:   []string{"192.168.1.1", "fd00:1:2:3::1"},
			initialPodCIDRs:  []string{"10.0.0.0/8", "fd01:2345::/64"},
			updatedNodeIPs:   []string{"192.168.1.1", "fd00:1:2:3::1"},
			updatedPodCIDRs:  []string{"10.0.0.0/8", "fd01:2345::/64"},
			watchPodCIDRs:    true,
			expectedExitCode: nil,
		},
		{
			name:             "watchPodCIDR and node updated with different PodCIDRs",
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

			nodeManager, err := newNodeManager(ctx, client, 30*time.Second, testNodeName, tc.watchPodCIDRs, exitFunc, 10*time.Millisecond, time.Second, time.Second)
			require.NoError(t, err)

			nodeManager.OnNodeChange(makeNode(tweakNodeIPs(tc.updatedNodeIPs...), tweakPodCIDRs(tc.updatedPodCIDRs...)))
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
	nodeManager, err := newNodeManager(ctx, client, 30*time.Second, testNodeName, false, exitFunc, 10*time.Millisecond, time.Second, time.Second)
	require.NoError(t, err)

	nodeManager.OnNodeDelete(makeNode())
	require.Equal(t, ptr.To(1), exitCode)
}

func TestNodeManagerNode(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	client := clientsetfake.NewClientset()
	_, _ = client.CoreV1().Nodes().Create(ctx, makeNode(
		tweakNodeIPs("192.168.1.1"),
		tweakResourceVersion("1")),
		metav1.CreateOptions{})

	nodeManager, err := newNodeManager(ctx, client, 30*time.Second, testNodeName, false, func(i int) {}, time.Nanosecond, time.Nanosecond, time.Nanosecond)
	require.NoError(t, err)
	require.Equal(t, "1", nodeManager.Node().ResourceVersion)

	nodeManager.OnNodeChange(makeNode(tweakResourceVersion("2")))
	require.NoError(t, err)
	require.Equal(t, "2", nodeManager.Node().ResourceVersion)
}

func TestDetectNodeIPs(t *testing.T) {
	cases := []struct {
		name           string
		rawNodeIPs     []net.IP
		bindAddress    string
		expectedFamily v1.IPFamily
		expectedIPv4   string
		expectedIPv6   string
	}{
		{
			name:           "Bind address IPv4 unicast address and no Node object",
			rawNodeIPs:     nil,
			bindAddress:    "10.0.0.1",
			expectedFamily: v1.IPv4Protocol,
			expectedIPv4:   "10.0.0.1",
			expectedIPv6:   "::1",
		},
		{
			name:           "Bind address IPv6 unicast address and no Node object",
			rawNodeIPs:     nil,
			bindAddress:    "fd00:4321::2",
			expectedFamily: v1.IPv6Protocol,
			expectedIPv4:   "127.0.0.1",
			expectedIPv6:   "fd00:4321::2",
		},
		{
			name:           "No Valid IP found and no bind address",
			rawNodeIPs:     nil,
			bindAddress:    "",
			expectedFamily: v1.IPv4Protocol,
			expectedIPv4:   "127.0.0.1",
			expectedIPv6:   "::1",
		},
		{
			name:           "No Valid IP found and unspecified bind address",
			rawNodeIPs:     nil,
			bindAddress:    "0.0.0.0",
			expectedFamily: v1.IPv4Protocol,
			expectedIPv4:   "127.0.0.1",
			expectedIPv6:   "::1",
		},
		{
			name:           "Bind address 0.0.0.0 and node with IPv4 InternalIP set",
			rawNodeIPs:     []net.IP{netutils.ParseIPSloppy("192.168.1.1")},
			bindAddress:    "0.0.0.0",
			expectedFamily: v1.IPv4Protocol,
			expectedIPv4:   "192.168.1.1",
			expectedIPv6:   "::1",
		},
		{
			name:           "Bind address :: and node with IPv4 InternalIP set",
			rawNodeIPs:     []net.IP{netutils.ParseIPSloppy("192.168.1.1")},
			bindAddress:    "::",
			expectedFamily: v1.IPv4Protocol,
			expectedIPv4:   "192.168.1.1",
			expectedIPv6:   "::1",
		},
		{
			name:           "Bind address 0.0.0.0 and node with IPv6 InternalIP set",
			rawNodeIPs:     []net.IP{netutils.ParseIPSloppy("fd00:1234::1")},
			bindAddress:    "0.0.0.0",
			expectedFamily: v1.IPv6Protocol,
			expectedIPv4:   "127.0.0.1",
			expectedIPv6:   "fd00:1234::1",
		},
		{
			name:           "Bind address :: and node with IPv6 InternalIP set",
			rawNodeIPs:     []net.IP{netutils.ParseIPSloppy("fd00:1234::1")},
			bindAddress:    "::",
			expectedFamily: v1.IPv6Protocol,
			expectedIPv4:   "127.0.0.1",
			expectedIPv6:   "fd00:1234::1",
		},
		{
			name: "Dual stack, primary IPv4",
			rawNodeIPs: []net.IP{
				netutils.ParseIPSloppy("90.90.90.90"),
				netutils.ParseIPSloppy("2001:db8::2"),
			},
			bindAddress:    "::",
			expectedFamily: v1.IPv4Protocol,
			expectedIPv4:   "90.90.90.90",
			expectedIPv6:   "2001:db8::2",
		},
		{
			name: "Dual stack, primary IPv6",
			rawNodeIPs: []net.IP{
				netutils.ParseIPSloppy("2001:db8::2"),
				netutils.ParseIPSloppy("90.90.90.90"),
			},
			bindAddress:    "0.0.0.0",
			expectedFamily: v1.IPv6Protocol,
			expectedIPv4:   "90.90.90.90",
			expectedIPv6:   "2001:db8::2",
		},
		{
			name: "Dual stack, override IPv4",
			rawNodeIPs: []net.IP{
				netutils.ParseIPSloppy("2001:db8::2"),
				netutils.ParseIPSloppy("90.90.90.90"),
			},
			bindAddress:    "80.80.80.80",
			expectedFamily: v1.IPv4Protocol,
			expectedIPv4:   "80.80.80.80",
			expectedIPv6:   "2001:db8::2",
		},
		{
			name: "Dual stack, override IPv6",
			rawNodeIPs: []net.IP{
				netutils.ParseIPSloppy("90.90.90.90"),
				netutils.ParseIPSloppy("2001:db8::2"),
			},
			bindAddress:    "2001:db8::555",
			expectedFamily: v1.IPv6Protocol,
			expectedIPv4:   "90.90.90.90",
			expectedIPv6:   "2001:db8::555",
		},
		{
			name: "Dual stack, override primary family, IPv4",
			rawNodeIPs: []net.IP{
				netutils.ParseIPSloppy("2001:db8::2"),
				netutils.ParseIPSloppy("90.90.90.90"),
			},
			bindAddress:    "127.0.0.1",
			expectedFamily: v1.IPv4Protocol,
			expectedIPv4:   "127.0.0.1",
			expectedIPv6:   "2001:db8::2",
		},
		{
			name: "Dual stack, override primary family, IPv6",
			rawNodeIPs: []net.IP{
				netutils.ParseIPSloppy("90.90.90.90"),
				netutils.ParseIPSloppy("2001:db8::2"),
			},
			bindAddress:    "::1",
			expectedFamily: v1.IPv6Protocol,
			expectedIPv4:   "90.90.90.90",
			expectedIPv6:   "::1",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			primaryFamily, ips := DetectNodeIPs(c.rawNodeIPs, c.bindAddress)
			if primaryFamily != c.expectedFamily {
				t.Errorf("Expected family %q got %q", c.expectedFamily, primaryFamily)
			}
			if ips[v1.IPv4Protocol].String() != c.expectedIPv4 {
				t.Errorf("Expected IPv4 %q got %q", c.expectedIPv4, ips[v1.IPv4Protocol].String())
			}
			if ips[v1.IPv6Protocol].String() != c.expectedIPv6 {
				t.Errorf("Expected IPv6 %q got %q", c.expectedIPv6, ips[v1.IPv6Protocol].String())
			}
		})
	}
}
