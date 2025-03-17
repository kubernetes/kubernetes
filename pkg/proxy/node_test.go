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
	"net"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
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
	_, ctx := ktesting.NewTestContext(t)
	client := clientsetfake.NewClientset()
	informer := informers.NewSharedInformerFactoryWithOptions(client, 0)
	nodeLister := informer.Core().V1().Nodes().Lister()
	nodeStore := informer.Core().V1().Nodes().Informer().GetStore()
	nodeIP := "192.168.0.1"

	go func() {
		// node object doesn't exist
		// wait for 1.5µs for 1µs poll interval to finish
		time.Sleep(1500 * time.Nanosecond)

		// node initially has no IP
		_ = nodeStore.Add(makeNode())
		// wait for 1.5µs for 1µs poll interval to finish
		time.Sleep(1500 * time.Nanosecond)

		// node updated with IP
		_ = nodeStore.Add(makeNode(tweakNodeIPs(nodeIP)))
	}()

	nodeManager := newNodeManager(ctx, nodeLister, testNodeName, time.Microsecond, time.Second)
	require.Equal(t, []net.IP{netutils.ParseIPSloppy(nodeIP)}, nodeManager.NodeIPs())
}

func TestNodeManagerOnNodeChange(t *testing.T) {
	tests := []struct {
		name string
		// nodeIPs represent the initial NodeIPs fetched in NewNodeManager().
		nodeIPs          []net.IP
		makeNode         func() *v1.Node
		expectedExitCode *int
	}{
		{
			name:             "no initial NodeIPs and node updated without NodeIPs",
			nodeIPs:          []net.IP{},
			makeNode:         func() *v1.Node { return makeNode() },
			expectedExitCode: nil,
		},
		{
			name:             "no initial NodeIPs and node updated with NodeIPs",
			makeNode:         func() *v1.Node { return makeNode(tweakNodeIPs("192.168.1.1", "fd00:1:2:3::1")) },
			expectedExitCode: ptr.To(1),
		},
		{
			name:             "node updated with same NodeIPs",
			nodeIPs:          []net.IP{netutils.ParseIPSloppy("192.168.1.1"), netutils.ParseIPSloppy("fd00:1:2:3::1")},
			makeNode:         func() *v1.Node { return makeNode(tweakNodeIPs("192.168.1.1", "fd00:1:2:3::1")) },
			expectedExitCode: nil,
		},
		{
			name:             "node updated with different NodeIPs",
			nodeIPs:          []net.IP{netutils.ParseIPSloppy("192.168.1.1"), netutils.ParseIPSloppy("fd00:1:2:3::1")},
			makeNode:         func() *v1.Node { return makeNode(tweakNodeIPs("10.0.1.1", "fd00:3:2:1::2")) },
			expectedExitCode: ptr.To(1),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			var exitCode *int
			n := NodeManager{
				nodeIPs: tc.nodeIPs,
				logger:  klog.FromContext(ctx),
				exitFunc: func(code int) {
					exitCode = &code
				},
			}
			n.onNodeChange(tc.makeNode())
			require.Equal(t, tc.expectedExitCode, exitCode)
		})
	}
}

func TestNodeManagerOnNodeDelete(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	var exitCode *int
	n := NodeManager{
		logger: klog.FromContext(ctx),
		exitFunc: func(code int) {
			exitCode = &code
		},
	}
	n.OnNodeDelete(makeNode())
	require.Equal(t, ptr.To(1), exitCode)
}
