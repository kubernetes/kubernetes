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
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
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
		n.ObjectMeta.ResourceVersion = resourceVersion
	}
}

func TestNewNodeManager(t *testing.T) {
	testCases := []struct {
		name             string
		watchPodCIDRs    bool
		storeUpdateFuncs []func(store cache.Store)
		expectedNodeIPs  []net.IP
		expectedPodCIDRs []string
		expectError      bool
	}{
		{
			name: "without watch pod cidrs",
			storeUpdateFuncs: []func(store cache.Store){
				func(store cache.Store) {
					// node object doesn't exist
				},
				func(store cache.Store) {
					// node initially has no IP
					_ = store.Add(makeNode())
				},
				func(store cache.Store) {
					// node updated with NodeIPs
					_ = store.Add(makeNode(tweakNodeIPs("192.168.1.1")))
				},
				func(store cache.Store) {
					// node updated with NodeIPs and PodCIDRs
					_ = store.Add(makeNode(tweakNodeIPs("192.168.1.1"), tweakPodCIDRs("10.0.0.0/24")))
				},
			},
			watchPodCIDRs:    false,
			expectedNodeIPs:  []net.IP{netutils.ParseIPSloppy("192.168.1.1")},
			expectedPodCIDRs: nil,
			expectError:      false,
		},
		{
			name: "with watch pod cidrs",
			storeUpdateFuncs: []func(store cache.Store){
				func(store cache.Store) {
					// node object doesn't exist
				},
				func(store cache.Store) {
					// node initially has no IP
					_ = store.Add(makeNode())
				},
				func(store cache.Store) {
					// node updated with PodCIDRs
					_ = store.Add(makeNode(tweakPodCIDRs("10.0.0.0/24")))
				},
				func(store cache.Store) {
					// node updated with NodeIPs and PodCIDRs
					_ = store.Add(makeNode(tweakNodeIPs("192.168.1.1"), tweakPodCIDRs("10.0.0.0/24")))
				},
			},
			watchPodCIDRs:    true,
			expectedNodeIPs:  []net.IP{netutils.ParseIPSloppy("192.168.1.1")},
			expectedPodCIDRs: []string{"10.0.0.0/24"},
			expectError:      false,
		},
		{
			name: "with watch pod cidrs and without node update for PodCIDRs",
			storeUpdateFuncs: []func(store cache.Store){
				func(store cache.Store) {
					// node object doesn't exist
				},
				func(store cache.Store) {
					// node initially has no IP
					_ = store.Add(makeNode())
				},
				func(store cache.Store) {
					// node updated with NodeIPs
					_ = store.Add(makeNode(tweakNodeIPs("192.168.1.1")))
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
			informer := informers.NewSharedInformerFactoryWithOptions(client, 0)
			nodeLister := informer.Core().V1().Nodes().Lister()
			nodeStore := informer.Core().V1().Nodes().Informer().GetStore()

			// call the store update functions in go routine, we add 1.5µs sleep in between
			// each update function to wait for the 1µs poll interval to finish
			go func() {
				for _, update := range tc.storeUpdateFuncs {
					update(nodeStore)
					// wait for 1.5µs for 1µs poll interval to finish
					time.Sleep(1500 * time.Nanosecond)
				}
			}()

			nodeManager, err := newNodeManager(ctx, nodeLister, testNodeName, tc.watchPodCIDRs, time.Microsecond, time.Second)
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
		name string
		// nodeIPs represent the initial NodeIPs fetched in NewNodeManager().
		nodeIPs []net.IP
		// podCIDRs represent the initial PodCIDRs fetched in NewNodeManager().
		podCIDRs         []string
		makeNode         func() *v1.Node
		watchPodCIDRs    bool
		expectedExitCode *int
	}{
		{
			name:             "no initial NodeIPs and node updated without NodeIPs",
			podCIDRs:         []string{"10.0.0.0/24"},
			nodeIPs:          []net.IP{},
			makeNode:         func() *v1.Node { return makeNode(tweakPodCIDRs("10.0.0.0/24")) },
			expectedExitCode: nil,
		},
		{
			name:             "no initial NodeIPs and node updated with NodeIPs",
			podCIDRs:         []string{"10.0.0.0/24"},
			makeNode:         func() *v1.Node { return makeNode(tweakNodeIPs("192.168.1.1"), tweakPodCIDRs("10.0.0.0/24")) },
			expectedExitCode: ptr.To(1),
		},
		{
			name:             "without watchPodCIDR and node updated with same NodeIPs and PodCIDRs",
			nodeIPs:          []net.IP{netutils.ParseIPSloppy("192.168.1.1")},
			podCIDRs:         []string{"10.0.0.0/24"},
			makeNode:         func() *v1.Node { return makeNode(tweakNodeIPs("192.168.1.1"), tweakPodCIDRs("10.0.0.0/24")) },
			watchPodCIDRs:    false,
			expectedExitCode: nil,
		},
		{
			name:             "with watchPodCIDR and node updated with same NodeIPs and PodCIDRs",
			nodeIPs:          []net.IP{netutils.ParseIPSloppy("192.168.1.1")},
			podCIDRs:         []string{"10.0.0.0/24"},
			makeNode:         func() *v1.Node { return makeNode(tweakNodeIPs("192.168.1.1"), tweakPodCIDRs("10.0.0.0/24")) },
			watchPodCIDRs:    true,
			expectedExitCode: nil,
		},
		{
			name:             "without watchPodCIDR and node updated with different NodeIPs and same PodCIDRs",
			nodeIPs:          []net.IP{netutils.ParseIPSloppy("192.168.1.1")},
			podCIDRs:         []string{"10.0.0.0/24"},
			makeNode:         func() *v1.Node { return makeNode(tweakNodeIPs("172.16.10.10"), tweakPodCIDRs("10.0.0.0/24")) },
			watchPodCIDRs:    false,
			expectedExitCode: ptr.To(1),
		},
		{
			name:             "with watchPodCIDR and node updated with different NodeIPs and same PodCIDRs",
			nodeIPs:          []net.IP{netutils.ParseIPSloppy("192.168.1.1")},
			podCIDRs:         []string{"10.0.0.0/24"},
			makeNode:         func() *v1.Node { return makeNode(tweakNodeIPs("172.16.10.10"), tweakPodCIDRs("10.0.0.0/24")) },
			watchPodCIDRs:    true,
			expectedExitCode: ptr.To(1),
		},
		{
			name:             "without watchPodCIDR and node updated with same NodeIPs and different PodCIDRs",
			nodeIPs:          []net.IP{netutils.ParseIPSloppy("192.168.1.1")},
			podCIDRs:         []string{"10.0.0.0/24"},
			makeNode:         func() *v1.Node { return makeNode(tweakNodeIPs("192.168.1.1"), tweakPodCIDRs("172.16.10.0/24")) },
			watchPodCIDRs:    false,
			expectedExitCode: nil,
		},
		{
			name:             "with watchPodCIDR and node updated with same NodeIPs and different PodCIDRs",
			nodeIPs:          []net.IP{netutils.ParseIPSloppy("192.168.1.1")},
			podCIDRs:         []string{"10.0.0.0/24"},
			makeNode:         func() *v1.Node { return makeNode(tweakNodeIPs("192.168.1.1"), tweakPodCIDRs("172.16.10.0/24")) },
			watchPodCIDRs:    true,
			expectedExitCode: ptr.To(1),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			var exitCode *int
			n := NodeManager{
				nodeIPs:       tc.nodeIPs,
				podCIDRs:      tc.podCIDRs,
				watchPodCIDRs: tc.watchPodCIDRs,
				logger:        klog.FromContext(ctx),
				exitFunc: func(code int) {
					exitCode = &code
				},
			}
			n.OnNodeChange(tc.makeNode())
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

func TestNodeManagerNode(t *testing.T) {
	var node *v1.Node
	manager := NodeManager{}

	node = makeNode(tweakResourceVersion("1"))
	manager.OnNodeChange(node)
	require.Equal(t, node, manager.Node())

	node = makeNode(tweakResourceVersion("2"))
	manager.OnNodeChange(node)
	require.Equal(t, node, manager.Node())
}
