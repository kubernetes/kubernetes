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

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	"k8s.io/kubernetes/test/utils/ktesting"
	netutils "k8s.io/utils/net"
)

func TestNodeManagerUpsert(t *testing.T) {
	oldKlogOsExit := klog.OsExit
	defer func() {
		klog.OsExit = oldKlogOsExit
	}()
	klog.OsExit = customExit

	baseNode := &v1.Node{
		Spec: v1.NodeSpec{
			PodCIDRs: []string{"10.0.0.0/24", "fd00:1:2:3::/64"},
		},
		Status: v1.NodeStatus{
			Addresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "192.168.1.1"},
				{Type: v1.NodeInternalIP, Address: "fd00:1:2:3::1"},
			},
		},
	}

	tests := []struct {
		name string
		// nodeIPs represent the initial NodeIPs fetched via getNodeIPs() in cmd/kube-proxy/server.go
		nodeIPs []net.IP
		// podCIDRs represent the initial PodCIDRs fetched via getPodCIDRs() in cmd/kube-proxy/server_linux.go
		// note: the initial set of PodCIDRs will never be nil if local mode is NodeCIDR.
		podCIDRs          []string
		localModeNodeCIDR bool
		makeNode           func() *v1.Node
		expectPanic       bool
	}{
		{
			name:     "no initial NodeIPs and node updated without NodeIPs",
			podCIDRs: []string{"10.0.0.0/24", "fd00:1:2:3::/64"},
			makeNode: func() *v1.Node {
				node := baseNode.DeepCopy()
				node.Status.Addresses = []v1.NodeAddress{}
				return node
			},
		},
		{
			name:        "no initial NodeIPs and node updated with NodeIPs",
			podCIDRs:    []string{"10.0.0.0/24", "fd00:1:2:3::/64"},
			makeNode:     func() *v1.Node { return baseNode.DeepCopy() },
			expectPanic: true,
		},
		{
			name:              "DetectLocalMode=NodeCIDR, node updated with same NodeIPs and same PodCIDRs",
			nodeIPs:           []net.IP{netutils.ParseIPSloppy("192.168.1.1"), netutils.ParseIPSloppy("fd00:1:2:3::1")},
			podCIDRs:          []string{"10.0.0.0/24", "fd00:1:2:3::/64"},
			localModeNodeCIDR: true,
			makeNode:           func() *v1.Node { return baseNode.DeepCopy() },
		},
		{
			name:              "DetectLocalMode!=NodeCIDR, node updated with same NodeIPs and same PodCIDRs",
			nodeIPs:           []net.IP{netutils.ParseIPSloppy("192.168.1.1"), netutils.ParseIPSloppy("fd00:1:2:3::1")},
			podCIDRs:          []string{"10.0.0.0/24", "fd00:1:2:3::/64"},
			localModeNodeCIDR: false,
			makeNode:           func() *v1.Node { return baseNode.DeepCopy() },
		},
		{
			name:              "DetectLocalMode=NodeCIDR, node updated with different NodeIPs and same PodCIDRs",
			nodeIPs:           []net.IP{netutils.ParseIPSloppy("192.168.1.1"), netutils.ParseIPSloppy("fd00:1:2:3::1")},
			podCIDRs:          []string{"10.0.0.0/24", "fd00:1:2:3::/64"},
			localModeNodeCIDR: true,
			makeNode: func() *v1.Node {
				node := baseNode.DeepCopy()
				node.Status.Addresses = []v1.NodeAddress{
					{Type: v1.NodeInternalIP, Address: "10.0.1.1"},
					{Type: v1.NodeInternalIP, Address: "fd00:3:2:1::2"},
				}
				return node
			},
			expectPanic: true,
		},
		{
			name:              "DetectLocalMode!=NodeCIDR, node updated with different NodeIPs and same PodCIDRs",
			nodeIPs:           []net.IP{netutils.ParseIPSloppy("192.168.1.1"), netutils.ParseIPSloppy("fd00:1:2:3::1")},
			podCIDRs:          []string{"10.0.0.0/24", "fd00:1:2:3::/64"},
			localModeNodeCIDR: false,
			makeNode: func() *v1.Node {
				node := baseNode.DeepCopy()
				node.Status.Addresses = []v1.NodeAddress{
					{Type: v1.NodeInternalIP, Address: "10.0.1.1"},
					{Type: v1.NodeInternalIP, Address: "fd00:3:2:1::2"},
				}
				return node
			},
			expectPanic: true,
		},
		{
			name:              "DetectLocalMode=NodeCIDR, node updated with same NodeIPs and different PodCIDRs",
			nodeIPs:           []net.IP{netutils.ParseIPSloppy("192.168.1.1"), netutils.ParseIPSloppy("fd00:1:2:3::1")},
			podCIDRs:          []string{"10.0.0.0/24", "fd00:1:2:3::/64"},
			localModeNodeCIDR: true,
			makeNode: func() *v1.Node {
				node := baseNode.DeepCopy()
				node.Spec.PodCIDRs = []string{"192.168.1.0/24", "fd00:3:2:1::/64"}
				return node
			},
			expectPanic: true,
		},
		{
			name:              "DetectLocalMode!=NodeCIDR, node updated with same NodeIPs and different PodCIDRs",
			nodeIPs:           []net.IP{netutils.ParseIPSloppy("192.168.1.1"), netutils.ParseIPSloppy("fd00:1:2:3::1")},
			podCIDRs:          []string{"10.0.0.0/24", "fd00:1:2:3::/64"},
			localModeNodeCIDR: false,
			makeNode: func() *v1.Node {
				node := baseNode.DeepCopy()
				node.Spec.PodCIDRs = []string{"192.168.1.0/24", "fd00:3:2:1::/64"}
				return node
			},
			expectPanic: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			n := NewNodeManager(ctx, tc.nodeIPs, tc.podCIDRs, tc.localModeNodeCIDR, &healthcheck.ProxyHealthServer{})
			if tc.expectPanic {
				require.Panics(t, func() { n.OnNodeUpsert(tc.makeNode()) })
			} else {
				require.NotPanics(t, func() { n.OnNodeUpsert(tc.makeNode()) })
			}
		})
	}
}

func customExit(exitCode int) {
	panic(strconv.Itoa(exitCode))
}
