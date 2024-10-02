/*
Copyright 2015 The Kubernetes Authors.

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

package route

import (
	"context"
	"net"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	cloudprovider "k8s.io/cloud-provider"
	fakecloud "k8s.io/cloud-provider/fake"
	nodeutil "k8s.io/component-helpers/node/util"
	"k8s.io/klog/v2/ktesting"
	netutils "k8s.io/utils/net"

	"github.com/stretchr/testify/assert"
)

func alwaysReady() bool { return true }

func TestIsResponsibleForRoute(t *testing.T) {
	myClusterName := "my-awesome-cluster"
	myClusterRoute := "my-awesome-cluster-12345678-90ab-cdef-1234-567890abcdef"
	testCases := []struct {
		clusterCIDR         string
		routeName           string
		routeCIDR           string
		expectedResponsible bool
	}{
		// Routes that belong to this cluster
		{"10.244.0.0/16", myClusterRoute, "10.244.0.0/24", true},
		{"10.244.0.0/16", myClusterRoute, "10.244.10.0/24", true},
		{"10.244.0.0/16", myClusterRoute, "10.244.255.0/24", true},
		{"10.244.0.0/14", myClusterRoute, "10.244.0.0/24", true},
		{"10.244.0.0/14", myClusterRoute, "10.247.255.0/24", true},
		{"a00:100::/10", myClusterRoute, "a00:100::/24", true},
		// Routes that match our naming/tagging scheme, but are outside our cidr
		{"10.244.0.0/16", myClusterRoute, "10.224.0.0/24", false},
		{"10.244.0.0/16", myClusterRoute, "10.0.10.0/24", false},
		{"10.244.0.0/16", myClusterRoute, "10.255.255.0/24", false},
		{"10.244.0.0/14", myClusterRoute, "10.248.0.0/24", false},
		{"10.244.0.0/14", myClusterRoute, "10.243.255.0/24", false},
		{"a00:100::/10", myClusterRoute, "b00:100::/24", false},
	}
	for i, testCase := range testCases {
		_, cidr, err := netutils.ParseCIDRSloppy(testCase.clusterCIDR)
		if err != nil {
			t.Errorf("%d. Error in test case: unparsable cidr %q", i, testCase.clusterCIDR)
		}
		client := fake.NewSimpleClientset()
		informerFactory := informers.NewSharedInformerFactory(client, 0)
		rc := New(nil, nil, informerFactory.Core().V1().Nodes(), myClusterName, []*net.IPNet{cidr})
		rc.nodeListerSynced = alwaysReady
		route := &cloudprovider.Route{
			Name:            testCase.routeName,
			TargetNode:      types.NodeName("doesnt-matter-for-this-test"),
			DestinationCIDR: testCase.routeCIDR,
		}
		if resp := rc.isResponsibleForRoute(route); resp != testCase.expectedResponsible {
			t.Errorf("%d. isResponsibleForRoute() = %t; want %t", i, resp, testCase.expectedResponsible)
		}
	}
}

func TestReconcile(t *testing.T) {
	cluster := "my-k8s"
	node1 := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-1", UID: "01"}, Spec: v1.NodeSpec{PodCIDR: "10.120.0.0/24", PodCIDRs: []string{"10.120.0.0/24"}}, Status: v1.NodeStatus{Addresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.1.1"}}}}
	// node1NoAddr := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-1", UID: "01"}, Spec: v1.NodeSpec{PodCIDR: "10.120.0.0/24", PodCIDRs: []string{"10.120.0.0/24"}}}
	node2 := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-2", UID: "02"}, Spec: v1.NodeSpec{PodCIDR: "10.120.1.0/24", PodCIDRs: []string{"10.120.1.0/24"}}, Status: v1.NodeStatus{Addresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.2.1"}}}}
	nodeNoCidr := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-2", UID: "02"}, Spec: v1.NodeSpec{PodCIDR: ""}, Status: v1.NodeStatus{Addresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.5.1"}}}}

	node3 := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-3", UID: "03"}, Spec: v1.NodeSpec{PodCIDR: "10.120.0.0/24", PodCIDRs: []string{"10.120.0.0/24", "a00:100::/24"}}, Status: v1.NodeStatus{Addresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}}}
	node4 := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-4", UID: "04"}, Spec: v1.NodeSpec{PodCIDR: "10.120.1.0/24", PodCIDRs: []string{"10.120.1.0/24", "a00:200::/24"}}, Status: v1.NodeStatus{Addresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}}}

	testCases := []struct {
		description                string
		nodes                      []*v1.Node
		initialRoutes              []*cloudprovider.Route
		expectedRoutes             []*cloudprovider.Route
		expectedNetworkUnavailable []bool
		clientset                  *fake.Clientset
		dualStack                  bool
	}{
		{
			description: "routes have no TargetNodeAddresses at the beginning",
			dualStack:   true,
			nodes: []*v1.Node{
				&node3,
				&node4,
			},
			initialRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-3", DestinationCIDR: "10.120.0.0/24", Blackhole: false, EnableNodeAddresses: true},
				{Name: cluster + "-02", TargetNode: "node-4", DestinationCIDR: "10.120.1.0/24", Blackhole: false, EnableNodeAddresses: true},

				{Name: cluster + "-03", TargetNode: "node-3", DestinationCIDR: "a00:100::/24", Blackhole: false, EnableNodeAddresses: true},
				{Name: cluster + "-04", TargetNode: "node-4", DestinationCIDR: "a00:200::/24", Blackhole: false, EnableNodeAddresses: true},
			},
			expectedRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false, EnableNodeAddresses: true},
				{Name: cluster + "-02", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false, EnableNodeAddresses: true},

				{Name: cluster + "-03", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "a00:100::/24", Blackhole: false, EnableNodeAddresses: true},
				{Name: cluster + "-04", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "a00:200::/24", Blackhole: false, EnableNodeAddresses: true},
			},
			expectedNetworkUnavailable: []bool{true, true},
			clientset:                  fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{node1, node2}}),
		},
		{
			description: "routes' TargetNodeAddresses changed",
			dualStack:   true,
			nodes: []*v1.Node{
				&node3,
				&node4,
			},
			initialRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.13.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false, EnableNodeAddresses: true},
				{Name: cluster + "-02", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.14.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false, EnableNodeAddresses: true},

				{Name: cluster + "-03", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.13.1"}}, DestinationCIDR: "a00:100::/24", Blackhole: false, EnableNodeAddresses: true},
				{Name: cluster + "-04", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.14.1"}}, DestinationCIDR: "a00:200::/24", Blackhole: false, EnableNodeAddresses: true},
			},
			expectedRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false, EnableNodeAddresses: true},
				{Name: cluster + "-02", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false, EnableNodeAddresses: true},

				{Name: cluster + "-03", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "a00:100::/24", Blackhole: false, EnableNodeAddresses: true},
				{Name: cluster + "-04", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "a00:200::/24", Blackhole: false, EnableNodeAddresses: true},
			},
			expectedNetworkUnavailable: []bool{true, true},
			clientset:                  fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{node1, node2}}),
		},
		{
			description: "multicidr 2 nodes and no routes",
			dualStack:   true,
			nodes: []*v1.Node{
				&node3,
				&node4,
			},
			initialRoutes: []*cloudprovider.Route{},
			expectedRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},

				{Name: cluster + "-03", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "a00:100::/24", Blackhole: false},
				{Name: cluster + "-04", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "a00:200::/24", Blackhole: false},
			},
			expectedNetworkUnavailable: []bool{true, true},
			clientset:                  fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{node1, node2}}),
		},
		{
			description: "multicidr 2 nodes and all routes created",
			dualStack:   true,
			nodes: []*v1.Node{
				&node3,
				&node4,
			},
			initialRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},

				{Name: cluster + "-03", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "a00:100::/24", Blackhole: false},
				{Name: cluster + "-04", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "a00:200::/24", Blackhole: false},
			},
			expectedRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},

				{Name: cluster + "-03", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "a00:100::/24", Blackhole: false},
				{Name: cluster + "-04", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "a00:200::/24", Blackhole: false},
			},
			expectedNetworkUnavailable: []bool{true, true},
			clientset:                  fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{node1, node2}}),
		},
		{
			description: "multicidr 2 nodes and few wrong routes",
			dualStack:   true,
			nodes: []*v1.Node{
				&node3,
				&node4,
			},
			initialRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},

				{Name: cluster + "-03", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "a00:200::/24", Blackhole: false},
				{Name: cluster + "-04", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "a00:100::/24", Blackhole: false},
			},
			expectedRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},

				{Name: cluster + "-03", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "a00:100::/24", Blackhole: false},
				{Name: cluster + "-04", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "a00:200::/24", Blackhole: false},
			},
			expectedNetworkUnavailable: []bool{true, true},
			clientset:                  fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{node1, node2}}),
		},
		{
			description: "multicidr 2 nodes and some routes created",
			dualStack:   true,
			nodes: []*v1.Node{
				&node3,
				&node4,
			},
			initialRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-04", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "a00:200::/24", Blackhole: false},
			},
			expectedRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},

				{Name: cluster + "-03", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "a00:100::/24", Blackhole: false},
				{Name: cluster + "-04", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "a00:200::/24", Blackhole: false},
			},
			expectedNetworkUnavailable: []bool{true, true},
			clientset:                  fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{node1, node2}}),
		},
		{
			description: "multicidr 2 nodes and too many routes",
			dualStack:   true,
			nodes: []*v1.Node{
				&node3,
				&node4,
			},
			initialRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},
				{Name: cluster + "-001", TargetNode: "node-x", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.5.1"}}, DestinationCIDR: "10.120.2.0/24", Blackhole: false},

				{Name: cluster + "-03", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "a00:100::/24", Blackhole: false},
				{Name: cluster + "-04", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "a00:200::/24", Blackhole: false},
				{Name: cluster + "-0002", TargetNode: "node-y", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.6.1"}}, DestinationCIDR: "a00:300::/24", Blackhole: false},
			},
			expectedRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},

				{Name: cluster + "-03", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "a00:100::/24", Blackhole: false},
				{Name: cluster + "-04", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "a00:200::/24", Blackhole: false},
			},
			expectedNetworkUnavailable: []bool{true, true},
			clientset:                  fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{node1, node2}}),
		},
		{
			description: "single cidr 2 nodes and routes created",
			nodes: []*v1.Node{
				&node1,
				&node2,
			},
			initialRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-1", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.1.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-2", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.2.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},
			},
			expectedRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-1", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.1.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-2", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.2.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},
			},
			expectedNetworkUnavailable: []bool{true, true},
			clientset:                  fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{node1, node2}}),
		},
		{
			description: "single cidr node ips changed so routes should be updated",
			nodes: []*v1.Node{
				&node1,
				&node2,
			},
			initialRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-1", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.1.2"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false, EnableNodeAddresses: true},
				{Name: cluster + "-02", TargetNode: "node-2", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.2.2"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false, EnableNodeAddresses: true},
			},
			expectedRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-1", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.1.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false, EnableNodeAddresses: true},
				{Name: cluster + "-02", TargetNode: "node-2", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.2.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false, EnableNodeAddresses: true},
			},
			expectedNetworkUnavailable: []bool{true, true},
			clientset:                  fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{node1, node2}}),
		},
		{
			description: "single cidr 2 nodes and one route created",
			nodes: []*v1.Node{
				&node1,
				&node2,
			},
			initialRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-1", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.1.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
			},
			expectedRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-1", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.1.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-2", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.2.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},
			},
			expectedNetworkUnavailable: []bool{true, true},
			clientset:                  fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{node1, node2}}),
		},
		{
			description: "single cidr 2 nodes and no routes",
			nodes: []*v1.Node{
				&node1,
				&node2,
			},
			initialRoutes: []*cloudprovider.Route{},
			expectedRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-1", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.1.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-2", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.2.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},
			},
			expectedNetworkUnavailable: []bool{true, true},
			clientset:                  fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{node1, node2}}),
		},
		{
			description: "single cidr 2 nodes and too many routes",
			nodes: []*v1.Node{
				&node1,
				&node2,
			},
			initialRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-1", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.1.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-2", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.2.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},
				{Name: cluster + "-03", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.3.1"}}, DestinationCIDR: "10.120.2.0/24", Blackhole: false},
				{Name: cluster + "-04", TargetNode: "node-4", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.4.1"}}, DestinationCIDR: "10.120.3.0/24", Blackhole: false},
			},
			expectedRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-1", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.1.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-2", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.2.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},
			},
			expectedNetworkUnavailable: []bool{true, true},
			clientset:                  fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{node1, node2}}),
		},
		{
			description: "single cidr 2 nodes and 2 routes with 1 incorrect",
			nodes: []*v1.Node{
				&node1,
				&node2,
			},
			initialRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-1", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.1.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-03", TargetNode: "node-3", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.2.1"}}, DestinationCIDR: "10.120.2.0/24", Blackhole: false},
			},
			expectedRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-1", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.1.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-2", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.2.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},
			},
			expectedNetworkUnavailable: []bool{true, true},
			clientset:                  fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{node1, node2}}),
		},
		{
			description: "single cidr 2 nodes and one node without cidr assigned",
			nodes: []*v1.Node{
				&node1,
				&nodeNoCidr,
			},
			initialRoutes: []*cloudprovider.Route{},
			expectedRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-1", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.1.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
			},
			expectedNetworkUnavailable: []bool{true, false},
			clientset:                  fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{node1, nodeNoCidr}}),
		},
		{
			description: "single cidr 2 nodes and an extra blackhole route in our range",
			nodes: []*v1.Node{
				&node1,
				&node2,
			},
			initialRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-1", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.1.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-2", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.2.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},
				{Name: cluster + "-03", TargetNode: "", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.100.1"}}, DestinationCIDR: "10.120.2.0/24", Blackhole: true},
			},
			expectedRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-1", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.1.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-2", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.2.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},
			},
			expectedNetworkUnavailable: []bool{true, true},
			clientset:                  fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{node1, node2}}),
		},
		{
			description: "single cidr 2 nodes and an extra blackhole route not in our range",
			nodes: []*v1.Node{
				&node1,
				&node2,
			},
			initialRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-1", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.1.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-2", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.2.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},
				{Name: cluster + "-03", TargetNode: "", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.100.1"}}, DestinationCIDR: "10.1.2.0/24", Blackhole: true},
			},
			expectedRoutes: []*cloudprovider.Route{
				{Name: cluster + "-01", TargetNode: "node-1", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.1.1"}}, DestinationCIDR: "10.120.0.0/24", Blackhole: false},
				{Name: cluster + "-02", TargetNode: "node-2", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.2.1"}}, DestinationCIDR: "10.120.1.0/24", Blackhole: false},
				{Name: cluster + "-03", TargetNode: "", TargetNodeAddresses: []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: "10.0.100.1"}}, DestinationCIDR: "10.1.2.0/24", Blackhole: true},
			},
			expectedNetworkUnavailable: []bool{true, true},
			clientset:                  fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{node1, node2}}),
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			cloud := &fakecloud.Cloud{RouteMap: make(map[string]*fakecloud.Route)}
			for _, route := range testCase.initialRoutes {
				fakeRoute := &fakecloud.Route{}
				fakeRoute.ClusterName = cluster
				fakeRoute.Route = *route
				cloud.RouteMap[route.Name] = fakeRoute
			}
			routes, ok := cloud.Routes()
			assert.True(t, ok, "fakecloud failed to run Routes()")
			cidrs := make([]*net.IPNet, 0)
			_, cidr, _ := netutils.ParseCIDRSloppy("10.120.0.0/16")
			cidrs = append(cidrs, cidr)
			if testCase.dualStack {
				_, cidrv6, _ := netutils.ParseCIDRSloppy("ace:cab:deca::/8")
				cidrs = append(cidrs, cidrv6)
			}

			informerFactory := informers.NewSharedInformerFactory(testCase.clientset, 0)
			rc := New(routes, testCase.clientset, informerFactory.Core().V1().Nodes(), cluster, cidrs)
			rc.nodeListerSynced = alwaysReady
			assert.NoError(t, rc.reconcile(ctx, testCase.nodes, testCase.initialRoutes), "failed to reconcile")
			for _, action := range testCase.clientset.Actions() {
				if action.GetVerb() == "update" && action.GetResource().Resource == "nodes" {
					node := action.(core.UpdateAction).GetObject().(*v1.Node)
					_, condition := nodeutil.GetNodeCondition(&node.Status, v1.NodeNetworkUnavailable)
					assert.NotEmpty(t, condition, "Missing NodeNetworkUnavailable condition for Node %q", node.Name)
					check := func(index int) bool {
						return (condition.Status == v1.ConditionFalse) == testCase.expectedNetworkUnavailable[index]
					}
					index := -1
					for j := range testCase.nodes {
						if testCase.nodes[j].Name == node.Name {
							index = j
						}
					}
					if index == -1 {
						// Something's wrong
						continue
					}
					assert.True(t, check(index), "Invalid NodeNetworkUnavailable condition for Node %q, expected %v, got %v",
						node.Name, testCase.expectedNetworkUnavailable[index], (condition.Status == v1.ConditionFalse))
				}
			}
			var finalRoutes []*cloudprovider.Route
			var err error
			timeoutChan := time.After(200 * time.Millisecond)
			tick := time.NewTicker(10 * time.Millisecond)
			defer tick.Stop()
		poll:
			for {
				select {
				case <-tick.C:
					if finalRoutes, err = routes.ListRoutes(ctx, cluster); err == nil && routeListEqual(finalRoutes, testCase.expectedRoutes) {
						break poll
					}
				case <-timeoutChan:
					t.Errorf("rc.reconcile() err is %v,\nfound routes:\n%v\nexpected routes:\n%v\n",
						err, flatten(finalRoutes), flatten(testCase.expectedRoutes))
					break poll
				}
			}
		})

	}
}

func routeListEqual(list1, list2 []*cloudprovider.Route) bool {
	if len(list1) != len(list2) {
		return false
	}

	// nodename+cidr:bool
	seen := make(map[string]bool)

	for _, route1 := range list1 {
		for _, route2 := range list2 {
			if route1.DestinationCIDR == route2.DestinationCIDR && route1.TargetNode == route2.TargetNode &&
				equalNodeAddrs(route1.TargetNodeAddresses, route2.TargetNodeAddresses) {
				seen[string(route1.TargetNode)+route1.DestinationCIDR] = true
				break
			}
		}
	}
	if len(seen) == len(list1) {
		return true
	}
	return false
}

func flatten(list []*cloudprovider.Route) []cloudprovider.Route {
	var structList []cloudprovider.Route
	for _, route := range list {
		structList = append(structList, *route)
	}
	return structList
}
