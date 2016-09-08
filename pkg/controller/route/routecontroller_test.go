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
	"net"
	"testing"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/cloudprovider"
	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
)

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
		// Routes that match our naming/tagging scheme, but are outside our cidr
		{"10.244.0.0/16", myClusterRoute, "10.224.0.0/24", false},
		{"10.244.0.0/16", myClusterRoute, "10.0.10.0/24", false},
		{"10.244.0.0/16", myClusterRoute, "10.255.255.0/24", false},
		{"10.244.0.0/14", myClusterRoute, "10.248.0.0/24", false},
		{"10.244.0.0/14", myClusterRoute, "10.243.255.0/24", false},
	}
	for i, testCase := range testCases {
		_, cidr, err := net.ParseCIDR(testCase.clusterCIDR)
		if err != nil {
			t.Errorf("%d. Error in test case: unparsable cidr %q", i, testCase.clusterCIDR)
		}
		rc := New(nil, nil, myClusterName, cidr, 0)
		route := &cloudprovider.Route{
			Name:            testCase.routeName,
			TargetInstance:  "doesnt-matter-for-this-test",
			DestinationCIDR: testCase.routeCIDR,
		}
		if resp := rc.isResponsibleForRoute(route); resp != testCase.expectedResponsible {
			t.Errorf("%d. isResponsibleForRoute() = %t; want %t", i, resp, testCase.expectedResponsible)
		}
	}
}

func TestReconcile(t *testing.T) {
	cluster := "my-k8s"
	node1 := api.Node{ObjectMeta: api.ObjectMeta{Name: "node-1", UID: "01"}, Spec: api.NodeSpec{PodCIDR: "10.120.0.0/24"}}
	node2 := api.Node{ObjectMeta: api.ObjectMeta{Name: "node-2", UID: "02"}, Spec: api.NodeSpec{PodCIDR: "10.120.1.0/24"}}
	nodeNoCidr := api.Node{ObjectMeta: api.ObjectMeta{Name: "node-2", UID: "02"}, Spec: api.NodeSpec{PodCIDR: ""}}

	testCases := []struct {
		node                       api.Node
		initialRoutes              []*cloudprovider.Route
		expectedRoutes             []*cloudprovider.Route
		expectedNetworkUnavailable bool
		clientset                  *fake.Clientset
		statusType                 StatusType
	}{
		// Add a node
		{
			node:          node1,
			initialRoutes: []*cloudprovider.Route{},
			expectedRoutes: []*cloudprovider.Route{
				{cluster + "-01", node1.Name, node1.Spec.PodCIDR},
			},
			expectedNetworkUnavailable: true,
			clientset:                  fake.NewSimpleClientset(&api.NodeList{Items: []api.Node{node1}}),
			statusType:                 statusAdd,
		},
		// Remove a node
		{
			node: node1,
			initialRoutes: []*cloudprovider.Route{
				{cluster + "-01", node1.Name, node1.Spec.PodCIDR},
				{cluster + "-02", node2.Name, node2.Spec.PodCIDR},
			},
			expectedRoutes: []*cloudprovider.Route{
				{cluster + "-02", node2.Name, node2.Spec.PodCIDR},
			},
			expectedNetworkUnavailable: true,
			clientset:                  fake.NewSimpleClientset(&api.NodeList{Items: []api.Node{node1, node2}}),
			statusType:                 statusDelete,
		},
		// Update a node, add CIDR
		{
			node: node1,
			initialRoutes: []*cloudprovider.Route{
				{cluster + "-02", node2.Name, node2.Spec.PodCIDR},
			},
			expectedRoutes: []*cloudprovider.Route{
				{cluster + "-01", node1.Name, node1.Spec.PodCIDR},
				{cluster + "-02", node2.Name, node2.Spec.PodCIDR},
			},
			expectedNetworkUnavailable: true,
			clientset:                  fake.NewSimpleClientset(&api.NodeList{Items: []api.Node{node1, node2}}),
			statusType:                 statusUpdate,
		},
		// Added node without CIDR assigned.
		{
			node:                       nodeNoCidr,
			initialRoutes:              []*cloudprovider.Route{},
			expectedRoutes:             []*cloudprovider.Route{},
			expectedNetworkUnavailable: false,
			clientset:                  fake.NewSimpleClientset(&api.NodeList{Items: []api.Node{nodeNoCidr}}),
			statusType:                 statusAdd,
		},
	}
	for i, testCase := range testCases {
		glog.Infof("Testcase: %d\n", i)
		cloud := &fakecloud.FakeCloud{RouteMap: make(map[string]*fakecloud.FakeRoute)}
		for _, route := range testCase.initialRoutes {
			fakeRoute := &fakecloud.FakeRoute{}
			fakeRoute.ClusterName = cluster
			fakeRoute.Route = *route
			cloud.RouteMap[route.Name] = fakeRoute
		}
		routes, ok := cloud.Routes()
		if !ok {
			t.Error("Error in test: fakecloud doesn't support Routes()")
		}
		_, cidr, _ := net.ParseCIDR("10.120.0.0/16")
		rc := New(routes, testCase.clientset, cluster, cidr, 0)
		nodeStatus := NodeStatus{
			node:   &testCase.node,
			status: testCase.statusType,
		}
		if err := rc.reconcile(nodeStatus, testCase.initialRoutes); err != nil {
			t.Errorf("%d. Error from rc.reconcile(): %v", i, err)
		}
		for _, action := range testCase.clientset.Actions() {
			if action.GetVerb() == "update" && action.GetResource().Resource == "nodes" {
				node := action.(core.UpdateAction).GetObject().(*api.Node)
				_, condition := api.GetNodeCondition(&node.Status, api.NodeNetworkUnavailable)
				if condition == nil {
					t.Errorf("%d. Missing NodeNetworkUnavailable condition for Node %v", i, node.Name)
				} else {
					check := func() bool {
						return (condition.Status == api.ConditionFalse) == testCase.expectedNetworkUnavailable
					}
					if !check() {
						t.Errorf("%d. Invalid NodeNetworkUnavailable condition for Node %v, expected %v, got %v",
							i, node.Name, testCase.expectedNetworkUnavailable, (condition.Status == api.ConditionFalse))
					}
				}
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
				if finalRoutes, err = routes.ListRoutes(cluster); err == nil && routeListEqual(finalRoutes, testCase.expectedRoutes) {
					break poll
				}
			case <-timeoutChan:
				t.Errorf("%d. rc.reconcile() = %v, routes:\n%v\nexpected: nil, routes:\n%v\n", i, err, flatten(finalRoutes), flatten(testCase.expectedRoutes))
				break poll
			}
		}
	}
}

func routeListEqual(list1, list2 []*cloudprovider.Route) bool {
	if len(list1) != len(list2) {
		return false
	}
	routeMap1 := make(map[string]*cloudprovider.Route)
	for _, route1 := range list1 {
		routeMap1[route1.Name] = route1
	}
	for _, route2 := range list2 {
		if route1, exists := routeMap1[route2.Name]; !exists || *route1 != *route2 {
			return false
		}
	}
	return true
}

func flatten(list []*cloudprovider.Route) []cloudprovider.Route {
	var structList []cloudprovider.Route
	for _, route := range list {
		structList = append(structList, *route)
	}
	return structList
}
