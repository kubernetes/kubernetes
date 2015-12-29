/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/cloudprovider"
	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
	"k8s.io/kubernetes/pkg/controller"
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
		rc := NewRouteController(nil, nil, controller.NoResyncPeriodFunc, myClusterName, cidr)
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

func TestSyncNodeRoute(t *testing.T) {
	cluster := "my-k8s"
	testCases := []struct {
		storeNodes     []*api.Node
		syncNodes      []*api.Node
		initialRoutes  []*cloudprovider.Route
		expectedRoutes []*cloudprovider.Route
	}{
		// 2 nodes, routes already there
		{
			storeNodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "node-1", UID: "01"}, Spec: api.NodeSpec{PodCIDR: "10.120.0.0/24"}},
				{ObjectMeta: api.ObjectMeta{Name: "node-2", UID: "02"}, Spec: api.NodeSpec{PodCIDR: "10.120.1.0/24"}},
			},
			syncNodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "node-1", UID: "01"}, Spec: api.NodeSpec{PodCIDR: "10.120.0.0/24"}},
				{ObjectMeta: api.ObjectMeta{Name: "node-2", UID: "02"}, Spec: api.NodeSpec{PodCIDR: "10.120.1.0/24"}},
			},
			initialRoutes: []*cloudprovider.Route{
				{cluster + "-01", "node-1", "10.120.0.0/24"},
				{cluster + "-02", "node-2", "10.120.1.0/24"},
			},
			expectedRoutes: []*cloudprovider.Route{
				{cluster + "-01", "node-1", "10.120.0.0/24"},
				{cluster + "-02", "node-2", "10.120.1.0/24"},
			},
		},
		// 2 nodes, one route already there
		{
			storeNodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "node-1", UID: "01"}, Spec: api.NodeSpec{PodCIDR: "10.120.0.0/24"}},
				{ObjectMeta: api.ObjectMeta{Name: "node-2", UID: "02"}, Spec: api.NodeSpec{PodCIDR: "10.120.1.0/24"}},
			},
			syncNodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "node-1", UID: "01"}, Spec: api.NodeSpec{PodCIDR: "10.120.0.0/24"}},
				{ObjectMeta: api.ObjectMeta{Name: "node-2", UID: "02"}, Spec: api.NodeSpec{PodCIDR: "10.120.1.0/24"}},
			},
			initialRoutes: []*cloudprovider.Route{
				{cluster + "-01", "node-1", "10.120.0.0/24"},
			},
			expectedRoutes: []*cloudprovider.Route{
				{cluster + "-01", "node-1", "10.120.0.0/24"},
				{cluster + "-02", "node-2", "10.120.1.0/24"},
			},
		},
		// 2 nodes, no routes yet
		{
			storeNodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "node-1", UID: "01"}, Spec: api.NodeSpec{PodCIDR: "10.120.0.0/24"}},
				{ObjectMeta: api.ObjectMeta{Name: "node-2", UID: "02"}, Spec: api.NodeSpec{PodCIDR: "10.120.1.0/24"}},
			},
			syncNodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "node-1", UID: "01"}, Spec: api.NodeSpec{PodCIDR: "10.120.0.0/24"}},
				{ObjectMeta: api.ObjectMeta{Name: "node-2", UID: "02"}, Spec: api.NodeSpec{PodCIDR: "10.120.1.0/24"}},
			},
			initialRoutes: []*cloudprovider.Route{},
			expectedRoutes: []*cloudprovider.Route{
				{cluster + "-01", "node-1", "10.120.0.0/24"},
				{cluster + "-02", "node-2", "10.120.1.0/24"},
			},
		},
		// 2 nodes, a few too many routes
		{
			storeNodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "node-1", UID: "01"}, Spec: api.NodeSpec{PodCIDR: "10.120.0.0/24"}},
				{ObjectMeta: api.ObjectMeta{Name: "node-2", UID: "02"}, Spec: api.NodeSpec{PodCIDR: "10.120.1.0/24"}},
			},
			syncNodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "node-1", UID: "01"}, Spec: api.NodeSpec{PodCIDR: "10.120.0.0/24"}},
				{ObjectMeta: api.ObjectMeta{Name: "node-2", UID: "02"}, Spec: api.NodeSpec{PodCIDR: "10.120.1.0/24"}},
				{ObjectMeta: api.ObjectMeta{Name: "node-3", UID: "03"}, Spec: api.NodeSpec{PodCIDR: "10.120.2.0/24"}},
				{ObjectMeta: api.ObjectMeta{Name: "node-4", UID: "04"}, Spec: api.NodeSpec{PodCIDR: "10.120.3.0/24"}},
			},
			initialRoutes: []*cloudprovider.Route{
				{cluster + "-01", "node-1", "10.120.0.0/24"},
				{cluster + "-02", "node-2", "10.120.1.0/24"},
				{cluster + "-03", "node-3", "10.120.2.0/24"},
				{cluster + "-04", "node-4", "10.120.3.0/24"},
			},
			expectedRoutes: []*cloudprovider.Route{
				{cluster + "-01", "node-1", "10.120.0.0/24"},
				{cluster + "-02", "node-2", "10.120.1.0/24"},
			},
		},
		// 2 nodes, 2 routes, but only 1 is right
		{
			storeNodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "node-1", UID: "01"}, Spec: api.NodeSpec{PodCIDR: "10.120.0.0/24"}},
				{ObjectMeta: api.ObjectMeta{Name: "node-2", UID: "02"}, Spec: api.NodeSpec{PodCIDR: "10.120.1.0/24"}},
			},
			syncNodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "node-1", UID: "01"}, Spec: api.NodeSpec{PodCIDR: "10.120.0.0/24"}},
				{ObjectMeta: api.ObjectMeta{Name: "node-2", UID: "02"}, Spec: api.NodeSpec{PodCIDR: "10.120.1.0/24"}},
			},
			initialRoutes: []*cloudprovider.Route{
				{cluster + "-01", "node-1", "10.120.0.0/24"},
				{cluster + "-02", "node-2", "10.120.2.0/24"}, // the route for node-2 is incorrect
			},
			expectedRoutes: []*cloudprovider.Route{
				{cluster + "-01", "node-1", "10.120.0.0/24"},
				{cluster + "-02", "node-2", "10.120.1.0/24"},
			},
		},
	}

	for i, testCase := range testCases {
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

		client := clientset.NewForConfigOrDie(&client.Config{Host: "", ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
		rc := NewRouteController(routes, client, controller.NoResyncPeriodFunc, cluster, cidr)

		for _, node := range testCase.storeNodes {
			rc.nodeStore.Add(node)
		}
		for _, node := range testCase.syncNodes {
			if err := rc.syncNodeRoute(getKey(node, t)); err != nil {
				t.Errorf("%d. Error from rc.reconcile(): %v", i, err)
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

func TestCheckLeftoverRoutes(t *testing.T) {
	cluster := "my-k8s"
	testCases := []struct {
		storeNodes     []*api.Node
		initialRoutes  []*cloudprovider.Route
		expectedRoutes []*cloudprovider.Route
	}{
		// 2 nodes, no leftover routes
		{
			storeNodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "node-1", UID: "01"}, Spec: api.NodeSpec{PodCIDR: "10.120.0.0/24"}},
				{ObjectMeta: api.ObjectMeta{Name: "node-2", UID: "02"}, Spec: api.NodeSpec{PodCIDR: "10.120.1.0/24"}},
			},
			initialRoutes: []*cloudprovider.Route{
				{cluster + "-01", "node-1", "10.120.0.0/24"},
				{cluster + "-02", "node-2", "10.120.1.0/24"},
			},
			expectedRoutes: []*cloudprovider.Route{
				{cluster + "-01", "node-1", "10.120.0.0/24"},
				{cluster + "-02", "node-2", "10.120.1.0/24"},
			},
		},
		// 1 nodes, one leftover route
		{
			storeNodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "node-1", UID: "01"}, Spec: api.NodeSpec{PodCIDR: "10.120.0.0/24"}},
			},
			initialRoutes: []*cloudprovider.Route{
				{cluster + "-01", "node-1", "10.120.0.0/24"},
				{cluster + "-02", "node-2", "10.120.1.0/24"},
			},
			expectedRoutes: []*cloudprovider.Route{
				{cluster + "-01", "node-1", "10.120.0.0/24"},
			},
		},
		// 2 nodes, no routes yet
		{
			storeNodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "node-1", UID: "01"}, Spec: api.NodeSpec{PodCIDR: "10.120.0.0/24"}},
				{ObjectMeta: api.ObjectMeta{Name: "node-2", UID: "02"}, Spec: api.NodeSpec{PodCIDR: "10.120.1.0/24"}},
			},
			initialRoutes:  []*cloudprovider.Route{},
			expectedRoutes: []*cloudprovider.Route{},
		},
		// 2 nodes, 2 routes, but only 1 is right, the incorrect one should be removed
		{
			storeNodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "node-1", UID: "01"}, Spec: api.NodeSpec{PodCIDR: "10.120.0.0/24"}},
				{ObjectMeta: api.ObjectMeta{Name: "node-2", UID: "02"}, Spec: api.NodeSpec{PodCIDR: "10.120.1.0/24"}},
			},
			initialRoutes: []*cloudprovider.Route{
				{cluster + "-01", "node-1", "10.120.0.0/24"},
				{cluster + "-02", "node-2", "10.120.2.0/24"}, // the route for node-2 is incorrect
			},
			expectedRoutes: []*cloudprovider.Route{
				{cluster + "-01", "node-1", "10.120.0.0/24"},
			},
		},
	}

	for i, testCase := range testCases {
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

		client := clientset.NewForConfigOrDie(&client.Config{Host: "", ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
		rc := NewRouteController(routes, client, controller.NoResyncPeriodFunc, cluster, cidr)

		for _, node := range testCase.storeNodes {
			rc.nodeStore.Add(node)
		}

		rc.checkLeftoverRoutes()

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

func getKey(node *api.Node, t *testing.T) string {
	if key, err := controller.KeyFunc(node); err != nil {
		t.Errorf("Unexpected error getting key for rc %v: %v", node.Name, err)
		return ""
	} else {
		return key
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
