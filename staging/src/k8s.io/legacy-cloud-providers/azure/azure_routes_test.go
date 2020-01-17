// +build !providerless

/*
Copyright 2018 The Kubernetes Authors.

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

package azure

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	cloudprovider "k8s.io/cloud-provider"
)

func TestDeleteRoute(t *testing.T) {
	fakeRoutes := newFakeRoutesClient()

	cloud := &Cloud{
		RoutesClient: fakeRoutes,
		Config: Config{
			RouteTableResourceGroup: "foo",
			RouteTableName:          "bar",
			Location:                "location",
		},
		unmanagedNodes:     sets.NewString(),
		nodeInformerSynced: func() bool { return true },
	}
	route := cloudprovider.Route{TargetNode: "node", DestinationCIDR: "1.2.3.4/24"}
	routeName := mapNodeNameToRouteName(false, route.TargetNode, route.DestinationCIDR)

	fakeRoutes.FakeStore = map[string]map[string]network.Route{
		cloud.RouteTableName: {
			routeName: {},
		},
	}

	err := cloud.DeleteRoute(context.TODO(), "cluster", &route)
	if err != nil {
		t.Errorf("unexpected error deleting route: %v", err)
		t.FailNow()
	}

	mp, found := fakeRoutes.FakeStore[cloud.RouteTableName]
	if !found {
		t.Errorf("unexpected missing item for %s", cloud.RouteTableName)
		t.FailNow()
	}
	ob, found := mp[routeName]
	if found {
		t.Errorf("unexpectedly found: %v that should have been deleted.", ob)
		t.FailNow()
	}

	// test delete route for unmanaged nodes.
	nodeName := "node1"
	nodeCIDR := "4.3.2.1/24"
	cloud.unmanagedNodes.Insert(nodeName)
	cloud.routeCIDRs = map[string]string{
		nodeName: nodeCIDR,
	}
	route1 := cloudprovider.Route{
		TargetNode:      mapRouteNameToNodeName(false, nodeName),
		DestinationCIDR: nodeCIDR,
	}
	err = cloud.DeleteRoute(context.TODO(), "cluster", &route1)
	if err != nil {
		t.Errorf("unexpected error deleting route: %v", err)
		t.FailNow()
	}
	cidr, found := cloud.routeCIDRs[nodeName]
	if found {
		t.Errorf("unexpected CIDR item (%q) for %s", cidr, nodeName)
	}
}

func TestCreateRoute(t *testing.T) {
	fakeTable := newFakeRouteTablesClient()
	fakeVM := &fakeVMSet{}
	fakeRoutes := newFakeRoutesClient()

	cloud := &Cloud{
		RouteTablesClient: fakeTable,
		RoutesClient:      fakeRoutes,
		vmSet:             fakeVM,
		Config: Config{
			RouteTableResourceGroup: "foo",
			RouteTableName:          "bar",
			Location:                "location",
		},
		unmanagedNodes:     sets.NewString(),
		nodeInformerSynced: func() bool { return true },
	}
	cache, _ := cloud.newRouteTableCache()
	cloud.rtCache = cache

	expectedTable := network.RouteTable{
		Name:     &cloud.RouteTableName,
		Location: &cloud.Location,
	}
	fakeTable.FakeStore = map[string]map[string]network.RouteTable{}
	fakeTable.FakeStore[cloud.RouteTableResourceGroup] = map[string]network.RouteTable{
		cloud.RouteTableName: expectedTable,
	}
	route := cloudprovider.Route{TargetNode: "node", DestinationCIDR: "1.2.3.4/24"}

	nodeIP := "2.4.6.8"
	fakeVM.NodeToIP = map[string]string{
		"node": nodeIP,
	}

	err := cloud.CreateRoute(context.TODO(), "cluster", "unused", &route)
	if err != nil {
		t.Errorf("unexpected error create if not exists route table: %v", err)
		t.FailNow()
	}
	if len(fakeTable.Calls) != 1 || fakeTable.Calls[0] != "Get" {
		t.Errorf("unexpected calls create if not exists, exists: %v", fakeTable.Calls)
	}
	if len(fakeRoutes.Calls) != 1 || fakeRoutes.Calls[0] != "CreateOrUpdate" {
		t.Errorf("unexpected route calls create if not exists, exists: %v", fakeRoutes.Calls)
	}

	routeName := mapNodeNameToRouteName(false, route.TargetNode, string(route.DestinationCIDR))
	routeInfo, found := fakeRoutes.FakeStore[cloud.RouteTableName][routeName]
	if !found {
		t.Errorf("could not find route: %v in %v", routeName, fakeRoutes.FakeStore)
		t.FailNow()
	}
	if *routeInfo.AddressPrefix != route.DestinationCIDR {
		t.Errorf("Expected cidr: %s, saw %s", *routeInfo.AddressPrefix, route.DestinationCIDR)
	}
	if routeInfo.NextHopType != network.RouteNextHopTypeVirtualAppliance {
		t.Errorf("Expected next hop: %v, saw %v", network.RouteNextHopTypeVirtualAppliance, routeInfo.NextHopType)
	}
	if *routeInfo.NextHopIPAddress != nodeIP {
		t.Errorf("Expected IP address: %s, saw %s", nodeIP, *routeInfo.NextHopIPAddress)
	}

	// test create again without real creation, clean fakeRoute calls
	fakeRoutes.Calls = []string{}
	routeInfo.Name = &routeName
	route.Name = routeName
	expectedTable.RouteTablePropertiesFormat = &network.RouteTablePropertiesFormat{
		Routes: &[]network.Route{routeInfo},
	}
	cloud.rtCache.Set(cloud.RouteTableName, &expectedTable)

	err = cloud.CreateRoute(context.TODO(), "cluster", "unused", &route)
	if err != nil {
		t.Errorf("unexpected error creating route: %v", err)
		t.FailNow()
	}
	if len(fakeRoutes.Calls) != 0 {
		t.Errorf("unexpected route calls create if not exists, exists: %v", fakeRoutes.Calls)
	}

	// test create route for unmanaged nodes.
	nodeName := "node1"
	nodeCIDR := "4.3.2.1/24"
	cloud.unmanagedNodes.Insert(nodeName)
	cloud.routeCIDRs = map[string]string{}
	route1 := cloudprovider.Route{
		TargetNode:      mapRouteNameToNodeName(false, nodeName),
		DestinationCIDR: nodeCIDR,
	}
	err = cloud.CreateRoute(context.TODO(), "cluster", "unused", &route1)
	if err != nil {
		t.Errorf("unexpected error creating route: %v", err)
		t.FailNow()
	}
	cidr, found := cloud.routeCIDRs[nodeName]
	if !found {
		t.Errorf("unexpected missing item for %s", nodeName)
		t.FailNow()
	}
	if cidr != nodeCIDR {
		t.Errorf("unexpected cidr %s, saw %s", nodeCIDR, cidr)
	}
}

func TestCreateRouteTableIfNotExists_Exists(t *testing.T) {
	fake := newFakeRouteTablesClient()
	cloud := &Cloud{
		RouteTablesClient: fake,
		Config: Config{
			RouteTableResourceGroup: "foo",
			RouteTableName:          "bar",
			Location:                "location",
		},
	}
	cache, _ := cloud.newRouteTableCache()
	cloud.rtCache = cache

	expectedTable := network.RouteTable{
		Name:     &cloud.RouteTableName,
		Location: &cloud.Location,
	}
	fake.FakeStore = map[string]map[string]network.RouteTable{}
	fake.FakeStore[cloud.RouteTableResourceGroup] = map[string]network.RouteTable{
		cloud.RouteTableName: expectedTable,
	}
	err := cloud.createRouteTableIfNotExists("clusterName", &cloudprovider.Route{TargetNode: "node", DestinationCIDR: "1.2.3.4/16"})
	if err != nil {
		t.Errorf("unexpected error create if not exists route table: %v", err)
		t.FailNow()
	}
	if len(fake.Calls) != 1 || fake.Calls[0] != "Get" {
		t.Errorf("unexpected calls create if not exists, exists: %v", fake.Calls)
	}
}

func TestCreateRouteTableIfNotExists_NotExists(t *testing.T) {
	fake := newFakeRouteTablesClient()
	cloud := &Cloud{
		RouteTablesClient: fake,
		Config: Config{
			RouteTableResourceGroup: "foo",
			RouteTableName:          "bar",
			Location:                "location",
		},
	}
	cache, _ := cloud.newRouteTableCache()
	cloud.rtCache = cache

	expectedTable := network.RouteTable{
		Name:     &cloud.RouteTableName,
		Location: &cloud.Location,
	}

	err := cloud.createRouteTableIfNotExists("clusterName", &cloudprovider.Route{TargetNode: "node", DestinationCIDR: "1.2.3.4/16"})
	if err != nil {
		t.Errorf("unexpected error create if not exists route table: %v", err)
		t.FailNow()
	}

	table := fake.FakeStore[cloud.RouteTableResourceGroup][cloud.RouteTableName]
	if *table.Location != *expectedTable.Location {
		t.Errorf("mismatch: %s vs %s", *table.Location, *expectedTable.Location)
	}
	if *table.Name != *expectedTable.Name {
		t.Errorf("mismatch: %s vs %s", *table.Name, *expectedTable.Name)
	}
	if len(fake.Calls) != 2 || fake.Calls[0] != "Get" || fake.Calls[1] != "CreateOrUpdate" {
		t.Errorf("unexpected calls create if not exists, exists: %v", fake.Calls)
	}
}

func TestCreateRouteTable(t *testing.T) {
	fake := newFakeRouteTablesClient()
	cloud := &Cloud{
		RouteTablesClient: fake,
		Config: Config{
			RouteTableResourceGroup: "foo",
			RouteTableName:          "bar",
			Location:                "location",
		},
	}
	cache, _ := cloud.newRouteTableCache()
	cloud.rtCache = cache

	expectedTable := network.RouteTable{
		Name:     &cloud.RouteTableName,
		Location: &cloud.Location,
	}

	err := cloud.createRouteTable()
	if err != nil {
		t.Errorf("unexpected error in creating route table: %v", err)
		t.FailNow()
	}

	table := fake.FakeStore["foo"]["bar"]
	if *table.Location != *expectedTable.Location {
		t.Errorf("mismatch: %s vs %s", *table.Location, *expectedTable.Location)
	}
	if *table.Name != *expectedTable.Name {
		t.Errorf("mismatch: %s vs %s", *table.Name, *expectedTable.Name)
	}
}

func TestProcessRoutes(t *testing.T) {
	tests := []struct {
		rt            network.RouteTable
		exists        bool
		err           error
		expectErr     bool
		expectedError string
		expectedRoute []cloudprovider.Route
		name          string
	}{
		{
			err:           fmt.Errorf("test error"),
			expectErr:     true,
			expectedError: "test error",
		},
		{
			exists: false,
			name:   "doesn't exist",
		},
		{
			rt:     network.RouteTable{},
			exists: true,
			name:   "nil routes",
		},
		{
			rt: network.RouteTable{
				RouteTablePropertiesFormat: &network.RouteTablePropertiesFormat{},
			},
			exists: true,
			name:   "no routes",
		},
		{
			rt: network.RouteTable{
				RouteTablePropertiesFormat: &network.RouteTablePropertiesFormat{
					Routes: &[]network.Route{
						{
							Name: to.StringPtr("name"),
							RoutePropertiesFormat: &network.RoutePropertiesFormat{
								AddressPrefix: to.StringPtr("1.2.3.4/16"),
							},
						},
					},
				},
			},
			exists: true,
			expectedRoute: []cloudprovider.Route{
				{
					Name:            "name",
					TargetNode:      mapRouteNameToNodeName(false, "name"),
					DestinationCIDR: "1.2.3.4/16",
				},
			},
			name: "one route",
		},
		{
			rt: network.RouteTable{
				RouteTablePropertiesFormat: &network.RouteTablePropertiesFormat{
					Routes: &[]network.Route{
						{
							Name: to.StringPtr("name"),
							RoutePropertiesFormat: &network.RoutePropertiesFormat{
								AddressPrefix: to.StringPtr("1.2.3.4/16"),
							},
						},
						{
							Name: to.StringPtr("name2"),
							RoutePropertiesFormat: &network.RoutePropertiesFormat{
								AddressPrefix: to.StringPtr("5.6.7.8/16"),
							},
						},
					},
				},
			},
			exists: true,
			expectedRoute: []cloudprovider.Route{
				{
					Name:            "name",
					TargetNode:      mapRouteNameToNodeName(false, "name"),
					DestinationCIDR: "1.2.3.4/16",
				},
				{
					Name:            "name2",
					TargetNode:      mapRouteNameToNodeName(false, "name2"),
					DestinationCIDR: "5.6.7.8/16",
				},
			},
			name: "more routes",
		},
	}
	for _, test := range tests {
		routes, err := processRoutes(false, test.rt, test.exists, test.err)
		if test.expectErr {
			if err == nil {
				t.Errorf("%s: unexpected non-error", test.name)
				continue
			}
			if err.Error() != test.expectedError {
				t.Errorf("%s: Expected error: %v, saw error: %v", test.name, test.expectedError, err.Error())
				continue
			}
		}
		if !test.expectErr && err != nil {
			t.Errorf("%s; unexpected error: %v", test.name, err)
			continue
		}
		if len(routes) != len(test.expectedRoute) {
			t.Errorf("%s: Unexpected difference: %#v vs %#v", test.name, routes, test.expectedRoute)
			continue
		}
		for ix := range test.expectedRoute {
			if !reflect.DeepEqual(test.expectedRoute[ix], *routes[ix]) {
				t.Errorf("%s: Unexpected difference: %#v vs %#v", test.name, test.expectedRoute[ix], *routes[ix])
			}
		}
	}
}

func errorNotNil(t *testing.T, err error) {
	if nil != err {
		t.Errorf("%s: failure error: %v", t.Name(), err)
	}
}
func TestFindFirstIPByFamily(t *testing.T) {
	firstIPv4 := "10.0.0.1"
	firstIPv6 := "2001:1234:5678:9abc::9"
	ips := []string{
		firstIPv4,
		"11.0.0.1",
		firstIPv6,
		"fda4:6dee:effc:62a0:0:0:0:0",
	}
	outIPV4, err := findFirstIPByFamily(ips, false)
	errorNotNil(t, err)
	assert.Equal(t, outIPV4, firstIPv4)

	outIPv6, err := findFirstIPByFamily(ips, true)
	errorNotNil(t, err)
	assert.Equal(t, outIPv6, firstIPv6)
}

func TestRouteNameFuncs(t *testing.T) {
	v4CIDR := "10.0.0.1/16"
	v6CIDR := "fd3e:5f02:6ec0:30ba::/64"
	nodeName := "thisNode"

	routeName := mapNodeNameToRouteName(false, types.NodeName(nodeName), v4CIDR)
	outNodeName := mapRouteNameToNodeName(false, routeName)
	assert.Equal(t, string(outNodeName), nodeName)

	routeName = mapNodeNameToRouteName(false, types.NodeName(nodeName), v6CIDR)
	outNodeName = mapRouteNameToNodeName(false, routeName)
	assert.Equal(t, string(outNodeName), nodeName)
}
