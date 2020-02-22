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
	"time"

	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/legacy-cloud-providers/azure/clients/routetableclient/mockroutetableclient"
	"k8s.io/legacy-cloud-providers/azure/mockvmsets"
)

func TestDeleteRoute(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	routeTableClient := mockroutetableclient.NewMockInterface(ctrl)

	cloud := &Cloud{
		RouteTablesClient: routeTableClient,
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
	cloud.routeUpdater = newDelayedRouteUpdater(cloud, 100*time.Millisecond)
	go cloud.routeUpdater.run()
	route := cloudprovider.Route{
		TargetNode:      "node",
		DestinationCIDR: "1.2.3.4/24",
	}
	routeName := mapNodeNameToRouteName(false, route.TargetNode, route.DestinationCIDR)
	routeTables := network.RouteTable{
		Name:     &cloud.RouteTableName,
		Location: &cloud.Location,
		RouteTablePropertiesFormat: &network.RouteTablePropertiesFormat{
			Routes: &[]network.Route{
				{
					Name: &routeName,
				},
			},
		},
	}
	routeTablesAfterDeletion := network.RouteTable{
		Name:     &cloud.RouteTableName,
		Location: &cloud.Location,
		RouteTablePropertiesFormat: &network.RouteTablePropertiesFormat{
			Routes: &[]network.Route{},
		},
	}
	routeTableClient.EXPECT().Get(gomock.Any(), cloud.RouteTableResourceGroup, cloud.RouteTableName, "").Return(routeTables, nil)
	routeTableClient.EXPECT().CreateOrUpdate(gomock.Any(), cloud.RouteTableResourceGroup, cloud.RouteTableName, routeTablesAfterDeletion, "").Return(nil)
	err := cloud.DeleteRoute(context.TODO(), "cluster", &route)
	if err != nil {
		t.Errorf("unexpected error deleting route: %v", err)
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
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	routeTableClient := mockroutetableclient.NewMockInterface(ctrl)
	mockVMSet := mockvmsets.NewMockVMSet(ctrl)

	cloud := &Cloud{
		RouteTablesClient: routeTableClient,
		vmSet:             mockVMSet,
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
	cloud.routeUpdater = newDelayedRouteUpdater(cloud, 100*time.Millisecond)
	go cloud.routeUpdater.run()

	currentTable := network.RouteTable{
		Name:                       &cloud.RouteTableName,
		Location:                   &cloud.Location,
		RouteTablePropertiesFormat: &network.RouteTablePropertiesFormat{},
	}
	updatedTable := network.RouteTable{
		Name:     &cloud.RouteTableName,
		Location: &cloud.Location,
		RouteTablePropertiesFormat: &network.RouteTablePropertiesFormat{
			Routes: &[]network.Route{
				{
					Name: to.StringPtr("node"),
					RoutePropertiesFormat: &network.RoutePropertiesFormat{
						AddressPrefix:    to.StringPtr("1.2.3.4/24"),
						NextHopIPAddress: to.StringPtr("2.4.6.8"),
						NextHopType:      network.RouteNextHopTypeVirtualAppliance,
					},
				},
			},
		},
	}
	route := cloudprovider.Route{TargetNode: "node", DestinationCIDR: "1.2.3.4/24"}
	nodeIP := "2.4.6.8"

	routeTableClient.EXPECT().Get(gomock.Any(), cloud.RouteTableResourceGroup, cloud.RouteTableName, "").Return(currentTable, nil)
	routeTableClient.EXPECT().CreateOrUpdate(gomock.Any(), cloud.RouteTableResourceGroup, cloud.RouteTableName, updatedTable, "").Return(nil)
	mockVMSet.EXPECT().GetIPByNodeName("node").Return(nodeIP, "", nil).AnyTimes()
	err := cloud.CreateRoute(context.TODO(), "cluster", "unused", &route)
	if err != nil {
		t.Errorf("unexpected error create if not exists route table: %v", err)
		t.FailNow()
	}

	// test create again without real creation.
	routeTableClient.EXPECT().Get(gomock.Any(), cloud.RouteTableResourceGroup, cloud.RouteTableName, "").Return(updatedTable, nil).Times(1)
	err = cloud.CreateRoute(context.TODO(), "cluster", "unused", &route)
	if err != nil {
		t.Errorf("unexpected error creating route: %v", err)
		t.FailNow()
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

func TestCreateRouteTable(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	routeTableClient := mockroutetableclient.NewMockInterface(ctrl)

	cloud := &Cloud{
		RouteTablesClient: routeTableClient,
		Config: Config{
			RouteTableResourceGroup: "foo",
			RouteTableName:          "bar",
			Location:                "location",
		},
	}
	cache, _ := cloud.newRouteTableCache()
	cloud.rtCache = cache

	expectedTable := network.RouteTable{
		Name:                       &cloud.RouteTableName,
		Location:                   &cloud.Location,
		RouteTablePropertiesFormat: &network.RouteTablePropertiesFormat{},
	}
	routeTableClient.EXPECT().CreateOrUpdate(gomock.Any(), cloud.RouteTableResourceGroup, cloud.RouteTableName, expectedTable, "").Return(nil)
	err := cloud.createRouteTable()
	if err != nil {
		t.Errorf("unexpected error in creating route table: %v", err)
		t.FailNow()
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
