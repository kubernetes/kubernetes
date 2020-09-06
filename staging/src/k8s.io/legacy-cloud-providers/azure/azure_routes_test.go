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
	"net/http"
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
	"k8s.io/legacy-cloud-providers/azure/retry"
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
		VMSet:             mockVMSet,
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

	route := cloudprovider.Route{TargetNode: "node", DestinationCIDR: "1.2.3.4/24"}
	nodePrivateIP := "2.4.6.8"
	networkRoute := &[]network.Route{
		{
			Name: to.StringPtr("node"),
			RoutePropertiesFormat: &network.RoutePropertiesFormat{
				AddressPrefix:    to.StringPtr("1.2.3.4/24"),
				NextHopIPAddress: &nodePrivateIP,
				NextHopType:      network.RouteNextHopTypeVirtualAppliance,
			},
		},
	}

	testCases := []struct {
		name                  string
		routeTableName        string
		initialRoute          *[]network.Route
		updatedRoute          *[]network.Route
		hasUnmangedNodes      bool
		nodeInformerNotSynced bool
		ipv6DualStackEnabled  bool
		routeTableNotExist    bool
		unmanagedNodeName     string
		routeCIDRs            map[string]string
		expectedRouteCIDRs    map[string]string

		getIPError        error
		getErr            *retry.Error
		secondGetErr      *retry.Error
		createOrUpdateErr *retry.Error
		expectedErrMsg    error
	}{
		{
			name:           "CreateRoute should create route if route doesn't exist",
			routeTableName: "rt1",
			updatedRoute:   networkRoute,
		},
		{
			name:           "CreateRoute should report error if error occurs when invoke CreateOrUpdateRouteTable",
			routeTableName: "rt2",
			updatedRoute:   networkRoute,
			createOrUpdateErr: &retry.Error{
				HTTPStatusCode: http.StatusInternalServerError,
				RawError:       fmt.Errorf("CreateOrUpdate error"),
			},
			expectedErrMsg: fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 500, RawError: CreateOrUpdate error"),
		},
		{
			name:           "CreateRoute should do nothing if route already exists",
			routeTableName: "rt3",
			initialRoute:   networkRoute,
			updatedRoute:   networkRoute,
		},
		{
			name:           "CreateRoute should report error if error occurs when invoke createRouteTable",
			routeTableName: "rt4",
			getErr: &retry.Error{
				HTTPStatusCode: http.StatusNotFound,
				RawError:       cloudprovider.InstanceNotFound,
			},
			createOrUpdateErr: &retry.Error{
				HTTPStatusCode: http.StatusInternalServerError,
				RawError:       fmt.Errorf("CreateOrUpdate error"),
			},
			expectedErrMsg: fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 500, RawError: CreateOrUpdate error"),
		},
		{
			name:           "CreateRoute should report error if error occurs when invoke getRouteTable for the second time",
			routeTableName: "rt5",
			getErr: &retry.Error{
				HTTPStatusCode: http.StatusNotFound,
				RawError:       cloudprovider.InstanceNotFound,
			},
			secondGetErr: &retry.Error{
				HTTPStatusCode: http.StatusInternalServerError,
				RawError:       fmt.Errorf("Get error"),
			},
			expectedErrMsg: fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 500, RawError: Get error"),
		},
		{
			name:           "CreateRoute should report error if error occurs when invoke routeTableClient.Get",
			routeTableName: "rt6",
			getErr: &retry.Error{
				HTTPStatusCode: http.StatusInternalServerError,
				RawError:       fmt.Errorf("Get error"),
			},
			expectedErrMsg: fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 500, RawError: Get error"),
		},
		{
			name:           "CreateRoute should report error if error occurs when invoke GetIPByNodeName",
			routeTableName: "rt7",
			getIPError:     fmt.Errorf("getIP error"),
			expectedErrMsg: fmt.Errorf("timed out waiting for the condition"),
		},
		{
			name:               "CreateRoute should add route to cloud.RouteCIDRs if node is unmanaged",
			routeTableName:     "rt8",
			hasUnmangedNodes:   true,
			unmanagedNodeName:  "node",
			routeCIDRs:         map[string]string{},
			expectedRouteCIDRs: map[string]string{"node": "1.2.3.4/24"},
		},
		{
			name:                 "CreateRoute should report error if node is unmanaged and cloud.ipv6DualStackEnabled is true",
			hasUnmangedNodes:     true,
			ipv6DualStackEnabled: true,
			unmanagedNodeName:    "node",
			expectedErrMsg:       fmt.Errorf("unmanaged nodes are not supported in dual stack mode"),
		},
		{
			name:           "CreateRoute should create route if cloud.ipv6DualStackEnabled is true and route doesn't exist",
			routeTableName: "rt9",
			updatedRoute: &[]network.Route{
				{
					Name: to.StringPtr("node____123424"),
					RoutePropertiesFormat: &network.RoutePropertiesFormat{
						AddressPrefix:    to.StringPtr("1.2.3.4/24"),
						NextHopIPAddress: &nodePrivateIP,
						NextHopType:      network.RouteNextHopTypeVirtualAppliance,
					},
				},
			},
			ipv6DualStackEnabled: true,
		},
		{
			name:                  "CreateRoute should report error if node informer is not synced",
			nodeInformerNotSynced: true,
			expectedErrMsg:        fmt.Errorf("node informer is not synced when trying to GetUnmanagedNodes"),
		},
	}

	for _, test := range testCases {
		initialTable := network.RouteTable{
			Name:     to.StringPtr(test.routeTableName),
			Location: &cloud.Location,
			RouteTablePropertiesFormat: &network.RouteTablePropertiesFormat{
				Routes: test.initialRoute,
			},
		}
		updatedTable := network.RouteTable{
			Name:     to.StringPtr(test.routeTableName),
			Location: &cloud.Location,
			RouteTablePropertiesFormat: &network.RouteTablePropertiesFormat{
				Routes: test.updatedRoute,
			},
		}

		cloud.RouteTableName = test.routeTableName
		cloud.ipv6DualStackEnabled = test.ipv6DualStackEnabled
		if test.hasUnmangedNodes {
			cloud.unmanagedNodes.Insert(test.unmanagedNodeName)
			cloud.routeCIDRs = test.routeCIDRs
		} else {
			cloud.unmanagedNodes = sets.NewString()
			cloud.routeCIDRs = nil
		}
		if test.nodeInformerNotSynced {
			cloud.nodeInformerSynced = func() bool { return false }
		} else {
			cloud.nodeInformerSynced = func() bool { return true }
		}

		mockVMSet.EXPECT().GetIPByNodeName(gomock.Any()).Return(nodePrivateIP, "", test.getIPError).MaxTimes(1)
		mockVMSet.EXPECT().GetPrivateIPsByNodeName("node").Return([]string{nodePrivateIP, "10.10.10.10"}, nil).MaxTimes(1)
		routeTableClient.EXPECT().Get(gomock.Any(), cloud.RouteTableResourceGroup, cloud.RouteTableName, "").Return(initialTable, test.getErr).MaxTimes(1)
		routeTableClient.EXPECT().CreateOrUpdate(gomock.Any(), cloud.RouteTableResourceGroup, cloud.RouteTableName, updatedTable, "").Return(test.createOrUpdateErr).MaxTimes(1)

		//Here is the second invocation when route table doesn't exist
		routeTableClient.EXPECT().Get(gomock.Any(), cloud.RouteTableResourceGroup, cloud.RouteTableName, "").Return(initialTable, test.secondGetErr).MaxTimes(1)

		err := cloud.CreateRoute(context.TODO(), "cluster", "unused", &route)
		assert.Equal(t, cloud.routeCIDRs, test.expectedRouteCIDRs, test.name)
		assert.Equal(t, test.expectedErrMsg, err, test.name)
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

func TestFindFirstIPByFamily(t *testing.T) {
	firstIPv4 := "10.0.0.1"
	firstIPv6 := "2001:1234:5678:9abc::9"
	testIPs := []string{
		firstIPv4,
		"11.0.0.1",
		firstIPv6,
		"fda4:6dee:effc:62a0:0:0:0:0",
	}
	testCases := []struct {
		ipv6           bool
		ips            []string
		expectedIP     string
		expectedErrMsg error
	}{
		{
			ipv6:       true,
			ips:        testIPs,
			expectedIP: firstIPv6,
		},
		{
			ipv6:       false,
			ips:        testIPs,
			expectedIP: firstIPv4,
		},
		{
			ipv6:           true,
			ips:            []string{"10.0.0.1"},
			expectedErrMsg: fmt.Errorf("no match found matching the ipfamily requested"),
		},
	}
	for _, test := range testCases {
		ip, err := findFirstIPByFamily(test.ips, test.ipv6)
		assert.Equal(t, test.expectedErrMsg, err)
		assert.Equal(t, test.expectedIP, ip)
	}
}

func TestRouteNameFuncs(t *testing.T) {
	v4CIDR := "10.0.0.1/16"
	v6CIDR := "fd3e:5f02:6ec0:30ba::/64"
	nodeName := "thisNode"
	testCases := []struct {
		ipv6DualStackEnabled bool
	}{
		{
			ipv6DualStackEnabled: true,
		},
		{
			ipv6DualStackEnabled: false,
		},
	}
	for _, test := range testCases {
		routeName := mapNodeNameToRouteName(test.ipv6DualStackEnabled, types.NodeName(nodeName), v4CIDR)
		outNodeName := mapRouteNameToNodeName(test.ipv6DualStackEnabled, routeName)
		assert.Equal(t, string(outNodeName), nodeName)

		routeName = mapNodeNameToRouteName(test.ipv6DualStackEnabled, types.NodeName(nodeName), v6CIDR)
		outNodeName = mapRouteNameToNodeName(test.ipv6DualStackEnabled, routeName)
		assert.Equal(t, string(outNodeName), nodeName)
	}
}

func TestListRoutes(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	routeTableClient := mockroutetableclient.NewMockInterface(ctrl)
	mockVMSet := mockvmsets.NewMockVMSet(ctrl)

	cloud := &Cloud{
		RouteTablesClient: routeTableClient,
		VMSet:             mockVMSet,
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

	testCases := []struct {
		name                  string
		routeTableName        string
		routeTable            network.RouteTable
		hasUnmangedNodes      bool
		nodeInformerNotSynced bool
		unmanagedNodeName     string
		routeCIDRs            map[string]string
		expectedRoutes        []*cloudprovider.Route
		getErr                *retry.Error
		expectedErrMsg        error
	}{
		{
			name:           "ListRoutes should return correct routes",
			routeTableName: "rt1",
			routeTable: network.RouteTable{
				Name:     to.StringPtr("rt1"),
				Location: &cloud.Location,
				RouteTablePropertiesFormat: &network.RouteTablePropertiesFormat{
					Routes: &[]network.Route{
						{
							Name: to.StringPtr("node"),
							RoutePropertiesFormat: &network.RoutePropertiesFormat{
								AddressPrefix: to.StringPtr("1.2.3.4/24"),
							},
						},
					},
				},
			},
			expectedRoutes: []*cloudprovider.Route{
				{
					Name:            "node",
					TargetNode:      mapRouteNameToNodeName(false, "node"),
					DestinationCIDR: "1.2.3.4/24",
				},
			},
		},
		{
			name:              "ListRoutes should return correct routes if there's unmanaged nodes",
			routeTableName:    "rt2",
			hasUnmangedNodes:  true,
			unmanagedNodeName: "umanaged-node",
			routeCIDRs:        map[string]string{"umanaged-node": "2.2.3.4/24"},
			routeTable: network.RouteTable{
				Name:     to.StringPtr("rt2"),
				Location: &cloud.Location,
				RouteTablePropertiesFormat: &network.RouteTablePropertiesFormat{
					Routes: &[]network.Route{
						{
							Name: to.StringPtr("node"),
							RoutePropertiesFormat: &network.RoutePropertiesFormat{
								AddressPrefix: to.StringPtr("1.2.3.4/24"),
							},
						},
					},
				},
			},
			expectedRoutes: []*cloudprovider.Route{
				{
					Name:            "node",
					TargetNode:      mapRouteNameToNodeName(false, "node"),
					DestinationCIDR: "1.2.3.4/24",
				},
				{
					Name:            "umanaged-node",
					TargetNode:      mapRouteNameToNodeName(false, "umanaged-node"),
					DestinationCIDR: "2.2.3.4/24",
				},
			},
		},
		{
			name:           "ListRoutes should return nil if routeTabel don't exist",
			routeTableName: "rt3",
			routeTable:     network.RouteTable{},
			getErr: &retry.Error{
				HTTPStatusCode: http.StatusNotFound,
				RawError:       cloudprovider.InstanceNotFound,
			},
			expectedRoutes: []*cloudprovider.Route{},
		},
		{
			name:           "ListRoutes should report error if error occurs when invoke routeTableClient.Get",
			routeTableName: "rt4",
			routeTable:     network.RouteTable{},
			getErr: &retry.Error{
				HTTPStatusCode: http.StatusInternalServerError,
				RawError:       fmt.Errorf("Get error"),
			},
			expectedErrMsg: fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 500, RawError: Get error"),
		},
		{
			name:                  "ListRoutes should report error if node informer is not synced",
			routeTableName:        "rt5",
			nodeInformerNotSynced: true,
			routeTable:            network.RouteTable{},
			expectedErrMsg:        fmt.Errorf("node informer is not synced when trying to GetUnmanagedNodes"),
		},
	}

	for _, test := range testCases {
		if test.hasUnmangedNodes {
			cloud.unmanagedNodes.Insert(test.unmanagedNodeName)
			cloud.routeCIDRs = test.routeCIDRs
		} else {
			cloud.unmanagedNodes = sets.NewString()
			cloud.routeCIDRs = nil
		}

		if test.nodeInformerNotSynced {
			cloud.nodeInformerSynced = func() bool { return false }
		} else {
			cloud.nodeInformerSynced = func() bool { return true }
		}

		cloud.RouteTableName = test.routeTableName
		routeTableClient.EXPECT().Get(gomock.Any(), cloud.RouteTableResourceGroup, test.routeTableName, "").Return(test.routeTable, test.getErr)

		routes, err := cloud.ListRoutes(context.TODO(), "cluster")
		assert.Equal(t, test.expectedRoutes, routes, test.name)
		assert.Equal(t, test.expectedErrMsg, err, test.name)
	}
}
