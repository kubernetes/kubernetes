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
	"fmt"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/cloudprovider"

	"github.com/Azure/azure-sdk-for-go/arm/network"
	"github.com/Azure/go-autorest/autorest/to"
)

func TestCreateRoute(t *testing.T) {
	fake := newFakeRouteTablesClient()
	cloud := &Cloud{
		RouteTablesClient: fake,
		Config: Config{
			ResourceGroup:  "foo",
			RouteTableName: "bar",
			Location:       "location",
		},
	}
	cloud.rtCache, _ = cloud.newRouteTableCache()
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
					TargetNode:      mapRouteNameToNodeName("name"),
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
					TargetNode:      mapRouteNameToNodeName("name"),
					DestinationCIDR: "1.2.3.4/16",
				},
				{
					Name:            "name2",
					TargetNode:      mapRouteNameToNodeName("name2"),
					DestinationCIDR: "5.6.7.8/16",
				},
			},
			name: "more routes",
		},
	}
	for _, test := range tests {
		routes, err := processRoutes(test.rt, test.exists, test.err)
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
