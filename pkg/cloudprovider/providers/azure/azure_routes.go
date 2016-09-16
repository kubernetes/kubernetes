/*
Copyright 2016 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/cloudprovider"

	"github.com/Azure/azure-sdk-for-go/arm/network"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/glog"
)

// ListRoutes lists all managed routes that belong to the specified clusterName
func (az *Cloud) ListRoutes(clusterName string) (routes []*cloudprovider.Route, err error) {
	glog.V(10).Infof("list: START clusterName=%q", clusterName)
	routeTable, existsRouteTable, err := az.getRouteTable()
	if err != nil {
		return nil, err
	}
	if !existsRouteTable {
		return []*cloudprovider.Route{}, nil
	}

	var kubeRoutes []*cloudprovider.Route
	if routeTable.Properties.Routes != nil {
		kubeRoutes = make([]*cloudprovider.Route, len(*routeTable.Properties.Routes))
		for i, route := range *routeTable.Properties.Routes {
			instance := getInstanceName(*route.Name)
			cidr := *route.Properties.AddressPrefix
			glog.V(10).Infof("list: * instance=%q, cidr=%q", instance, cidr)

			kubeRoutes[i] = &cloudprovider.Route{
				Name:            *route.Name,
				TargetInstance:  instance,
				DestinationCIDR: cidr,
			}
		}
	}

	glog.V(10).Info("list: FINISH")
	return kubeRoutes, nil
}

// CreateRoute creates the described managed route
// route.Name will be ignored, although the cloud-provider may use nameHint
// to create a more user-meaningful name.
func (az *Cloud) CreateRoute(clusterName string, nameHint string, kubeRoute *cloudprovider.Route) error {
	glog.V(2).Infof("create: creating route. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetInstance, kubeRoute.DestinationCIDR)

	routeTable, existsRouteTable, err := az.getRouteTable()
	if err != nil {
		return err
	}
	if !existsRouteTable {
		routeTable = network.RouteTable{
			Name:       to.StringPtr(az.RouteTableName),
			Location:   to.StringPtr(az.Location),
			Properties: &network.RouteTablePropertiesFormat{},
		}

		glog.V(3).Infof("create: creating routetable. routeTableName=%q", az.RouteTableName)
		_, err = az.RouteTablesClient.CreateOrUpdate(az.ResourceGroup, az.RouteTableName, routeTable, nil)
		if err != nil {
			return err
		}

		routeTable, err = az.RouteTablesClient.Get(az.ResourceGroup, az.RouteTableName, "")
		if err != nil {
			return err
		}
	}

	// ensure the subnet is properly configured
	subnet, err := az.SubnetsClient.Get(az.ResourceGroup, az.VnetName, az.SubnetName, "")
	if err != nil {
		// 404 is fatal here
		return err
	}
	if subnet.Properties.RouteTable != nil {
		if *subnet.Properties.RouteTable.ID != *routeTable.ID {
			return fmt.Errorf("The subnet has a route table, but it was unrecognized. Refusing to modify it. active_routetable=%q expected_routetable=%q", *subnet.Properties.RouteTable.ID, *routeTable.ID)
		}
	} else {
		subnet.Properties.RouteTable = &network.RouteTable{
			ID: routeTable.ID,
		}
		glog.V(3).Info("create: updating subnet")
		_, err := az.SubnetsClient.CreateOrUpdate(az.ResourceGroup, az.VnetName, az.SubnetName, subnet, nil)
		if err != nil {
			return err
		}
	}

	targetIP, err := az.getIPForMachine(kubeRoute.TargetInstance)
	if err != nil {
		return err
	}

	routeName := getRouteName(kubeRoute.TargetInstance)
	route := network.Route{
		Name: to.StringPtr(routeName),
		Properties: &network.RoutePropertiesFormat{
			AddressPrefix:    to.StringPtr(kubeRoute.DestinationCIDR),
			NextHopType:      network.RouteNextHopTypeVirtualAppliance,
			NextHopIPAddress: to.StringPtr(targetIP),
		},
	}

	glog.V(3).Infof("create: creating route: instance=%q cidr=%q", kubeRoute.TargetInstance, kubeRoute.DestinationCIDR)
	_, err = az.RoutesClient.CreateOrUpdate(az.ResourceGroup, az.RouteTableName, *route.Name, route, nil)
	if err != nil {
		return err
	}

	glog.V(2).Infof("create: route created. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetInstance, kubeRoute.DestinationCIDR)
	return nil
}

// DeleteRoute deletes the specified managed route
// Route should be as returned by ListRoutes
func (az *Cloud) DeleteRoute(clusterName string, kubeRoute *cloudprovider.Route) error {
	glog.V(2).Infof("delete: deleting route. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetInstance, kubeRoute.DestinationCIDR)

	routeName := getRouteName(kubeRoute.TargetInstance)
	_, err := az.RoutesClient.Delete(az.ResourceGroup, az.RouteTableName, routeName, nil)
	if err != nil {
		return err
	}

	glog.V(2).Infof("delete: route deleted. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetInstance, kubeRoute.DestinationCIDR)
	return nil
}

// This must be kept in sync with getInstanceName.
// These two functions enable stashing the instance name in the route
// and then retrieving it later when listing. This is needed because
// Azure does not let you put tags/descriptions on the Route itself.
func getRouteName(instanceName string) string {
	return fmt.Sprintf("%s", instanceName)
}

// Used with getRouteName. See comment on getRouteName.
func getInstanceName(routeName string) string {
	return fmt.Sprintf("%s", routeName)
}
