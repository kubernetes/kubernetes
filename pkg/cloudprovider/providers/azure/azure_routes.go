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
	"context"
	"fmt"

	"k8s.io/kubernetes/pkg/cloudprovider"

	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-09-01/network"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
)

// ListRoutes lists all managed routes that belong to the specified clusterName
func (az *Cloud) ListRoutes(ctx context.Context, clusterName string) ([]*cloudprovider.Route, error) {
	glog.V(10).Infof("ListRoutes: START clusterName=%q", clusterName)
	routeTable, existsRouteTable, err := az.getRouteTable()
	routes, err := processRoutes(routeTable, existsRouteTable, err)
	if err != nil {
		return nil, err
	}

	// Compose routes for unmanaged routes so that node controller won't retry creating routes for them.
	unmanagedNodes, err := az.GetUnmanagedNodes()
	if err != nil {
		return nil, err
	}
	az.routeCIDRsLock.Lock()
	defer az.routeCIDRsLock.Unlock()
	for _, nodeName := range unmanagedNodes.List() {
		if cidr, ok := az.routeCIDRs[nodeName]; ok {
			routes = append(routes, &cloudprovider.Route{
				Name:            nodeName,
				TargetNode:      mapRouteNameToNodeName(nodeName),
				DestinationCIDR: cidr,
			})
		}
	}

	return routes, nil
}

// Injectable for testing
func processRoutes(routeTable network.RouteTable, exists bool, err error) ([]*cloudprovider.Route, error) {
	if err != nil {
		return nil, err
	}
	if !exists {
		return []*cloudprovider.Route{}, nil
	}

	var kubeRoutes []*cloudprovider.Route
	if routeTable.RouteTablePropertiesFormat != nil && routeTable.Routes != nil {
		kubeRoutes = make([]*cloudprovider.Route, len(*routeTable.Routes))
		for i, route := range *routeTable.Routes {
			instance := mapRouteNameToNodeName(*route.Name)
			cidr := *route.AddressPrefix
			glog.V(10).Infof("ListRoutes: * instance=%q, cidr=%q", instance, cidr)

			kubeRoutes[i] = &cloudprovider.Route{
				Name:            *route.Name,
				TargetNode:      instance,
				DestinationCIDR: cidr,
			}
		}
	}

	glog.V(10).Info("ListRoutes: FINISH")
	return kubeRoutes, nil
}

func (az *Cloud) createRouteTableIfNotExists(clusterName string, kubeRoute *cloudprovider.Route) error {
	if _, existsRouteTable, err := az.getRouteTable(); err != nil {
		glog.V(2).Infof("createRouteTableIfNotExists error: couldn't get routetable. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
		return err
	} else if existsRouteTable {
		return nil
	}
	return az.createRouteTable()
}

func (az *Cloud) createRouteTable() error {
	routeTable := network.RouteTable{
		Name:                       to.StringPtr(az.RouteTableName),
		Location:                   to.StringPtr(az.Location),
		RouteTablePropertiesFormat: &network.RouteTablePropertiesFormat{},
	}

	glog.V(3).Infof("createRouteTableIfNotExists: creating routetable. routeTableName=%q", az.RouteTableName)
	ctx, cancel := getContextWithCancel()
	defer cancel()
	resp, err := az.RouteTablesClient.CreateOrUpdate(ctx, az.ResourceGroup, az.RouteTableName, routeTable)
	glog.V(10).Infof("RouteTablesClient.CreateOrUpdate(%q): end", az.RouteTableName)
	if az.CloudProviderBackoff && shouldRetryHTTPRequest(resp, err) {
		glog.V(2).Infof("createRouteTableIfNotExists backing off: creating routetable. routeTableName=%q", az.RouteTableName)
		retryErr := az.CreateOrUpdateRouteTableWithRetry(routeTable)
		if retryErr != nil {
			err = retryErr
			glog.V(2).Infof("createRouteTableIfNotExists abort backoff: creating routetable. routeTableName=%q", az.RouteTableName)
		}
	}
	if err != nil {
		return err
	}

	// Invalidate the cache right after updating
	az.rtCache.Delete(az.RouteTableName)
	return nil
}

// CreateRoute creates the described managed route
// route.Name will be ignored, although the cloud-provider may use nameHint
// to create a more user-meaningful name.
func (az *Cloud) CreateRoute(ctx context.Context, clusterName string, nameHint string, kubeRoute *cloudprovider.Route) error {
	// Returns  for unmanaged nodes because azure cloud provider couldn't fetch information for them.
	nodeName := string(kubeRoute.TargetNode)
	unmanaged, err := az.IsNodeUnmanaged(nodeName)
	if err != nil {
		return err
	}
	if unmanaged {
		glog.V(2).Infof("CreateRoute: omitting unmanaged node %q", kubeRoute.TargetNode)
		az.routeCIDRsLock.Lock()
		defer az.routeCIDRsLock.Unlock()
		az.routeCIDRs[nodeName] = kubeRoute.DestinationCIDR
		return nil
	}

	glog.V(2).Infof("CreateRoute: creating route. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
	if err := az.createRouteTableIfNotExists(clusterName, kubeRoute); err != nil {
		return err
	}
	targetIP, _, err := az.getIPForMachine(kubeRoute.TargetNode)
	if err != nil {
		return err
	}

	routeName := mapNodeNameToRouteName(kubeRoute.TargetNode)
	route := network.Route{
		Name: to.StringPtr(routeName),
		RoutePropertiesFormat: &network.RoutePropertiesFormat{
			AddressPrefix:    to.StringPtr(kubeRoute.DestinationCIDR),
			NextHopType:      network.RouteNextHopTypeVirtualAppliance,
			NextHopIPAddress: to.StringPtr(targetIP),
		},
	}

	glog.V(3).Infof("CreateRoute: creating route: instance=%q cidr=%q", kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
	ctx, cancel := getContextWithCancel()
	defer cancel()
	resp, err := az.RoutesClient.CreateOrUpdate(ctx, az.ResourceGroup, az.RouteTableName, *route.Name, route)
	glog.V(10).Infof("RoutesClient.CreateOrUpdate(%q): end", az.RouteTableName)
	if az.CloudProviderBackoff && shouldRetryHTTPRequest(resp, err) {
		glog.V(2).Infof("CreateRoute backing off: creating route: instance=%q cidr=%q", kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
		retryErr := az.CreateOrUpdateRouteWithRetry(route)
		if retryErr != nil {
			err = retryErr
			glog.V(2).Infof("CreateRoute abort backoff: creating route: instance=%q cidr=%q", kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
		}
	}
	if err != nil {
		return err
	}

	glog.V(2).Infof("CreateRoute: route created. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
	return nil
}

// DeleteRoute deletes the specified managed route
// Route should be as returned by ListRoutes
func (az *Cloud) DeleteRoute(ctx context.Context, clusterName string, kubeRoute *cloudprovider.Route) error {
	// Returns  for unmanaged nodes because azure cloud provider couldn't fetch information for them.
	nodeName := string(kubeRoute.TargetNode)
	unmanaged, err := az.IsNodeUnmanaged(nodeName)
	if err != nil {
		return err
	}
	if unmanaged {
		glog.V(2).Infof("DeleteRoute: omitting unmanaged node %q", kubeRoute.TargetNode)
		az.routeCIDRsLock.Lock()
		defer az.routeCIDRsLock.Unlock()
		delete(az.routeCIDRs, nodeName)
		return nil
	}

	glog.V(2).Infof("DeleteRoute: deleting route. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)

	ctx, cancel := getContextWithCancel()
	defer cancel()
	routeName := mapNodeNameToRouteName(kubeRoute.TargetNode)
	resp, err := az.RoutesClient.Delete(ctx, az.ResourceGroup, az.RouteTableName, routeName)
	glog.V(10).Infof("RoutesClient.Delete(%q): end", az.RouteTableName)

	if az.CloudProviderBackoff && shouldRetryHTTPRequest(resp, err) {
		glog.V(2).Infof("DeleteRoute backing off: deleting route. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
		retryErr := az.DeleteRouteWithRetry(routeName)
		if retryErr != nil {
			err = retryErr
			glog.V(2).Infof("DeleteRoute abort backoff: deleting route. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
		}
	}
	if err != nil {
		return err
	}

	glog.V(2).Infof("DeleteRoute: route deleted. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
	return nil
}

// This must be kept in sync with mapRouteNameToNodeName.
// These two functions enable stashing the instance name in the route
// and then retrieving it later when listing. This is needed because
// Azure does not let you put tags/descriptions on the Route itself.
func mapNodeNameToRouteName(nodeName types.NodeName) string {
	return fmt.Sprintf("%s", nodeName)
}

// Used with mapNodeNameToRouteName. See comment on mapNodeNameToRouteName.
func mapRouteNameToNodeName(routeName string) types.NodeName {
	return types.NodeName(fmt.Sprintf("%s", routeName))
}
