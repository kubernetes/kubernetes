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
	"k8s.io/apimachinery/pkg/types"
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
	if routeTable.Routes != nil {
		kubeRoutes = make([]*cloudprovider.Route, len(*routeTable.Routes))
		for i, route := range *routeTable.Routes {
			instance := mapRouteNameToNodeName(*route.Name)
			cidr := *route.AddressPrefix
			glog.V(10).Infof("list: * instance=%q, cidr=%q", instance, cidr)

			kubeRoutes[i] = &cloudprovider.Route{
				Name:            *route.Name,
				TargetNode:      instance,
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
	glog.V(2).Infof("create: creating route. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)

	routeTable, existsRouteTable, err := az.getRouteTable()
	if err != nil {
		glog.V(2).Infof("create error: couldn't get routetable. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
		return err
	}
	if !existsRouteTable {
		routeTable = network.RouteTable{
			Name:                       to.StringPtr(az.RouteTableName),
			Location:                   to.StringPtr(az.Location),
			RouteTablePropertiesFormat: &network.RouteTablePropertiesFormat{},
		}

		glog.V(3).Infof("create: creating routetable. routeTableName=%q", az.RouteTableName)
		az.operationPollRateLimiter.Accept()
		respChan, errChan := az.RouteTablesClient.CreateOrUpdate(az.ResourceGroup, az.RouteTableName, routeTable, nil)
		resp := <-respChan
		err := <-errChan
		if az.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
			glog.V(2).Infof("create backing off: creating routetable. routeTableName=%q", az.RouteTableName)
			retryErr := az.CreateOrUpdateRouteTableWithRetry(routeTable)
			if retryErr != nil {
				err = retryErr
				glog.V(2).Infof("create abort backoff: creating routetable. routeTableName=%q", az.RouteTableName)
			}
		}
		if err != nil {
			return err
		}

		routeTable, err = az.RouteTablesClient.Get(az.ResourceGroup, az.RouteTableName, "")
		if err != nil {
			return err
		}
	}

	targetIP, err := az.getIPForMachine(kubeRoute.TargetNode)
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

	glog.V(3).Infof("create: creating route: instance=%q cidr=%q", kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
	az.operationPollRateLimiter.Accept()
	respChan, errChan := az.RoutesClient.CreateOrUpdate(az.ResourceGroup, az.RouteTableName, *route.Name, route, nil)
	resp := <-respChan
	err = <-errChan
	if az.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
		glog.V(2).Infof("create backing off: creating route: instance=%q cidr=%q", kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
		retryErr := az.CreateOrUpdateRouteWithRetry(route)
		if retryErr != nil {
			err = retryErr
			glog.V(2).Infof("create abort backoff: creating route: instance=%q cidr=%q", kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
		}
	}
	if err != nil {
		return err
	}

	glog.V(2).Infof("create: route created. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
	return nil
}

// DeleteRoute deletes the specified managed route
// Route should be as returned by ListRoutes
func (az *Cloud) DeleteRoute(clusterName string, kubeRoute *cloudprovider.Route) error {
	glog.V(2).Infof("delete: deleting route. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)

	routeName := mapNodeNameToRouteName(kubeRoute.TargetNode)
	az.operationPollRateLimiter.Accept()
	respChan, errChan := az.RoutesClient.Delete(az.ResourceGroup, az.RouteTableName, routeName, nil)
	resp := <-respChan
	err := <-errChan

	if az.CloudProviderBackoff && shouldRetryAPIRequest(resp, err) {
		glog.V(2).Infof("delete backing off: deleting route. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
		retryErr := az.DeleteRouteWithRetry(routeName)
		if retryErr != nil {
			err = retryErr
			glog.V(2).Infof("delete abort backoff: deleting route. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
		}
	}
	if err != nil {
		return err
	}

	glog.V(2).Infof("delete: route deleted. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
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
