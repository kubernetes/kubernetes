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
	"strings"

	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-08-01/network"
	"github.com/Azure/go-autorest/autorest/to"

	"k8s.io/apimachinery/pkg/types"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/klog"
	utilnet "k8s.io/utils/net"

	// Azure route controller changes behavior if ipv6dual stack feature is turned on
	// remove this once the feature graduates
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
)

// copied to minimize the number of cross reference
// and exceptions in publishing and allowed imports.
const (
	IPv6DualStack      featuregate.Feature = "IPv6DualStack"
	routeNameFmt                           = "%s____%s"
	routeNameSeparator                     = "____"
)

// ListRoutes lists all managed routes that belong to the specified clusterName
func (az *Cloud) ListRoutes(ctx context.Context, clusterName string) ([]*cloudprovider.Route, error) {
	klog.V(10).Infof("ListRoutes: START clusterName=%q", clusterName)
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
			klog.V(10).Infof("ListRoutes: * instance=%q, cidr=%q", instance, cidr)

			kubeRoutes[i] = &cloudprovider.Route{
				Name:            *route.Name,
				TargetNode:      instance,
				DestinationCIDR: cidr,
			}
		}
	}

	klog.V(10).Info("ListRoutes: FINISH")
	return kubeRoutes, nil
}

func (az *Cloud) createRouteTableIfNotExists(clusterName string, kubeRoute *cloudprovider.Route) error {
	if _, existsRouteTable, err := az.getRouteTable(); err != nil {
		klog.V(2).Infof("createRouteTableIfNotExists error: couldn't get routetable. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
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

	klog.V(3).Infof("createRouteTableIfNotExists: creating routetable. routeTableName=%q", az.RouteTableName)
	err := az.CreateOrUpdateRouteTable(routeTable)
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
	var targetIP string
	nodeName := string(kubeRoute.TargetNode)
	unmanaged, err := az.IsNodeUnmanaged(nodeName)
	if err != nil {
		return err
	}
	if unmanaged {
		if utilfeature.DefaultFeatureGate.Enabled(IPv6DualStack) {
			//TODO (khenidak) add support for unmanaged nodes when the feature reaches  beta
			return fmt.Errorf("unmanaged nodes are not supported in dual stack mode")
		}
		klog.V(2).Infof("CreateRoute: omitting unmanaged node %q", kubeRoute.TargetNode)
		az.routeCIDRsLock.Lock()
		defer az.routeCIDRsLock.Unlock()
		az.routeCIDRs[nodeName] = kubeRoute.DestinationCIDR
		return nil
	}

	klog.V(2).Infof("CreateRoute: creating route. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
	if err := az.createRouteTableIfNotExists(clusterName, kubeRoute); err != nil {
		return err
	}
	if !utilfeature.DefaultFeatureGate.Enabled(IPv6DualStack) {
		targetIP, _, err = az.getIPForMachine(kubeRoute.TargetNode)
		if err != nil {
			return err
		}
	} else {
		// for dual stack we need to select
		// a private ip that matches family of the cidr
		klog.V(4).Infof("CreateRoute: create route instance=%q cidr=%q is in dual stack mode", kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
		CIDRv6 := utilnet.IsIPv6CIDRString(string(kubeRoute.DestinationCIDR))
		nodePrivateIPs, err := az.getPrivateIPsForMachine(kubeRoute.TargetNode)
		if nil != err {
			klog.V(3).Infof("CreateRoute: create route: failed(GetPrivateIPsByNodeName) instance=%q cidr=%q with error=%v", kubeRoute.TargetNode, kubeRoute.DestinationCIDR, err)
			return err
		}

		targetIP, err = findFirstIPByFamily(nodePrivateIPs, CIDRv6)
		if nil != err {
			klog.V(3).Infof("CreateRoute: create route: failed(findFirstIpByFamily) instance=%q cidr=%q with error=%v", kubeRoute.TargetNode, kubeRoute.DestinationCIDR, err)
			return err
		}
	}
	routeName := mapNodeNameToRouteName(kubeRoute.TargetNode, string(kubeRoute.DestinationCIDR))
	route := network.Route{
		Name: to.StringPtr(routeName),
		RoutePropertiesFormat: &network.RoutePropertiesFormat{
			AddressPrefix:    to.StringPtr(kubeRoute.DestinationCIDR),
			NextHopType:      network.RouteNextHopTypeVirtualAppliance,
			NextHopIPAddress: to.StringPtr(targetIP),
		},
	}

	klog.V(3).Infof("CreateRoute: creating route: instance=%q cidr=%q", kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
	err = az.CreateOrUpdateRoute(route)
	if err != nil {
		return err
	}

	klog.V(2).Infof("CreateRoute: route created. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
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
		klog.V(2).Infof("DeleteRoute: omitting unmanaged node %q", kubeRoute.TargetNode)
		az.routeCIDRsLock.Lock()
		defer az.routeCIDRsLock.Unlock()
		delete(az.routeCIDRs, nodeName)
		return nil
	}

	klog.V(2).Infof("DeleteRoute: deleting route. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)

	routeName := mapNodeNameToRouteName(kubeRoute.TargetNode, string(kubeRoute.DestinationCIDR))
	err = az.DeleteRouteWithName(routeName)
	if err != nil {
		return err
	}

	klog.V(2).Infof("DeleteRoute: route deleted. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
	return nil
}

// This must be kept in sync with mapRouteNameToNodeName.
// These two functions enable stashing the instance name in the route
// and then retrieving it later when listing. This is needed because
// Azure does not let you put tags/descriptions on the Route itself.
func mapNodeNameToRouteName(nodeName types.NodeName, cidr string) string {
	if !utilfeature.DefaultFeatureGate.Enabled(IPv6DualStack) {
		return fmt.Sprintf("%s", nodeName)
	}
	return fmt.Sprintf(routeNameFmt, nodeName, cidrtoRfc1035(cidr))
}

// Used with mapNodeNameToRouteName. See comment on mapNodeNameToRouteName.
func mapRouteNameToNodeName(routeName string) types.NodeName {
	if !utilfeature.DefaultFeatureGate.Enabled(IPv6DualStack) {
		return types.NodeName(fmt.Sprintf("%s", routeName))
	}
	parts := strings.Split(routeName, routeNameSeparator)
	nodeName := parts[0]
	return types.NodeName(nodeName)

}

// given a list of ips, return the first one
// that matches the family requested
// error if no match, or failure to parse
// any of the ips
func findFirstIPByFamily(ips []string, v6 bool) (string, error) {
	for _, ip := range ips {
		bIPv6 := utilnet.IsIPv6String(ip)
		if v6 == bIPv6 {
			return ip, nil
		}
	}
	return "", fmt.Errorf("no match found matching the ipfamily requested")
}

//strips : . /
func cidrtoRfc1035(cidr string) string {
	cidr = strings.ReplaceAll(cidr, ":", "")
	cidr = strings.ReplaceAll(cidr, ".", "")
	cidr = strings.ReplaceAll(cidr, "/", "")
	return cidr
}
