// +build !providerless

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
	"sync"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/go-autorest/autorest/to"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/klog/v2"
	azcache "k8s.io/legacy-cloud-providers/azure/cache"
	"k8s.io/legacy-cloud-providers/azure/metrics"
	utilnet "k8s.io/utils/net"
)

var (
	// routeUpdateInterval defines the route reconciling interval.
	routeUpdateInterval = 30 * time.Second
)

// routeOperation defines the allowed operations for route updating.
type routeOperation string

// copied to minimize the number of cross reference
// and exceptions in publishing and allowed imports.
const (
	routeNameFmt       = "%s____%s"
	routeNameSeparator = "____"

	// Route operations.
	routeOperationAdd             routeOperation = "add"
	routeOperationDelete          routeOperation = "delete"
	routeTableOperationUpdateTags routeOperation = "updateRouteTableTags"
)

// delayedRouteOperation defines a delayed route operation which is used in delayedRouteUpdater.
type delayedRouteOperation struct {
	route          network.Route
	routeTableTags map[string]*string
	operation      routeOperation
	result         chan error
}

// wait waits for the operation completion and returns the result.
func (op *delayedRouteOperation) wait() error {
	return <-op.result
}

// delayedRouteUpdater defines a delayed route updater, which batches all the
// route updating operations within "interval" period.
// Example usage:
//   op, err := updater.addRouteOperation(routeOperationAdd, route)
//   err = op.wait()
type delayedRouteUpdater struct {
	az       *Cloud
	interval time.Duration

	lock           sync.Mutex
	routesToUpdate []*delayedRouteOperation
}

// newDelayedRouteUpdater creates a new delayedRouteUpdater.
func newDelayedRouteUpdater(az *Cloud, interval time.Duration) *delayedRouteUpdater {
	return &delayedRouteUpdater{
		az:             az,
		interval:       interval,
		routesToUpdate: make([]*delayedRouteOperation, 0),
	}
}

// run starts the updater reconciling loop.
func (d *delayedRouteUpdater) run() {
	err := wait.PollImmediateInfinite(d.interval, func() (bool, error) {
		d.updateRoutes()
		return false, nil
	})
	if err != nil { // this should never happen, if it does, panic
		panic(err)
	}
}

// updateRoutes invokes route table client to update all routes.
func (d *delayedRouteUpdater) updateRoutes() {
	d.lock.Lock()
	defer d.lock.Unlock()

	// No need to do any updating.
	if len(d.routesToUpdate) == 0 {
		return
	}

	var err error
	defer func() {
		// Notify all the goroutines.
		for _, rt := range d.routesToUpdate {
			rt.result <- err
		}
		// Clear all the jobs.
		d.routesToUpdate = make([]*delayedRouteOperation, 0)
	}()

	var (
		routeTable       network.RouteTable
		existsRouteTable bool
	)
	routeTable, existsRouteTable, err = d.az.getRouteTable(azcache.CacheReadTypeDefault)
	if err != nil {
		klog.Errorf("getRouteTable() failed with error: %v", err)
		return
	}

	// create route table if it doesn't exists yet.
	if !existsRouteTable {
		err = d.az.createRouteTable()
		if err != nil {
			klog.Errorf("createRouteTable() failed with error: %v", err)
			return
		}

		routeTable, _, err = d.az.getRouteTable(azcache.CacheReadTypeDefault)
		if err != nil {
			klog.Errorf("getRouteTable() failed with error: %v", err)
			return
		}
	}

	// reconcile routes.
	dirty := false
	routes := []network.Route{}
	if routeTable.Routes != nil {
		routes = *routeTable.Routes
	}
	onlyUpdateTags := true
	for _, rt := range d.routesToUpdate {
		if rt.operation == routeTableOperationUpdateTags {
			routeTable.Tags = rt.routeTableTags
			dirty = true
			continue
		}

		routeMatch := false
		onlyUpdateTags = false
		for i, existingRoute := range routes {
			if strings.EqualFold(to.String(existingRoute.Name), to.String(rt.route.Name)) {
				// delete the name-matched routes here (missing routes would be added later if the operation is add).
				routes = append(routes[:i], routes[i+1:]...)
				if existingRoute.RoutePropertiesFormat != nil &&
					rt.route.RoutePropertiesFormat != nil &&
					strings.EqualFold(to.String(existingRoute.AddressPrefix), to.String(rt.route.AddressPrefix)) &&
					strings.EqualFold(to.String(existingRoute.NextHopIPAddress), to.String(rt.route.NextHopIPAddress)) {
					routeMatch = true
				}
				if rt.operation == routeOperationDelete {
					dirty = true
				}
				break
			}
		}

		// Add missing routes if the operation is add.
		if rt.operation == routeOperationAdd {
			routes = append(routes, rt.route)
			if !routeMatch {
				dirty = true
			}
			continue
		}
	}

	if dirty {
		if !onlyUpdateTags {
			klog.V(2).Infof("updateRoutes: updating routes")
			routeTable.Routes = &routes
		}
		err = d.az.CreateOrUpdateRouteTable(routeTable)
		if err != nil {
			klog.Errorf("CreateOrUpdateRouteTable() failed with error: %v", err)
			return
		}
	}
}

// addRouteOperation adds the routeOperation to delayedRouteUpdater and returns a delayedRouteOperation.
func (d *delayedRouteUpdater) addRouteOperation(operation routeOperation, route network.Route) (*delayedRouteOperation, error) {
	d.lock.Lock()
	defer d.lock.Unlock()

	op := &delayedRouteOperation{
		route:     route,
		operation: operation,
		result:    make(chan error),
	}
	d.routesToUpdate = append(d.routesToUpdate, op)
	return op, nil
}

// addUpdateRouteTableTagsOperation adds a update route table tags operation to delayedRouteUpdater and returns a delayedRouteOperation.
func (d *delayedRouteUpdater) addUpdateRouteTableTagsOperation(operation routeOperation, tags map[string]*string) (*delayedRouteOperation, error) {
	d.lock.Lock()
	defer d.lock.Unlock()

	op := &delayedRouteOperation{
		routeTableTags: tags,
		operation:      operation,
		result:         make(chan error),
	}
	d.routesToUpdate = append(d.routesToUpdate, op)
	return op, nil
}

// ListRoutes lists all managed routes that belong to the specified clusterName
func (az *Cloud) ListRoutes(ctx context.Context, clusterName string) ([]*cloudprovider.Route, error) {
	klog.V(10).Infof("ListRoutes: START clusterName=%q", clusterName)
	routeTable, existsRouteTable, err := az.getRouteTable(azcache.CacheReadTypeDefault)
	routes, err := processRoutes(az.ipv6DualStackEnabled, routeTable, existsRouteTable, err)
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
				TargetNode:      mapRouteNameToNodeName(az.ipv6DualStackEnabled, nodeName),
				DestinationCIDR: cidr,
			})
		}
	}

	// ensure the route table is tagged as configured
	tags, changed := az.ensureRouteTableTagged(&routeTable)
	if changed {
		klog.V(2).Infof("ListRoutes: updating tags on route table %s", to.String(routeTable.Name))
		op, err := az.routeUpdater.addUpdateRouteTableTagsOperation(routeTableOperationUpdateTags, tags)
		if err != nil {
			klog.Errorf("ListRoutes: failed to add route table operation with error: %v", err)
			return nil, err
		}

		// Wait for operation complete.
		err = op.wait()
		if err != nil {
			klog.Errorf("ListRoutes: failed to update route table tags with error: %v", err)
			return nil, err
		}
	}

	return routes, nil
}

// Injectable for testing
func processRoutes(ipv6DualStackEnabled bool, routeTable network.RouteTable, exists bool, err error) ([]*cloudprovider.Route, error) {
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
			instance := mapRouteNameToNodeName(ipv6DualStackEnabled, *route.Name)
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
	mc := metrics.NewMetricContext("routes", "create_route", az.ResourceGroup, az.SubscriptionID, "")
	isOperationSucceeded := false
	defer func() {
		mc.ObserveOperationWithResult(isOperationSucceeded)
	}()

	// Returns  for unmanaged nodes because azure cloud provider couldn't fetch information for them.
	var targetIP string
	nodeName := string(kubeRoute.TargetNode)
	unmanaged, err := az.IsNodeUnmanaged(nodeName)
	if err != nil {
		return err
	}
	if unmanaged {
		if az.ipv6DualStackEnabled {
			//TODO (khenidak) add support for unmanaged nodes when the feature reaches beta
			return fmt.Errorf("unmanaged nodes are not supported in dual stack mode")
		}
		klog.V(2).Infof("CreateRoute: omitting unmanaged node %q", kubeRoute.TargetNode)
		az.routeCIDRsLock.Lock()
		defer az.routeCIDRsLock.Unlock()
		az.routeCIDRs[nodeName] = kubeRoute.DestinationCIDR
		return nil
	}

	CIDRv6 := utilnet.IsIPv6CIDRString(string(kubeRoute.DestinationCIDR))
	// if single stack IPv4 then get the IP for the primary ip config
	// single stack IPv6 is supported on dual stack host. So the IPv6 IP is secondary IP for both single stack IPv6 and dual stack
	// Get all private IPs for the machine and find the first one that matches the IPv6 family
	if !az.ipv6DualStackEnabled && !CIDRv6 {
		targetIP, _, err = az.getIPForMachine(kubeRoute.TargetNode)
		if err != nil {
			return err
		}
	} else {
		// for dual stack and single stack IPv6 we need to select
		// a private ip that matches family of the cidr
		klog.V(4).Infof("CreateRoute: create route instance=%q cidr=%q is in dual stack mode", kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
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
	routeName := mapNodeNameToRouteName(az.ipv6DualStackEnabled, kubeRoute.TargetNode, string(kubeRoute.DestinationCIDR))
	route := network.Route{
		Name: to.StringPtr(routeName),
		RoutePropertiesFormat: &network.RoutePropertiesFormat{
			AddressPrefix:    to.StringPtr(kubeRoute.DestinationCIDR),
			NextHopType:      network.RouteNextHopTypeVirtualAppliance,
			NextHopIPAddress: to.StringPtr(targetIP),
		},
	}

	klog.V(2).Infof("CreateRoute: creating route for clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
	op, err := az.routeUpdater.addRouteOperation(routeOperationAdd, route)
	if err != nil {
		klog.Errorf("CreateRoute failed for node %q with error: %v", kubeRoute.TargetNode, err)
		return err
	}

	// Wait for operation complete.
	err = op.wait()
	if err != nil {
		klog.Errorf("CreateRoute failed for node %q with error: %v", kubeRoute.TargetNode, err)
		return err
	}

	klog.V(2).Infof("CreateRoute: route created. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
	isOperationSucceeded = true

	return nil
}

// DeleteRoute deletes the specified managed route
// Route should be as returned by ListRoutes
func (az *Cloud) DeleteRoute(ctx context.Context, clusterName string, kubeRoute *cloudprovider.Route) error {
	mc := metrics.NewMetricContext("routes", "delete_route", az.ResourceGroup, az.SubscriptionID, "")
	isOperationSucceeded := false
	defer func() {
		mc.ObserveOperationWithResult(isOperationSucceeded)
	}()

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

	routeName := mapNodeNameToRouteName(az.ipv6DualStackEnabled, kubeRoute.TargetNode, string(kubeRoute.DestinationCIDR))
	route := network.Route{
		Name:                  to.StringPtr(routeName),
		RoutePropertiesFormat: &network.RoutePropertiesFormat{},
	}
	op, err := az.routeUpdater.addRouteOperation(routeOperationDelete, route)
	if err != nil {
		klog.Errorf("DeleteRoute failed for node %q with error: %v", kubeRoute.TargetNode, err)
		return err
	}

	// Wait for operation complete.
	err = op.wait()
	if err != nil {
		klog.Errorf("DeleteRoute failed for node %q with error: %v", kubeRoute.TargetNode, err)
		return err
	}

	klog.V(2).Infof("DeleteRoute: route deleted. clusterName=%q instance=%q cidr=%q", clusterName, kubeRoute.TargetNode, kubeRoute.DestinationCIDR)
	isOperationSucceeded = true

	return nil
}

// This must be kept in sync with mapRouteNameToNodeName.
// These two functions enable stashing the instance name in the route
// and then retrieving it later when listing. This is needed because
// Azure does not let you put tags/descriptions on the Route itself.
func mapNodeNameToRouteName(ipv6DualStackEnabled bool, nodeName types.NodeName, cidr string) string {
	if !ipv6DualStackEnabled {
		return fmt.Sprintf("%s", nodeName)
	}
	return fmt.Sprintf(routeNameFmt, nodeName, cidrtoRfc1035(cidr))
}

// Used with mapNodeNameToRouteName. See comment on mapNodeNameToRouteName.
func mapRouteNameToNodeName(ipv6DualStackEnabled bool, routeName string) types.NodeName {
	if !ipv6DualStackEnabled {
		return types.NodeName(routeName)
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

//  ensureRouteTableTagged ensures the route table is tagged as configured
func (az *Cloud) ensureRouteTableTagged(rt *network.RouteTable) (map[string]*string, bool) {
	if az.Tags == "" {
		return nil, false
	}
	changed := false
	tags := parseTags(az.Tags)
	if rt.Tags == nil {
		rt.Tags = make(map[string]*string)
	}
	for k, v := range tags {
		if vv, ok := rt.Tags[k]; !ok || !strings.EqualFold(to.String(v), to.String(vv)) {
			rt.Tags[k] = v
			changed = true
		}
	}
	return rt.Tags, changed
}
