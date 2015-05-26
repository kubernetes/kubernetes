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

package routecontroller

import (
	"fmt"
	"net"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

type RouteController struct {
	routes      cloudprovider.Routes
	kubeClient  client.Interface
	clusterName string
	clusterCIDR *net.IPNet
}

const k8sNodeRouteTag = "k8s-node-route"

func New(routes cloudprovider.Routes, kubeClient client.Interface, clusterName string, clusterCIDR *net.IPNet) *RouteController {
	return &RouteController{
		routes:      routes,
		kubeClient:  kubeClient,
		clusterName: clusterName,
		clusterCIDR: clusterCIDR,
	}
}

func (rc *RouteController) Run(syncPeriod time.Duration) {
	go util.Forever(func() {
		if err := rc.reconcileNodeRoutes(); err != nil {
			glog.Errorf("Couldn't reconcile node routes: %v", err)
		}
	}, syncPeriod)
}

func (rc *RouteController) reconcileNodeRoutes() error {
	routeList, err := rc.routes.ListRoutes(rc.truncatedClusterName() + "-.*")
	if err != nil {
		return fmt.Errorf("error listing routes: %v", err)
	}
	// TODO (cjcullen): use pkg/controller/framework.NewInformer to watch this
	// and reduce the number of lists needed.
	nodeList, err := rc.kubeClient.Nodes().List(labels.Everything(), fields.Everything())
	if err != nil {
		return fmt.Errorf("error listing nodes: %v", err)
	}
	return rc.reconcile(nodeList.Items, routeList)
}

func (rc *RouteController) reconcile(nodes []api.Node, routes []*cloudprovider.Route) error {
	// nodeCIDRs maps nodeName->nodeCIDR
	nodeCIDRs := make(map[string]string)
	// routeMap maps routeTargetInstance->route
	routeMap := make(map[string]*cloudprovider.Route)
	for _, route := range routes {
		routeMap[route.TargetInstance] = route
	}
	for _, node := range nodes {
		// Check if we have a route for this node w/ the correct CIDR.
		r := routeMap[node.Name]
		if r == nil || r.DestinationCIDR != node.Spec.PodCIDR {
			// If not, create the route.
			route := &cloudprovider.Route{
				Name:            rc.truncatedClusterName() + "-" + string(node.UID),
				TargetInstance:  node.Name,
				DestinationCIDR: node.Spec.PodCIDR,
				Description:     k8sNodeRouteTag,
			}
			go func(route *cloudprovider.Route) {
				if err := rc.routes.CreateRoute(route); err != nil {
					glog.Errorf("Could not create route %s: %v", route.Name, err)
				}
			}(route)
		}
		nodeCIDRs[node.Name] = node.Spec.PodCIDR
	}
	for _, route := range routes {
		if rc.isResponsibleForRoute(route) {
			// Check if this route applies to a node we know about & has correct CIDR.
			if nodeCIDRs[route.TargetInstance] != route.DestinationCIDR {
				// Delete the route.
				go func(routeName string) {
					if err := rc.routes.DeleteRoute(routeName); err != nil {
						glog.Errorf("Could not delete route %s: %v", routeName, err)
					}
				}(route.Name)
			}
		}
	}
	return nil
}

func (rc *RouteController) truncatedClusterName() string {
	if len(rc.clusterName) > 26 {
		return rc.clusterName[:26]
	}
	return rc.clusterName
}

func (rc *RouteController) isResponsibleForRoute(route *cloudprovider.Route) bool {
	_, cidr, err := net.ParseCIDR(route.DestinationCIDR)
	if err != nil {
		glog.Errorf("Ignoring route %s, unparsable CIDR: %v", route.Name, err)
		return false
	}
	// Not responsible if this route's CIDR is not within our clusterCIDR
	lastIP := make([]byte, len(cidr.IP))
	for i := range lastIP {
		lastIP[i] = cidr.IP[i] | ^cidr.Mask[i]
	}
	if !rc.clusterCIDR.Contains(cidr.IP) || !rc.clusterCIDR.Contains(lastIP) {
		return false
	}
	// Not responsible if route name doesn't start with <clusterName>
	if !strings.HasPrefix(route.Name, rc.clusterName) {
		return false
	}
	// Not responsible if route description != "k8s-node-route"
	if route.Description != k8sNodeRouteTag {
		return false
	}
	return true
}
