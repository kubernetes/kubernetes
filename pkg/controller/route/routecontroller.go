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

package route

import (
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/util/metrics"
	"k8s.io/kubernetes/pkg/util/wait"
)

const (
	// Maximal number of concurrent CreateRoute API calls.
	// TODO: This should be per-provider.
	maxConcurrentRouteCreations int = 200
	// Maximum number of retries of route creations.
	maxRetries int = 5
	// Maximum number of retries of node status update.
	updateNodeStatusMaxRetries int = 3
)

type RouteController struct {
	routes      cloudprovider.Routes
	kubeClient  clientset.Interface
	clusterName string
	clusterCIDR *net.IPNet
}

func New(routes cloudprovider.Routes, kubeClient clientset.Interface, clusterName string, clusterCIDR *net.IPNet) *RouteController {
	if kubeClient != nil && kubeClient.Core().GetRESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("route_controller", kubeClient.Core().GetRESTClient().GetRateLimiter())
	}
	return &RouteController{
		routes:      routes,
		kubeClient:  kubeClient,
		clusterName: clusterName,
		clusterCIDR: clusterCIDR,
	}
}

func (rc *RouteController) Run(syncPeriod time.Duration) {
	// TODO: If we do just the full Resync every 5 minutes (default value)
	// that means that we may wait up to 5 minutes before even starting
	// creating a route for it. This is bad.
	// We should have a watch on node and if we observe a new node (with CIDR?)
	// trigger reconciliation for that node.
	go wait.NonSlidingUntil(func() {
		if err := rc.reconcileNodeRoutes(); err != nil {
			glog.Errorf("Couldn't reconcile node routes: %v", err)
		}
	}, syncPeriod, wait.NeverStop)
}

func (rc *RouteController) reconcileNodeRoutes() error {
	routeList, err := rc.routes.ListRoutes(rc.clusterName)
	if err != nil {
		return fmt.Errorf("error listing routes: %v", err)
	}
	// TODO (cjcullen): use pkg/controller/framework.NewInformer to watch this
	// and reduce the number of lists needed.
	nodeList, err := rc.kubeClient.Core().Nodes().List(api.ListOptions{})
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

	wg := sync.WaitGroup{}
	rateLimiter := make(chan struct{}, maxConcurrentRouteCreations)

	for _, node := range nodes {
		// Skip if the node hasn't been assigned a CIDR yet.
		if node.Spec.PodCIDR == "" {
			continue
		}
		// Check if we have a route for this node w/ the correct CIDR.
		r := routeMap[node.Name]
		if r == nil || r.DestinationCIDR != node.Spec.PodCIDR {
			// If not, create the route.
			route := &cloudprovider.Route{
				TargetInstance:  node.Name,
				DestinationCIDR: node.Spec.PodCIDR,
			}
			nameHint := string(node.UID)
			wg.Add(1)
			go func(nodeName string, nameHint string, route *cloudprovider.Route) {
				defer wg.Done()
				for i := 0; i < maxRetries; i++ {
					startTime := time.Now()
					// Ensure that we don't have more than maxConcurrentRouteCreations
					// CreateRoute calls in flight.
					rateLimiter <- struct{}{}
					glog.Infof("Creating route for node %s %s with hint %s, throttled %v", nodeName, route.DestinationCIDR, nameHint, time.Now().Sub(startTime))
					err := rc.routes.CreateRoute(rc.clusterName, nameHint, route)
					<-rateLimiter

					rc.updateNetworkingCondition(nodeName, err == nil)
					if err != nil {
						glog.Errorf("Could not create route %s %s for node %s after %v: %v", nameHint, route.DestinationCIDR, nodeName, time.Now().Sub(startTime), err)
					} else {
						glog.Infof("Created route for node %s %s with hint %s after %v", nodeName, route.DestinationCIDR, nameHint, time.Now().Sub(startTime))
						return
					}
				}
			}(node.Name, nameHint, route)
		} else {
			rc.updateNetworkingCondition(node.Name, true)
		}
		nodeCIDRs[node.Name] = node.Spec.PodCIDR
	}
	for _, route := range routes {
		if rc.isResponsibleForRoute(route) {
			// Check if this route applies to a node we know about & has correct CIDR.
			if nodeCIDRs[route.TargetInstance] != route.DestinationCIDR {
				wg.Add(1)
				// Delete the route.
				go func(route *cloudprovider.Route, startTime time.Time) {
					glog.Infof("Deleting route %s %s", route.Name, route.DestinationCIDR)
					if err := rc.routes.DeleteRoute(rc.clusterName, route); err != nil {
						glog.Errorf("Could not delete route %s %s after %v: %v", route.Name, route.DestinationCIDR, time.Now().Sub(startTime), err)
					} else {
						glog.Infof("Deleted route %s %s after %v", route.Name, route.DestinationCIDR, time.Now().Sub(startTime))
					}
					wg.Done()

				}(route, time.Now())
			}
		}
	}
	wg.Wait()
	return nil
}

func updateNetworkingCondition(node *api.Node, routeCreated bool) {
	_, networkingCondition := api.GetNodeCondition(&node.Status, api.NodeNetworkUnavailable)
	currentTime := unversioned.Now()
	if routeCreated {
		if networkingCondition != nil && networkingCondition.Status != api.ConditionFalse {
			networkingCondition.Status = api.ConditionFalse
			networkingCondition.Reason = "RouteCreated"
			networkingCondition.Message = "RouteController created a route"
			networkingCondition.LastTransitionTime = currentTime
		} else if networkingCondition == nil {
			node.Status.Conditions = append(node.Status.Conditions, api.NodeCondition{
				Type:               api.NodeNetworkUnavailable,
				Status:             api.ConditionFalse,
				Reason:             "RouteCreated",
				Message:            "RouteController created a route",
				LastTransitionTime: currentTime,
			})
		}
	} else {
		if networkingCondition != nil && networkingCondition.Status != api.ConditionTrue {
			networkingCondition.Status = api.ConditionTrue
			networkingCondition.Reason = "NoRouteCreated"
			networkingCondition.Message = "RouteController failed to create a route"
			networkingCondition.LastTransitionTime = currentTime
		} else if networkingCondition == nil {
			node.Status.Conditions = append(node.Status.Conditions, api.NodeCondition{
				Type:               api.NodeNetworkUnavailable,
				Status:             api.ConditionTrue,
				Reason:             "NoRouteCreated",
				Message:            "RouteController failed to create a route",
				LastTransitionTime: currentTime,
			})
		}
	}
}

func (rc *RouteController) updateNetworkingCondition(nodeName string, routeCreated bool) error {
	var err error
	for i := 0; i < updateNodeStatusMaxRetries; i++ {
		node, err := rc.kubeClient.Core().Nodes().Get(nodeName)
		if err != nil {
			glog.Errorf("Error geting node: %v", err)
			continue
		}
		updateNetworkingCondition(node, routeCreated)
		// TODO: Use Patch instead once #26381 is merged.
		// See kubernetes/node-problem-detector#9 for details.
		if _, err = rc.kubeClient.Core().Nodes().UpdateStatus(node); err == nil {
			return nil
		}
		if i+1 < updateNodeStatusMaxRetries {
			glog.Errorf("Error updating node %s, retrying: %v", node.Name, err)
		} else {
			glog.Errorf("Error updating node %s: %v", node.Name, err)
		}
	}
	return err
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
	return true
}
