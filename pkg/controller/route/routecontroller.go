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
	updateNodeStatusMaxRetries = 3
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

func tryUpdateNodeStatus(node *api.Node, kubeClient clientset.Interface) error {
	for i := 0; i < updateNodeStatusMaxRetries; i++ {
		if _, err := kubeClient.Core().Nodes().UpdateStatus(node); err == nil {
			break
		} else {
			if i+1 < updateNodeStatusMaxRetries {
				glog.Errorf("Error updating node %s - will retry: %v", node.Name, err)
			} else {
				glog.Errorf("Error updating node %s - wont retry: %v", node.Name, err)
				return err
			}
		}
	}
	return nil
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
			glog.V(2).Infof("Creating route for node %s %s with hint %s", node.Name, route.DestinationCIDR, nameHint)
			go func(node api.Node, nameHint string, route *cloudprovider.Route, startTime time.Time) {
				_, networkingCondition := api.GetNodeCondition(&node.Status, api.NodeNetworkingReady)
				if err := rc.routes.CreateRoute(rc.clusterName, nameHint, route); err != nil {
					currentTime := unversioned.Now()
					if networkingCondition != nil && networkingCondition.Status != api.ConditionFalse {
						networkingCondition.Status = api.ConditionFalse
						networkingCondition.Reason = "NoRouteCreated"
						networkingCondition.Message = "RouteController failed to create a route"
						networkingCondition.LastTransitionTime = currentTime
					} else if networkingCondition == nil {
						node.Status.Conditions = append(node.Status.Conditions, api.NodeCondition{
							Type:               api.NodeNetworkingReady,
							Status:             api.ConditionFalse,
							Reason:             "NoRouteCreated",
							Message:            "RouteController failed to create a route",
							LastTransitionTime: currentTime,
						})
					}
					glog.Errorf("Could not create route %s %s for node %s after %v: %v", nameHint, route.DestinationCIDR, node.Name, time.Now().Sub(startTime), err)
				} else {
					currentTime := unversioned.Now()
					if networkingCondition != nil && networkingCondition.Status != api.ConditionTrue {
						networkingCondition.Status = api.ConditionTrue
						networkingCondition.Reason = "RouteCreated"
						networkingCondition.Message = "RouteController created a route"
						networkingCondition.LastTransitionTime = currentTime
					} else if networkingCondition == nil {
						node.Status.Conditions = append(node.Status.Conditions, api.NodeCondition{
							Type:               api.NodeNetworkingReady,
							Status:             api.ConditionTrue,
							Reason:             "RouteCreated",
							Message:            "RouteController created a route",
							LastTransitionTime: currentTime,
						})
					}
					glog.V(2).Infof("Created route for node %s %s with hint %s after %v", node.Name, route.DestinationCIDR, nameHint, time.Now().Sub(startTime))

				}
				if rc.kubeClient != nil {
					tryUpdateNodeStatus(&node, rc.kubeClient)
				}
				wg.Done()
			}(node, nameHint, route, time.Now())
		} else {
			// Check if the NodeNetworking condition is correctly set and if not do it.
			currentTime := unversioned.Now()
			_, networkingCondition := api.GetNodeCondition(&node.Status, api.NodeNetworkingReady)
			if networkingCondition != nil && networkingCondition.Status != api.ConditionTrue {
				networkingCondition.Status = api.ConditionTrue
				networkingCondition.Reason = "RouteCreated"
				networkingCondition.Message = "RouteController created a route"
				networkingCondition.LastTransitionTime = currentTime
			} else if networkingCondition == nil {
				node.Status.Conditions = append(node.Status.Conditions, api.NodeCondition{
					Type:               api.NodeNetworkingReady,
					Status:             api.ConditionTrue,
					Reason:             "RouteCreated",
					Message:            "RouteController created a route",
					LastTransitionTime: currentTime,
				})
			}
			if rc.kubeClient != nil {
				tryUpdateNodeStatus(&node, rc.kubeClient)
			}
		}
		nodeCIDRs[node.Name] = node.Spec.PodCIDR
	}
	for _, route := range routes {
		if rc.isResponsibleForRoute(route) {
			// Check if this route applies to a node we know about & has correct CIDR.
			if nodeCIDRs[route.TargetInstance] != route.DestinationCIDR {
				wg.Add(1)
				// Delete the route.
				glog.V(2).Infof("Deleting route %s %s", route.Name, route.DestinationCIDR)
				go func(route *cloudprovider.Route, startTime time.Time) {
					if err := rc.routes.DeleteRoute(rc.clusterName, route); err != nil {
						glog.Errorf("Could not delete route %s %s after %v: %v", route.Name, route.DestinationCIDR, time.Now().Sub(startTime), err)
					} else {
						glog.V(2).Infof("Deleted route %s %s after %v", route.Name, route.DestinationCIDR, time.Now().Sub(startTime))
					}
					wg.Done()

				}(route, time.Now())
			}
		}
	}
	wg.Wait()
	return nil
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
