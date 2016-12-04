/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/metrics"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"
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
	// Node framework and store
	nodeController *cache.Controller
	nodeStore      cache.StoreToNodeLister
}

func New(routes cloudprovider.Routes, kubeClient clientset.Interface, clusterName string, clusterCIDR *net.IPNet) *RouteController {
	if kubeClient != nil && kubeClient.Core().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("route_controller", kubeClient.Core().RESTClient().GetRateLimiter())
	}
	rc := &RouteController{
		routes:      routes,
		kubeClient:  kubeClient,
		clusterName: clusterName,
		clusterCIDR: clusterCIDR,
	}

	rc.nodeStore.Store, rc.nodeController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				return rc.kubeClient.Core().Nodes().List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				return rc.kubeClient.Core().Nodes().Watch(options)
			},
		},
		&v1.Node{},
		controller.NoResyncPeriodFunc(),
		cache.ResourceEventHandlerFuncs{},
	)

	return rc
}

func (rc *RouteController) Run(syncPeriod time.Duration) {
	go rc.nodeController.Run(wait.NeverStop)

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
	if !rc.nodeController.HasSynced() {
		return fmt.Errorf("nodeController is not yet synced")
	}
	nodeList, err := rc.nodeStore.List()
	if err != nil {
		return fmt.Errorf("error listing nodes: %v", err)
	}
	return rc.reconcile(nodeList.Items, routeList)
}

func (rc *RouteController) reconcile(nodes []v1.Node, routes []*cloudprovider.Route) error {
	// nodeCIDRs maps nodeName->nodeCIDR
	nodeCIDRs := make(map[types.NodeName]string)
	// routeMap maps routeTargetNode->route
	routeMap := make(map[types.NodeName]*cloudprovider.Route)
	for _, route := range routes {
		routeMap[route.TargetNode] = route
	}

	wg := sync.WaitGroup{}
	rateLimiter := make(chan struct{}, maxConcurrentRouteCreations)

	for _, node := range nodes {
		// Skip if the node hasn't been assigned a CIDR yet.
		if node.Spec.PodCIDR == "" {
			continue
		}
		nodeName := types.NodeName(node.Name)
		// Check if we have a route for this node w/ the correct CIDR.
		r := routeMap[nodeName]
		if r == nil || r.DestinationCIDR != node.Spec.PodCIDR {
			// If not, create the route.
			route := &cloudprovider.Route{
				TargetNode:      nodeName,
				DestinationCIDR: node.Spec.PodCIDR,
			}
			nameHint := string(node.UID)
			wg.Add(1)
			go func(nodeName types.NodeName, nameHint string, route *cloudprovider.Route) {
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
			}(nodeName, nameHint, route)
		} else {
			// Update condition only if it doesn't reflect the current state.
			_, condition := v1.GetNodeCondition(&node.Status, v1.NodeNetworkUnavailable)
			if condition == nil || condition.Status != v1.ConditionFalse {
				rc.updateNetworkingCondition(types.NodeName(node.Name), true)
			}
		}
		nodeCIDRs[nodeName] = node.Spec.PodCIDR
	}
	for _, route := range routes {
		if rc.isResponsibleForRoute(route) {
			// Check if this route applies to a node we know about & has correct CIDR.
			if nodeCIDRs[route.TargetNode] != route.DestinationCIDR {
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

func (rc *RouteController) updateNetworkingCondition(nodeName types.NodeName, routeCreated bool) error {
	var err error
	for i := 0; i < updateNodeStatusMaxRetries; i++ {
		// Patch could also fail, even though the chance is very slim. So we still do
		// patch in the retry loop.
		currentTime := metav1.Now()
		if routeCreated {
			err = nodeutil.SetNodeCondition(rc.kubeClient, nodeName, v1.NodeCondition{
				Type:               v1.NodeNetworkUnavailable,
				Status:             v1.ConditionFalse,
				Reason:             "RouteCreated",
				Message:            "RouteController created a route",
				LastTransitionTime: currentTime,
			})
		} else {
			err = nodeutil.SetNodeCondition(rc.kubeClient, nodeName, v1.NodeCondition{
				Type:               v1.NodeNetworkUnavailable,
				Status:             v1.ConditionTrue,
				Reason:             "NoRouteCreated",
				Message:            "RouteController failed to create a route",
				LastTransitionTime: currentTime,
			})
		}
		if err == nil {
			return nil
		}
		if i == updateNodeStatusMaxRetries || !errors.IsConflict(err) {
			glog.Errorf("Error updating node %s: %v", nodeName, err)
			return err
		}
		glog.Errorf("Error updating node %s, retrying: %v", nodeName, err)
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
