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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/metrics"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

type StatusType string

const (
	// Maximal number of concurrent CreateRoute API calls.
	// TODO: This should be per-provider.
	maxConcurrentRouteCreations int = 200
	// Maximum number of retries of route creations.
	maxRetries int = 5
	// Maximum number of retries of node status update.
	updateNodeStatusMaxRetries int        = 3
	statusDelete               StatusType = "Delete"
	statusAdd                  StatusType = "Add"
	statusUpdate               StatusType = "Update"
)

type RouteController struct {
	routes      cloudprovider.Routes
	kubeClient  clientset.Interface
	clusterName string
	clusterCIDR *net.IPNet
	// Node framework and store
	nodeController *framework.Controller
	nodeStore      cache.StoreToNodeLister
	// Nodes that need to be synced
	queue *workqueue.Type
	// To allow injection of syncRoutes for testing.
	syncHandler func(nodeStatus NodeStatus) error
	rateLimiter chan struct{}
}

type NodeStatus struct {
	node   *api.Node
	status StatusType
}

func New(routes cloudprovider.Routes, kubeClient clientset.Interface, clusterName string, clusterCIDR *net.IPNet, syncPeriod time.Duration) *RouteController {
	if kubeClient != nil && kubeClient.Core().GetRESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("route_controller", kubeClient.Core().GetRESTClient().GetRateLimiter())
	}
	rc := &RouteController{
		routes:      routes,
		kubeClient:  kubeClient,
		clusterName: clusterName,
		clusterCIDR: clusterCIDR,
		queue:       workqueue.New(),
	}
	rc.nodeStore.Store, rc.nodeController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return rc.kubeClient.Core().Nodes().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return rc.kubeClient.Core().Nodes().Watch(options)
			},
		},
		&api.Node{},
		syncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				nodeStatus := NodeStatus{
					node:   obj.(*api.Node),
					status: statusAdd,
				}
				rc.queue.Add(nodeStatus)
			},
			UpdateFunc: func(old, new interface{}) {
				nodeStatus := NodeStatus{
					node:   new.(*api.Node),
					status: statusUpdate,
				}
				rc.queue.Add(nodeStatus)
			},
			DeleteFunc: func(obj interface{}) {
				nodeStatus := NodeStatus{
					node:   obj.(*api.Node),
					status: statusDelete,
				}
				rc.queue.Add(nodeStatus)
			},
		},
	)
	rc.syncHandler = rc.reconcileNodeRoutes
	rc.rateLimiter = make(chan struct{}, maxConcurrentRouteCreations)
	return rc
}

func (rc *RouteController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	go rc.nodeController.Run(stopCh)

	for i := 0; i < workers; i++ {
		go wait.Until(rc.worker, time.Second, stopCh)
	}
	<-stopCh
	rc.queue.ShutDown()
}

func (rc *RouteController) worker() {
	workFunc := func() bool {
		statusObj, quit := rc.queue.Get()
		if quit {
			return true
		}
		defer rc.queue.Done(statusObj)
		err := rc.syncHandler(statusObj.(NodeStatus))
		if err != nil {
			glog.Errorf("Error syncing route: %v", err)
		}
		return false
	}
	for {
		if quit := workFunc(); quit {
			glog.Infof("route controller worker shutting down")
			return
		}
	}
}

func (rc *RouteController) reconcileNodeRoutes(nodeStatus NodeStatus) error {
	routeList, err := rc.routes.ListRoutes(rc.clusterName)
	if err != nil {
		return fmt.Errorf("error listing routes: %v", err)
	}
	return rc.reconcile(nodeStatus, routeList)
}

func (rc *RouteController) reconcile(nodeStatus NodeStatus, routes []*cloudprovider.Route) error {
	node := *nodeStatus.node

	routeMap := make(map[string]*cloudprovider.Route)
	for _, route := range routes {
		routeMap[route.TargetInstance] = route
	}

	wg := sync.WaitGroup{}

	if (nodeStatus.status == statusAdd || nodeStatus.status == statusUpdate) && node.Spec.PodCIDR != "" {
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
					rc.rateLimiter <- struct{}{}
					glog.Infof("Creating route for node %s %s with hint %s, throttled %v", nodeName, route.DestinationCIDR, nameHint, time.Now().Sub(startTime))
					err := rc.routes.CreateRoute(rc.clusterName, nameHint, route)
					<-rc.rateLimiter

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
	}
	if nodeStatus.status == statusDelete {
		for _, route := range routes {
			if rc.isResponsibleForRoute(route) {
				// Check if this route applies to a node we know about & has correct CIDR.
				if node.Name == route.TargetInstance && node.Spec.PodCIDR == route.DestinationCIDR {
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
	}
	wg.Wait()
	return nil
}

func (rc *RouteController) updateNetworkingCondition(nodeName string, routeCreated bool) error {
	var err error
	for i := 0; i < updateNodeStatusMaxRetries; i++ {
		// Patch could also fail, even though the chance is very slim. So we still do
		// patch in the retry loop.
		currentTime := unversioned.Now()
		if routeCreated {
			err = nodeutil.SetNodeCondition(rc.kubeClient, nodeName, api.NodeCondition{
				Type:               api.NodeNetworkUnavailable,
				Status:             api.ConditionFalse,
				Reason:             "RouteCreated",
				Message:            "RouteController created a route",
				LastTransitionTime: currentTime,
			})
		} else {
			err = nodeutil.SetNodeCondition(rc.kubeClient, nodeName, api.NodeCondition{
				Type:               api.NodeNetworkUnavailable,
				Status:             api.ConditionTrue,
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
