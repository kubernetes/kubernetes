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
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	workerGoroutines = 20
)

type RouteController struct {
	routes      cloudprovider.Routes
	kubeClient  clientset.Interface
	clusterName string
	clusterCIDR *net.IPNet

	// To allow injection of syncRoutine for testing.
	syncHandler func(key string) error

	// Node framework and store
	nodeController *framework.Controller
	nodeStore      cache.StoreToNodeLister

	// routes that need to be created
	queue *workqueue.Type
}

func NewRouteController(routes cloudprovider.Routes, client clientset.Interface, resyncPeriod controller.ResyncPeriodFunc, clusterName string, clusterCIDR *net.IPNet) *RouteController {
	rc := &RouteController{
		routes:      routes,
		kubeClient:  client,
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
		resyncPeriod(),
		framework.ResourceEventHandlerFuncs{
			AddFunc: rc.enqueueNode,
			UpdateFunc: func(old, cur interface{}) {
				oldNode := old.(*api.Node)
				curNode := cur.(*api.Node)
				// if PodCIDR of node changed, we need recreate the route
				if oldNode.Spec.PodCIDR != curNode.Spec.PodCIDR {
					glog.Warningf("Observed PodCIDR updated for node: %v, %v->%v", curNode.Name, oldNode.Spec.PodCIDR, curNode.Spec.PodCIDR)
					rc.enqueueNode(cur)
				}
			},
			DeleteFunc: rc.enqueueNode,
		},
	)

	rc.syncHandler = rc.syncNodeRoute

	return rc
}

// Run begins watching and syncing.
func (rc *RouteController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	go rc.nodeController.Run(stopCh)
	for i := 0; i < workerGoroutines; i++ {
		go wait.Until(rc.worker, time.Second, stopCh)
	}
	go func() {
		defer utilruntime.HandleCrash()
		time.Sleep(time.Minute * 5)
		rc.checkLeftoverRoutes()
	}()
	<-stopCh
	glog.Infof("Shutting down route controller")
	rc.queue.ShutDown()
}

func (rc *RouteController) worker() {
	for {
		func() {
			key, quit := rc.queue.Get()
			if quit {
				return
			}
			defer rc.queue.Done(key)
			err := rc.syncHandler(key.(string))
			if err != nil {
				glog.Errorf("Error syncing route: %v", err)
			}
		}()
	}
}

func (rc *RouteController) enqueueNode(obj interface{}) {
	// node has no namespace, the key of a node is it's name
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	rc.queue.Add(key)
}

func (rc *RouteController) syncNodeRoute(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing route %q (%v)", key, time.Now().Sub(startTime))
	}()

	var node *api.Node
	obj, exists, err := rc.nodeStore.Store.GetByKey(key)
	if err != nil {
		glog.Infof("Unable to retrieve node %v from store: %v", key, err)
		rc.queue.Add(key)
		return err
	}
	if exists {
		node = obj.(*api.Node)
	}

	var r *cloudprovider.Route
	routes, err := rc.routes.ListRoutes(rc.clusterName)
	if err != nil {
		return fmt.Errorf("error listing routes: %v", err)
	}
	// Check if we have a route for this node
	// TODO(mqliang): we should compute this once and dynamically update it using Watch, not constantly re-compute.
	for _, route := range routes {
		if route.TargetInstance != key {
			continue
		}
		r = route
	}

	var shouldDelete bool
	var shouldCreate bool

	// if we have a route but the node has been deleted, remove the route
	if node == nil && r != nil {
		glog.Infof("Node %v has been deleted, remove it's route", key)
		shouldDelete = true
	}
	// if we have no route for node, create one
	if node != nil && r == nil {
		glog.Infof("No route for Node %v, create one", key)
		shouldCreate = true
	}
	// check if the route for this node has the correct CIDR, if not, remove the old route and create a new one
	if node != nil && r != nil && r.DestinationCIDR != node.Spec.PodCIDR {
		glog.Infof("The route for %v has the incorrect CIDR, remove it and create a new one", key)
		shouldDelete = true
		shouldCreate = true
	}

	if shouldDelete {
		if err := rc.routes.DeleteRoute(rc.clusterName, r); err != nil {
			glog.Errorf("Could not delete route %s %s: %v", r.Name, r.DestinationCIDR, err)
		}
	}

	if shouldCreate && node.Spec.PodCIDR != "" {
		route := &cloudprovider.Route{
			TargetInstance:  node.Name,
			DestinationCIDR: node.Spec.PodCIDR,
		}
		nameHint := string(node.UID)
		if err := rc.routes.CreateRoute(rc.clusterName, nameHint, route); err != nil {
			glog.Errorf("Could not create route %s %s: %v", nameHint, route.DestinationCIDR, err)
			// retry
			rc.enqueueNode(node)
		}
	}

	return nil
}

// checkLeftoverRoutes lists all currently existing routes and detect routes
// that exist with no corresponding nodes; these routes need to be deleted.
// We only need to do this once on startup, because in steady-state these are
// detected (but some stragglers could have been left behind if the route controller
// reboots).
func (rc *RouteController) checkLeftoverRoutes() {

	// routeMap maps routeTargetInstance->route
	routeMap := map[string]*cloudprovider.Route{}
	// nodeCIDRs maps nodeName->nodeCIDR
	nodeCIDRs := make(map[string]string)

	routes, err := rc.routes.ListRoutes(rc.clusterName)
	if err != nil {
		glog.Errorf("Unable to list routes (%v); orphaned routes will not be cleaned up. (They're pretty harmless, but you can restart this component if you want another attempt made.)", err)
		return
	}
	for _, route := range routes {
		routeMap[route.TargetInstance] = route
	}

	nodeList, err := rc.nodeStore.List()
	if err != nil {
		glog.Errorf("error listing nodes: %v", err)
		return
	}
	for _, node := range nodeList.Items {
		if node.Spec.PodCIDR == "" {
			continue
		}
		nodeCIDRs[node.Name] = node.Spec.PodCIDR
	}

	for _, route := range routes {
		if !rc.isResponsibleForRoute(route) {
			continue
		}
		// Check if this route applies to a node we know about & has correct CIDR.
		if nodeCIDRs[route.TargetInstance] == route.DestinationCIDR {
			continue
		}
		// Delete the route.
		go func(route *cloudprovider.Route) {
			if err := rc.routes.DeleteRoute(rc.clusterName, route); err != nil {
				glog.Errorf("Could not delete route %s %s: %v", route.Name, route.DestinationCIDR, err)
			}
		}(route)
	}
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
