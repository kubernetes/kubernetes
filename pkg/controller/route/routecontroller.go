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
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

type RouteController struct {
	routes      cloudprovider.Routes
	kubeClient  client.Interface
	clusterName string
	clusterCIDR *net.IPNet

	// To allow injection of syncRoutine for testing.
	syncHandler func(key string) error

	// nodeCIDRs maps nodeName->nodeCIDR
	nodeCIDRs map[string]string
	// routeMap maps routeTargetInstance->route
	routeMap map[string]*cloudprovider.Route

	// Node framework and store
	nodeController *framework.Controller
	nodeStore      cache.StoreToNodeLister

	// routes that need to be created
	queue *workqueue.Type
}

func NewRouteController(routes cloudprovider.Routes, client client.Interface, resyncPeriod controller.ResyncPeriodFunc, clusterName string, clusterCIDR *net.IPNet) *RouteController {
	rc := &RouteController{
		routes:      routes,
		kubeClient:  client,
		clusterName: clusterName,
		clusterCIDR: clusterCIDR,
		nodeCIDRs:   make(map[string]string),
		routeMap:    make(map[string]*cloudprovider.Route),
	}

	rc.nodeStore.Store, rc.nodeController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return rc.kubeClient.Nodes().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return rc.kubeClient.Nodes().Watch(options)
			},
		},
		&api.Node{},
		resyncPeriod(),
		framework.ResourceEventHandlerFuncs{
			AddFunc: rc.enqueueNode,
			UpdateFunc: func(old, cur interface{}) {
				oldNode := old.(*api.Node)
				curNode := cur.(*api.Node)
				if oldNode.Spec.PodCIDR != curNode.Spec.PodCIDR {
					glog.V(4).Infof("Observed PodCIDR updated for node: %v, %d->%d", curNode.Name, oldNode.Spec.PodCIDR, curNode.Spec.PodCIDR)
				}
				rc.enqueueNode(old)
				rc.enqueueNode(cur)
			},
			DeleteFunc: rc.enqueueNode,
		},
	)

	rc.syncHandler = rc.syncNodeRoute

	return rc
}

// Run begins watching and syncing.
func (rc *RouteController) Run(stopCh <-chan struct{}) error {
	if err := rc.init(); err != nil {
		return err
	}
	defer util.HandleCrash()
	go rc.nodeController.Run(stopCh)

	go util.Until(rc.worker, time.Second, stopCh)

	go func() {
		defer util.HandleCrash()
		time.Sleep(time.Minute * 5)
		rc.checkLeftoverRoutes()
	}()
	<-stopCh
	glog.Infof("Shutting down route controller")
	rc.queue.ShutDown()
	return nil
}

func (rc *RouteController) init() error {
	routeList, err := rc.routes.ListRoutes(rc.clusterName)
	if err != nil {
		return fmt.Errorf("error listing routes: %v", err)
	}

	for _, route := range routeList {
		rc.routeMap[route.TargetInstance] = route
	}

	return nil
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
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	node := obj.(*api.Node)
	rc.nodeCIDRs[node.Name] = node.Spec.PodCIDR
	rc.queue.Add(key)
}

func (rc *RouteController) syncNodeRoute(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing route %q (%v)", key, time.Now().Sub(startTime))
	}()

	obj, exists, err := rc.nodeStore.Store.GetByKey(key)
	if err != nil {
		glog.Infof("Unable to retrieve node %v from store: %v", key, err)
		rc.queue.Add(key)
		return err
	}

	node := obj.(*api.Node)
	if !exists {
		glog.Infof("Node has been deleted %v", key)
		// Delete the route.
		go func(route *cloudprovider.Route) {
			if err := rc.routes.DeleteRoute(rc.clusterName, route); err != nil {
				glog.Errorf("Could not delete route %s %s: %v", route.Name, route.DestinationCIDR, err)
			}
		}(rc.routeMap[node.Name])

		return nil
	}

	// Check if we have a route for this node w/ the correct CIDR.
	r := rc.routeMap[node.Name]

	if r == nil || r.DestinationCIDR != node.Spec.PodCIDR {
		// If not, create the route.
		route := &cloudprovider.Route{
			TargetInstance:  node.Name,
			DestinationCIDR: node.Spec.PodCIDR,
		}
		nameHint := string(node.UID)
		go func(nameHint string, route *cloudprovider.Route) {
			if err := rc.routes.CreateRoute(rc.clusterName, nameHint, route); err != nil {
				glog.Errorf("Could not create route %s %s: %v", nameHint, route.DestinationCIDR, err)
			}
		}(nameHint, route)
	}
	rc.nodeCIDRs[node.Name] = node.Spec.PodCIDR

	return nil
}

// checkLeftoverRoutes lists all currently existing routes and detect routes
// that exist with no corresponding nodes; these routes need to be deleted.
// We only need to do this once on startup, because in steady-state these are
// detected (butsome stragglers could have been left behind if the route controller
// reboots).
func (rc *RouteController) checkLeftoverRoutes() {
	routes, err := rc.routes.ListRoutes(rc.clusterName)
	if err != nil {
		glog.Errorf("Unable to list routes (%v); orphaned routes will not be cleaned up. (They're pretty harmless, but you can restart this component if you want another attempt made.)", err)
		return
	}
	for _, route := range routes {
		if rc.isResponsibleForRoute(route) {
			// Check if this route applies to a node we know about & has correct CIDR.
			if rc.nodeCIDRs[route.TargetInstance] == route.DestinationCIDR {
				continue
			}
			// Delete the route.
			go func(route *cloudprovider.Route) {
				if err := rc.routes.DeleteRoute(rc.clusterName, route); err != nil {
					glog.Errorf("Could not delete route %s %s: %v", route.Name, route.DestinationCIDR, err)
				}
			}(route)

			delete(rc.routeMap, route.TargetInstance)
		}
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
