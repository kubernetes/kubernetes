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
	"context"
	"fmt"
	"net"
	"sync"
	"time"

	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	clientretry "k8s.io/client-go/util/retry"
	cloudprovider "k8s.io/cloud-provider"
	controllersmetrics "k8s.io/component-base/metrics/prometheus/controllers"
	nodeutil "k8s.io/component-helpers/node/util"
)

const (
	// Maximal number of concurrent CreateRoute API calls.
	// TODO: This should be per-provider.
	maxConcurrentRouteCreations int = 200
)

var updateNetworkConditionBackoff = wait.Backoff{
	Steps:    5, // Maximum number of retries.
	Duration: 100 * time.Millisecond,
	Jitter:   1.0,
}

type RouteController struct {
	routes           cloudprovider.Routes
	kubeClient       clientset.Interface
	clusterName      string
	clusterCIDRs     []*net.IPNet
	nodeLister       corelisters.NodeLister
	nodeListerSynced cache.InformerSynced
	broadcaster      record.EventBroadcaster
	recorder         record.EventRecorder
}

func New(routes cloudprovider.Routes, kubeClient clientset.Interface, nodeInformer coreinformers.NodeInformer, clusterName string, clusterCIDRs []*net.IPNet) *RouteController {
	if len(clusterCIDRs) == 0 {
		klog.Fatal("RouteController: Must specify clusterCIDR.")
	}

	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "route_controller"})

	rc := &RouteController{
		routes:           routes,
		kubeClient:       kubeClient,
		clusterName:      clusterName,
		clusterCIDRs:     clusterCIDRs,
		nodeLister:       nodeInformer.Lister(),
		nodeListerSynced: nodeInformer.Informer().HasSynced,
		broadcaster:      eventBroadcaster,
		recorder:         recorder,
	}

	return rc
}

func (rc *RouteController) Run(ctx context.Context, syncPeriod time.Duration, controllerManagerMetrics *controllersmetrics.ControllerManagerMetrics) {
	defer utilruntime.HandleCrash()

	// Start event processing pipeline.
	if rc.broadcaster != nil {
		rc.broadcaster.StartStructuredLogging(0)
		rc.broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: rc.kubeClient.CoreV1().Events("")})
		defer rc.broadcaster.Shutdown()
	}

	klog.Info("Starting route controller")
	defer klog.Info("Shutting down route controller")
	controllerManagerMetrics.ControllerStarted("route")
	defer controllerManagerMetrics.ControllerStopped("route")

	if !cache.WaitForNamedCacheSync("route", ctx.Done(), rc.nodeListerSynced) {
		return
	}

	// TODO: If we do just the full Resync every 5 minutes (default value)
	// that means that we may wait up to 5 minutes before even starting
	// creating a route for it. This is bad.
	// We should have a watch on node and if we observe a new node (with CIDR?)
	// trigger reconciliation for that node.
	go wait.NonSlidingUntil(func() {
		if err := rc.reconcileNodeRoutes(ctx); err != nil {
			klog.Errorf("Couldn't reconcile node routes: %v", err)
		}
	}, syncPeriod, ctx.Done())

	<-ctx.Done()
}

func (rc *RouteController) reconcileNodeRoutes(ctx context.Context) error {
	routeList, err := rc.routes.ListRoutes(ctx, rc.clusterName)
	if err != nil {
		return fmt.Errorf("error listing routes: %v", err)
	}
	nodes, err := rc.nodeLister.List(labels.Everything())
	if err != nil {
		return fmt.Errorf("error listing nodes: %v", err)
	}
	return rc.reconcile(ctx, nodes, routeList)
}

func (rc *RouteController) reconcile(ctx context.Context, nodes []*v1.Node, routes []*cloudprovider.Route) error {
	var l sync.Mutex
	// for each node a map of podCIDRs and their created status
	nodeRoutesStatuses := make(map[types.NodeName]map[string]bool)
	// routeMap maps routeTargetNode->route
	routeMap := make(map[types.NodeName][]*cloudprovider.Route)
	for _, route := range routes {
		if route.TargetNode != "" {
			routeMap[route.TargetNode] = append(routeMap[route.TargetNode], route)
		}
	}

	wg := sync.WaitGroup{}
	rateLimiter := make(chan struct{}, maxConcurrentRouteCreations)
	// searches existing routes by node for a matching route

	for _, node := range nodes {
		// Skip if the node hasn't been assigned a CIDR yet.
		if len(node.Spec.PodCIDRs) == 0 {
			continue
		}
		nodeName := types.NodeName(node.Name)
		l.Lock()
		nodeRoutesStatuses[nodeName] = make(map[string]bool)
		l.Unlock()
		// for every node, for every cidr
		for _, podCIDR := range node.Spec.PodCIDRs {
			// we add it to our nodeCIDRs map here because add and delete go routines run at the same time
			l.Lock()
			nodeRoutesStatuses[nodeName][podCIDR] = false
			l.Unlock()
			// ignore if already created
			if hasRoute(routeMap, nodeName, podCIDR) {
				l.Lock()
				nodeRoutesStatuses[nodeName][podCIDR] = true // a route for this podCIDR is already created
				l.Unlock()
				continue
			}
			// if we are here, then a route needs to be created for this node
			route := &cloudprovider.Route{
				TargetNode:      nodeName,
				DestinationCIDR: podCIDR,
			}
			// cloud providers that:
			// - depend on nameHint
			// - trying to support dual stack
			// will have to carefully generate new route names that allow node->(multi cidr)
			nameHint := string(node.UID)
			wg.Add(1)
			go func(nodeName types.NodeName, nameHint string, route *cloudprovider.Route) {
				defer wg.Done()
				err := clientretry.RetryOnConflict(updateNetworkConditionBackoff, func() error {
					startTime := time.Now()
					// Ensure that we don't have more than maxConcurrentRouteCreations
					// CreateRoute calls in flight.
					rateLimiter <- struct{}{}
					klog.Infof("Creating route for node %s %s with hint %s, throttled %v", nodeName, route.DestinationCIDR, nameHint, time.Since(startTime))
					err := rc.routes.CreateRoute(ctx, rc.clusterName, nameHint, route)
					<-rateLimiter
					if err != nil {
						msg := fmt.Sprintf("Could not create route %s %s for node %s after %v: %v", nameHint, route.DestinationCIDR, nodeName, time.Since(startTime), err)
						if rc.recorder != nil {
							rc.recorder.Eventf(
								&v1.ObjectReference{
									Kind:      "Node",
									Name:      string(nodeName),
									UID:       types.UID(nodeName),
									Namespace: "",
								}, v1.EventTypeWarning, "FailedToCreateRoute", msg)
							klog.V(4).Infof(msg)
							return err
						}
					}
					l.Lock()
					nodeRoutesStatuses[nodeName][route.DestinationCIDR] = true
					l.Unlock()
					klog.Infof("Created route for node %s %s with hint %s after %v", nodeName, route.DestinationCIDR, nameHint, time.Since(startTime))
					return nil
				})
				if err != nil {
					klog.Errorf("Could not create route %s %s for node %s: %v", nameHint, route.DestinationCIDR, nodeName, err)
				}
			}(nodeName, nameHint, route)
		}
	}

	// searches our bag of node->cidrs for a match
	nodeHasCidr := func(nodeName types.NodeName, cidr string) bool {
		l.Lock()
		defer l.Unlock()

		nodeRoutes := nodeRoutesStatuses[nodeName]
		if nodeRoutes == nil {
			return false
		}
		_, exist := nodeRoutes[cidr]
		return exist
	}
	// delete routes that are not in use
	for _, route := range routes {
		if rc.isResponsibleForRoute(route) {
			// Check if this route is a blackhole, or applies to a node we know about & has an incorrect CIDR.
			if route.Blackhole || !nodeHasCidr(route.TargetNode, route.DestinationCIDR) {
				wg.Add(1)
				// Delete the route.
				go func(route *cloudprovider.Route, startTime time.Time) {
					defer wg.Done()
					// respect the rate limiter
					rateLimiter <- struct{}{}
					klog.Infof("Deleting route %s %s", route.Name, route.DestinationCIDR)
					if err := rc.routes.DeleteRoute(ctx, rc.clusterName, route); err != nil {
						klog.Errorf("Could not delete route %s %s after %v: %v", route.Name, route.DestinationCIDR, time.Since(startTime), err)
					} else {
						klog.Infof("Deleted route %s %s after %v", route.Name, route.DestinationCIDR, time.Since(startTime))
					}
					<-rateLimiter
				}(route, time.Now())
			}
		}
	}
	wg.Wait()

	// after all routes have been created (or not), we start updating
	// all nodes' statuses with the outcome
	for _, node := range nodes {
		wg.Add(1)
		nodeRoutes := nodeRoutesStatuses[types.NodeName(node.Name)]
		allRoutesCreated := true

		if len(nodeRoutes) == 0 {
			go func(n *v1.Node) {
				defer wg.Done()
				klog.Infof("node %v has no routes assigned to it. NodeNetworkUnavailable will be set to true", n.Name)
				if err := rc.updateNetworkingCondition(n, false); err != nil {
					klog.Errorf("failed to update networking condition when no nodeRoutes: %v", err)
				}
			}(node)
			continue
		}

		// check if all routes were created. if so, then it should be ready
		for _, created := range nodeRoutes {
			if !created {
				allRoutesCreated = false
				break
			}
		}
		go func(n *v1.Node) {
			defer wg.Done()
			if err := rc.updateNetworkingCondition(n, allRoutesCreated); err != nil {
				klog.Errorf("failed to update networking condition: %v", err)
			}
		}(node)
	}
	wg.Wait()
	return nil
}

func (rc *RouteController) updateNetworkingCondition(node *v1.Node, routesCreated bool) error {
	_, condition := nodeutil.GetNodeCondition(&(node.Status), v1.NodeNetworkUnavailable)
	if routesCreated && condition != nil && condition.Status == v1.ConditionFalse {
		klog.V(2).Infof("set node %v with NodeNetworkUnavailable=false was canceled because it is already set", node.Name)
		return nil
	}

	if !routesCreated && condition != nil && condition.Status == v1.ConditionTrue {
		klog.V(2).Infof("set node %v with NodeNetworkUnavailable=true was canceled because it is already set", node.Name)
		return nil
	}

	klog.Infof("Patching node status %v with %v previous condition was:%+v", node.Name, routesCreated, condition)

	// either condition is not there, or has a value != to what we need
	// start setting it
	err := clientretry.RetryOnConflict(updateNetworkConditionBackoff, func() error {
		var err error
		// Patch could also fail, even though the chance is very slim. So we still do
		// patch in the retry loop.
		currentTime := metav1.Now()
		if routesCreated {
			err = nodeutil.SetNodeCondition(rc.kubeClient, types.NodeName(node.Name), v1.NodeCondition{
				Type:               v1.NodeNetworkUnavailable,
				Status:             v1.ConditionFalse,
				Reason:             "RouteCreated",
				Message:            "RouteController created a route",
				LastTransitionTime: currentTime,
			})
		} else {
			err = nodeutil.SetNodeCondition(rc.kubeClient, types.NodeName(node.Name), v1.NodeCondition{
				Type:               v1.NodeNetworkUnavailable,
				Status:             v1.ConditionTrue,
				Reason:             "NoRouteCreated",
				Message:            "RouteController failed to create a route",
				LastTransitionTime: currentTime,
			})
		}
		if err != nil {
			klog.V(4).Infof("Error updating node %s, retrying: %v", types.NodeName(node.Name), err)
		}
		return err
	})

	if err != nil {
		klog.Errorf("Error updating node %s: %v", node.Name, err)
	}

	return err
}

func (rc *RouteController) isResponsibleForRoute(route *cloudprovider.Route) bool {
	_, cidr, err := netutils.ParseCIDRSloppy(route.DestinationCIDR)
	if err != nil {
		klog.Errorf("Ignoring route %s, unparsable CIDR: %v", route.Name, err)
		return false
	}
	// Not responsible if this route's CIDR is not within our clusterCIDR
	lastIP := make([]byte, len(cidr.IP))
	for i := range lastIP {
		lastIP[i] = cidr.IP[i] | ^cidr.Mask[i]
	}

	// check across all cluster cidrs
	for _, clusterCIDR := range rc.clusterCIDRs {
		if clusterCIDR.Contains(cidr.IP) || clusterCIDR.Contains(lastIP) {
			return true
		}
	}
	return false
}

// checks if a node owns a route with a specific cidr
func hasRoute(rm map[types.NodeName][]*cloudprovider.Route, nodeName types.NodeName, cidr string) bool {
	if routes, ok := rm[nodeName]; ok {
		for _, route := range routes {
			if route.DestinationCIDR == cidr {
				return true
			}
		}
	}
	return false
}
