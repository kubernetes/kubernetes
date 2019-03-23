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

	"k8s.io/klog"

	"k8s.io/api/core/v1"
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
	"k8s.io/kubernetes/pkg/controller"
	nodeutil "k8s.io/kubernetes/pkg/controller/util/node"
	"k8s.io/kubernetes/pkg/util/metrics"
	utilnode "k8s.io/kubernetes/pkg/util/node"
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
	clusterCIDR      *net.IPNet
	nodeLister       corelisters.NodeLister
	nodeListerSynced cache.InformerSynced
	broadcaster      record.EventBroadcaster
	recorder         record.EventRecorder
}

func New(routes cloudprovider.Routes, kubeClient clientset.Interface, nodeInformer coreinformers.NodeInformer, clusterName string, clusterCIDR *net.IPNet) *RouteController {
	if kubeClient != nil && kubeClient.CoreV1().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("route_controller", kubeClient.CoreV1().RESTClient().GetRateLimiter())
	}

	if clusterCIDR == nil {
		klog.Fatal("RouteController: Must specify clusterCIDR.")
	}

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(klog.Infof)
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "route_controller"})

	rc := &RouteController{
		routes:           routes,
		kubeClient:       kubeClient,
		clusterName:      clusterName,
		clusterCIDR:      clusterCIDR,
		nodeLister:       nodeInformer.Lister(),
		nodeListerSynced: nodeInformer.Informer().HasSynced,
		broadcaster:      eventBroadcaster,
		recorder:         recorder,
	}

	return rc
}

func (rc *RouteController) Run(stopCh <-chan struct{}, syncPeriod time.Duration) {
	defer utilruntime.HandleCrash()

	klog.Info("Starting route controller")
	defer klog.Info("Shutting down route controller")

	if !controller.WaitForCacheSync("route", stopCh, rc.nodeListerSynced) {
		return
	}

	if rc.broadcaster != nil {
		rc.broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: rc.kubeClient.CoreV1().Events("")})
	}

	// TODO: If we do just the full Resync every 5 minutes (default value)
	// that means that we may wait up to 5 minutes before even starting
	// creating a route for it. This is bad.
	// We should have a watch on node and if we observe a new node (with CIDR?)
	// trigger reconciliation for that node.
	go wait.NonSlidingUntil(func() {
		if err := rc.reconcileNodeRoutes(); err != nil {
			klog.Errorf("Couldn't reconcile node routes: %v", err)
		}
	}, syncPeriod, stopCh)

	<-stopCh
}

func (rc *RouteController) reconcileNodeRoutes() error {
	routeList, err := rc.routes.ListRoutes(context.TODO(), rc.clusterName)
	if err != nil {
		return fmt.Errorf("error listing routes: %v", err)
	}
	nodes, err := rc.nodeLister.List(labels.Everything())
	if err != nil {
		return fmt.Errorf("error listing nodes: %v", err)
	}
	return rc.reconcile(nodes, routeList)
}

func (rc *RouteController) reconcile(nodes []*v1.Node, routes []*cloudprovider.Route) error {
	// nodeCIDRs maps nodeName->nodeCIDR
	nodeCIDRs := make(map[types.NodeName]string)
	// routeMap maps routeTargetNode->route
	routeMap := make(map[types.NodeName]*cloudprovider.Route)
	for _, route := range routes {
		if route.TargetNode != "" {
			routeMap[route.TargetNode] = route
		}
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
				err := clientretry.RetryOnConflict(updateNetworkConditionBackoff, func() error {
					startTime := time.Now()
					// Ensure that we don't have more than maxConcurrentRouteCreations
					// CreateRoute calls in flight.
					rateLimiter <- struct{}{}
					klog.Infof("Creating route for node %s %s with hint %s, throttled %v", nodeName, route.DestinationCIDR, nameHint, time.Since(startTime))
					err := rc.routes.CreateRoute(context.TODO(), rc.clusterName, nameHint, route)
					<-rateLimiter

					rc.updateNetworkingCondition(nodeName, err == nil)
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
						}
						klog.V(4).Infof(msg)
						return err
					}
					klog.Infof("Created route for node %s %s with hint %s after %v", nodeName, route.DestinationCIDR, nameHint, time.Now().Sub(startTime))
					return nil
				})
				if err != nil {
					klog.Errorf("Could not create route %s %s for node %s: %v", nameHint, route.DestinationCIDR, nodeName, err)
				}
			}(nodeName, nameHint, route)
		} else {
			// Update condition only if it doesn't reflect the current state.
			_, condition := nodeutil.GetNodeCondition(&node.Status, v1.NodeNetworkUnavailable)
			if condition == nil || condition.Status != v1.ConditionFalse {
				rc.updateNetworkingCondition(types.NodeName(node.Name), true)
			}
		}
		nodeCIDRs[nodeName] = node.Spec.PodCIDR
	}
	for _, route := range routes {
		if rc.isResponsibleForRoute(route) {
			// Check if this route is a blackhole, or applies to a node we know about & has an incorrect CIDR.
			if route.Blackhole || (nodeCIDRs[route.TargetNode] != route.DestinationCIDR) {
				wg.Add(1)
				// Delete the route.
				go func(route *cloudprovider.Route, startTime time.Time) {
					defer wg.Done()
					klog.Infof("Deleting route %s %s", route.Name, route.DestinationCIDR)
					if err := rc.routes.DeleteRoute(context.TODO(), rc.clusterName, route); err != nil {
						klog.Errorf("Could not delete route %s %s after %v: %v", route.Name, route.DestinationCIDR, time.Since(startTime), err)
					} else {
						klog.Infof("Deleted route %s %s after %v", route.Name, route.DestinationCIDR, time.Since(startTime))
					}
				}(route, time.Now())
			}
		}
	}
	wg.Wait()
	return nil
}

func (rc *RouteController) updateNetworkingCondition(nodeName types.NodeName, routeCreated bool) error {
	err := clientretry.RetryOnConflict(updateNetworkConditionBackoff, func() error {
		var err error
		// Patch could also fail, even though the chance is very slim. So we still do
		// patch in the retry loop.
		currentTime := metav1.Now()
		if routeCreated {
			err = utilnode.SetNodeCondition(rc.kubeClient, nodeName, v1.NodeCondition{
				Type:               v1.NodeNetworkUnavailable,
				Status:             v1.ConditionFalse,
				Reason:             "RouteCreated",
				Message:            "RouteController created a route",
				LastTransitionTime: currentTime,
			})
		} else {
			err = utilnode.SetNodeCondition(rc.kubeClient, nodeName, v1.NodeCondition{
				Type:               v1.NodeNetworkUnavailable,
				Status:             v1.ConditionTrue,
				Reason:             "NoRouteCreated",
				Message:            "RouteController failed to create a route",
				LastTransitionTime: currentTime,
			})
		}
		if err != nil {
			klog.V(4).Infof("Error updating node %s, retrying: %v", nodeName, err)
		}
		return err
	})

	if err != nil {
		klog.Errorf("Error updating node %s: %v", nodeName, err)
	}

	return err
}

func (rc *RouteController) isResponsibleForRoute(route *cloudprovider.Route) bool {
	_, cidr, err := net.ParseCIDR(route.DestinationCIDR)
	if err != nil {
		klog.Errorf("Ignoring route %s, unparsable CIDR: %v", route.Name, err)
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
