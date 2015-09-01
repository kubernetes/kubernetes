/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package lb

import (
	"fmt"
	"reflect"
	"time"

	compute "google.golang.org/api/compute/v1"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/cache"
	"k8s.io/kubernetes/pkg/client/unversioned/record"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

var (
	keyFunc          = framework.DeletionHandlingMetaNamespaceKeyFunc
	resyncPeriod     = 60 * time.Second
	lbControllerName = "lbcontroller"
)

// loadBalancerController watches the kubernetes api and adds/removes services
// from the loadbalancer, via loadBalancerConfig.
type loadBalancerController struct {
	client         *client.Client
	inpController  *framework.Controller
	nodeController *framework.Controller
	svcController  *framework.Controller
	inpLister      cache.StoreToIngressPointLister
	nodeLister     cache.StoreToNodeLister
	svcLister      cache.StoreToServiceLister
	clusterManager *ClusterManager
	recorder       record.EventRecorder
	nodeQueue      *taskQueue
	inpQueue       *taskQueue
	tr             *gceTranslator
}

// NewLoadBalancerController creates a controller for gce loadbalancers.
func NewLoadBalancerController(kubeClient *client.Client, lbName string) (*loadBalancerController, error) {
	var clusterManager *ClusterManager
	var err error
	if lbName == "" {
		lbName = lbControllerName
	}
	clusterManager, err = NewClusterManager(lbName)
	if err != nil {
		return nil, err
	}
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(kubeClient.Events(""))

	lbc := loadBalancerController{
		client:         kubeClient,
		clusterManager: clusterManager,
		recorder: eventBroadcaster.NewRecorder(
			api.EventSource{Component: "loadbalancer-controller"}),
	}
	lbc.nodeQueue = NewTaskQueue(lbc.syncNodes)
	lbc.inpQueue = NewTaskQueue(lbc.sync)

	pathHandlers := framework.ResourceEventHandlerFuncs{
		AddFunc:    lbc.inpQueue.enqueue,
		DeleteFunc: lbc.inpQueue.enqueue,
		UpdateFunc: func(old, cur interface{}) {
			if !reflect.DeepEqual(old, cur) {
				lbc.inpQueue.enqueue(cur)
			}
		},
	}
	lbc.inpLister.Store, lbc.inpController = framework.NewInformer(
		cache.NewListWatchFromClient(
			lbc.client, "ingressPoints", api.NamespaceAll, fields.Everything()),
		&api.IngressPoint{}, resyncPeriod, pathHandlers)

	nodeHandlers := framework.ResourceEventHandlerFuncs{
		AddFunc:    lbc.nodeQueue.enqueue,
		DeleteFunc: lbc.nodeQueue.enqueue,
		// Nodes are updated every 10s and we don't care.
	}

	lbc.nodeLister.Store, lbc.nodeController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return lbc.client.Get().
					Resource("nodes").
					FieldsSelectorParam(fields.Everything()).
					Do().
					Get()
			},
			WatchFunc: func(resourceVersion string) (watch.Interface, error) {
				return lbc.client.Get().
					Prefix("watch").
					Resource("nodes").
					FieldsSelectorParam(fields.Everything()).
					Param("resourceVersion", resourceVersion).Watch()
			},
		},
		&api.Node{}, 0, nodeHandlers)

	lbc.svcLister.Store, lbc.svcController = framework.NewInformer(
		cache.NewListWatchFromClient(
			lbc.client, "services", api.NamespaceAll, fields.Everything()),
		&api.Service{}, resyncPeriod, framework.ResourceEventHandlerFuncs{})

	lbc.tr = &gceTranslator{&lbc}
	glog.Infof("Created new loadbalancer controller")

	return &lbc, nil
}

// Run starts the loadbalancer controller.
func (lbc *loadBalancerController) Run(stopCh <-chan struct{}) {
	glog.Infof("Starting loadbalancer controller")
	go lbc.inpController.Run(stopCh)
	go lbc.nodeController.Run(stopCh)
	go lbc.svcController.Run(stopCh)
	go lbc.inpQueue.run(time.Second, stopCh)
	go lbc.nodeQueue.run(time.Second, stopCh)
	<-stopCh
	glog.Infof("Shutting down Loadbalancer Controller")
}

// sync manages the syncing of backends and pathmaps.
func (lbc *loadBalancerController) sync(key string) {
	glog.Infof("Syncing %v", key)

	paths, err := lbc.inpLister.List()
	if err != nil {
		lbc.inpQueue.requeue(key, err)
		return
	}

	// Create/delete cluster-wide shared resources.
	// TODO: If this turns into a bottleneck, split them out
	// into independent sync pools.
	if err := lbc.clusterManager.SyncL7s(
		lbc.inpLister.Store.ListKeys()); err != nil {
		lbc.inpQueue.requeue(key, err)
		return
	}
	if err := lbc.clusterManager.SyncBackends(
		lbc.tr.toNodePorts(&paths)); err != nil {
		lbc.inpQueue.requeue(key, err)
		return
	}

	// Deal with the single loadbalancer that came through the watch
	obj, inpExists, err := lbc.inpLister.Store.GetByKey(key)
	if err != nil {
		lbc.inpQueue.requeue(key, err)
		return
	}
	if !inpExists {
		return
	}
	l7, err := lbc.clusterManager.GetL7(key)
	if err != nil {
		lbc.inpQueue.requeue(key, err)
		return
	}

	inp := *obj.(*api.IngressPoint)
	if urlMap, err := lbc.tr.toUrlMap(&inp); err != nil {
		lbc.inpQueue.requeue(key, err)
	} else if err := l7.UpdateUrlMap(urlMap); err != nil {
		lbc.inpQueue.requeue(key, err)
	} else {
		glog.Infof("Finished syncing %v", key)
	}
	return
}

// syncNodes manages the syncing of kubernetes nodes to gce instance groups.
// The instancegroups are referenced by loadbalancer backends.
func (lbc *loadBalancerController) syncNodes(key string) {
	kubeNodes, err := lbc.nodeLister.List()
	if err != nil {
		lbc.nodeQueue.requeue(key, err)
		return
	}
	nodeNames := []string{}
	// TODO: delete unhealthy kubernetes nodes from cluster?
	for _, n := range kubeNodes.Items {
		nodeNames = append(nodeNames, n.Name)
	}
	if err := lbc.clusterManager.SyncNodes(nodeNames); err != nil {
		lbc.nodeQueue.requeue(key, err)
	}
	return
}

// gceTranslator helps with kubernetes -> gce api conversion.
type gceTranslator struct {
	*loadBalancerController
}

// toUrlMap converts a pathmap to a map of subdomain: url-regex: gce backend.
func (t *gceTranslator) toUrlMap(inp *api.IngressPoint) (gceUrlMap, error) {
	subdomainUrlBackend := map[string]map[string]*compute.BackendService{}
	for subdomain, paths := range inp.Spec.PathMap {
		urlToBackend := map[string]*compute.BackendService{}
		for _, p := range paths {
			port, err := t.getServiceNodePort(p.Service)
			if err != nil || port == 0 {
				glog.Infof("Could not find nodeport %v", err)
				continue
			}

			backend, err := t.clusterManager.GetBackend(int64(port))
			if err != nil {
				return map[string]map[string]*compute.BackendService{},
					fmt.Errorf(
						"No backend for pathmap %v, port %v", inp.Name, port)
			}
			urlToBackend[p.Url] = backend
		}
		subdomainUrlBackend[subdomain] = urlToBackend
	}
	return subdomainUrlBackend, nil
}

// getServiceNodePort looks in the svc store for a matching service:port,
// and returns the nodeport.
func (t *gceTranslator) getServiceNodePort(s api.ServiceRef) (int, error) {
	obj, exists, err := t.svcLister.Store.Get(
		&api.Service{
			ObjectMeta: api.ObjectMeta{
				Name:      s.Name,
				Namespace: s.Namespace,
			},
		})
	if !exists {
		return 0, fmt.Errorf(
			"Service %v/%v not found in store: %v",
			s.Namespace, s.Name, t.svcLister.Store.ListKeys())
	}
	if err != nil {
		return 0, err
	}
	svc := obj.(*api.Service)
	for _, p := range svc.Spec.Ports {
		if p.Port == s.Port {
			return p.NodePort, nil
		}
	}
	return 0, fmt.Errorf("No nodeport found for %+v", s)
}

// toNodePorts converts a pathlist to a flat list of nodeports.
func (t *gceTranslator) toNodePorts(paths *api.IngressPointList) []int64 {
	knownPorts := []int64{}
	for _, inp := range paths.Items {
		for _, subdomainToPath := range inp.Spec.PathMap {
			for _, path := range subdomainToPath {
				port, err := t.getServiceNodePort(path.Service)
				if err != nil || port == 0 {
					glog.Infof("Could not find nodeport %v", err)
					continue
				}
				knownPorts = append(knownPorts, int64(port))
			}
		}
	}
	return knownPorts
}
