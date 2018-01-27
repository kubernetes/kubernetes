/*
Copyright 2014 The Kubernetes Authors.

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

package config

import (
	"fmt"
	"time"

	"github.com/golang/glog"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	api "k8s.io/kubernetes/pkg/apis/core"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	coreinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion/core/internalversion"
	listers "k8s.io/kubernetes/pkg/client/listers/core/internalversion"
	"k8s.io/kubernetes/pkg/controller"
)

// ServiceHandler is an abstract interface of objects which receive
// notifications about service object changes.
type ServiceHandler interface {
	// OnServiceAdd is called whenever creation of new service object
	// is observed.
	OnServiceAdd(service *api.Service)
	// OnServiceUpdate is called whenever modification of an existing
	// service object is observed.
	OnServiceUpdate(oldService, service *api.Service)
	// OnServiceDelete is called whenever deletion of an existing service
	// object is observed.
	OnServiceDelete(service *api.Service)
	// OnServiceSynced is called once all the initial even handlers were
	// called and the state is fully propagated to local cache.
	OnServiceSynced()
}

// EndpointsHandler is an abstract interface of objects which receive
// notifications about endpoints object changes.
type EndpointsHandler interface {
	// OnEndpointsAdd is called whenever creation of new endpoints object
	// is observed.
	OnEndpointsAdd(endpoints *api.Endpoints)
	// OnEndpointsUpdate is called whenever modification of an existing
	// endpoints object is observed.
	OnEndpointsUpdate(oldEndpoints, endpoints *api.Endpoints)
	// OnEndpointsDelete is called whever deletion of an existing endpoints
	// object is observed.
	OnEndpointsDelete(endpoints *api.Endpoints)
	// OnEndpointsSynced is called once all the initial event handlers were
	// called and the state is fully propagated to local cache.
	OnEndpointsSynced()
}

// EndpointsConfig tracks a set of endpoints configurations.
// It accepts "set", "add" and "remove" operations of endpoints via channels, and invokes registered handlers on change.
type EndpointsConfig struct {
	lister        listers.EndpointsLister
	listerSynced  cache.InformerSynced
	eventHandlers []EndpointsHandler
}

// NewEndpointsConfig creates a new EndpointsConfig.
func NewEndpointsConfig(endpointsInformer coreinformers.EndpointsInformer, resyncPeriod time.Duration) *EndpointsConfig {
	result := &EndpointsConfig{
		lister:       endpointsInformer.Lister(),
		listerSynced: endpointsInformer.Informer().HasSynced,
	}

	endpointsInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    result.handleAddEndpoints,
			UpdateFunc: result.handleUpdateEndpoints,
			DeleteFunc: result.handleDeleteEndpoints,
		},
		resyncPeriod,
	)

	return result
}

// RegisterEventHandler registers a handler which is called on every endpoints change.
func (c *EndpointsConfig) RegisterEventHandler(handler EndpointsHandler) {
	c.eventHandlers = append(c.eventHandlers, handler)
}

// Run starts the goroutine responsible for calling registered handlers.
func (c *EndpointsConfig) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	glog.Info("Starting endpoints config controller")
	defer glog.Info("Shutting down endpoints config controller")

	if !controller.WaitForCacheSync("endpoints config", stopCh, c.listerSynced) {
		return
	}

	for i := range c.eventHandlers {
		glog.V(3).Infof("Calling handler.OnEndpointsSynced()")
		c.eventHandlers[i].OnEndpointsSynced()
	}

	<-stopCh
}

func (c *EndpointsConfig) handleAddEndpoints(obj interface{}) {
	endpoints, ok := obj.(*api.Endpoints)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
		return
	}
	for i := range c.eventHandlers {
		glog.V(4).Infof("Calling handler.OnEndpointsAdd")
		c.eventHandlers[i].OnEndpointsAdd(endpoints)
	}
}

func (c *EndpointsConfig) handleUpdateEndpoints(oldObj, newObj interface{}) {
	oldEndpoints, ok := oldObj.(*api.Endpoints)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", oldObj))
		return
	}
	endpoints, ok := newObj.(*api.Endpoints)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", newObj))
		return
	}
	for i := range c.eventHandlers {
		glog.V(4).Infof("Calling handler.OnEndpointsUpdate")
		c.eventHandlers[i].OnEndpointsUpdate(oldEndpoints, endpoints)
	}
}

func (c *EndpointsConfig) handleDeleteEndpoints(obj interface{}) {
	endpoints, ok := obj.(*api.Endpoints)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
			return
		}
		if endpoints, ok = tombstone.Obj.(*api.Endpoints); !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
			return
		}
	}
	for i := range c.eventHandlers {
		glog.V(4).Infof("Calling handler.OnEndpointsDelete")
		c.eventHandlers[i].OnEndpointsDelete(endpoints)
	}
}

// ServiceConfig tracks a set of service configurations.
// It accepts "set", "add" and "remove" operations of services via channels, and invokes registered handlers on change.
type ServiceConfig struct {
	lister        listers.ServiceLister
	listerSynced  cache.InformerSynced
	eventHandlers []ServiceHandler
}

// NewServiceConfig creates a new ServiceConfig.
func NewServiceConfig(serviceInformer coreinformers.ServiceInformer, resyncPeriod time.Duration) *ServiceConfig {
	result := &ServiceConfig{
		lister:       serviceInformer.Lister(),
		listerSynced: serviceInformer.Informer().HasSynced,
	}

	serviceInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    result.handleAddService,
			UpdateFunc: result.handleUpdateService,
			DeleteFunc: result.handleDeleteService,
		},
		resyncPeriod,
	)

	return result
}

// RegisterEventHandler registers a handler which is called on every service change.
func (c *ServiceConfig) RegisterEventHandler(handler ServiceHandler) {
	c.eventHandlers = append(c.eventHandlers, handler)
}

// Run starts the goroutine responsible for calling
// registered handlers.
func (c *ServiceConfig) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	glog.Info("Starting service config controller")
	defer glog.Info("Shutting down service config controller")

	if !controller.WaitForCacheSync("service config", stopCh, c.listerSynced) {
		return
	}

	for i := range c.eventHandlers {
		glog.V(3).Infof("Calling handler.OnServiceSynced()")
		c.eventHandlers[i].OnServiceSynced()
	}

	<-stopCh
}

func (c *ServiceConfig) handleAddService(obj interface{}) {
	service, ok := obj.(*api.Service)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
		return
	}
	for i := range c.eventHandlers {
		glog.V(4).Infof("Calling handler.OnServiceAdd")
		c.eventHandlers[i].OnServiceAdd(service)
	}
}

func (c *ServiceConfig) handleUpdateService(oldObj, newObj interface{}) {
	oldService, ok := oldObj.(*api.Service)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", oldObj))
		return
	}
	service, ok := newObj.(*api.Service)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", newObj))
		return
	}
	for i := range c.eventHandlers {
		glog.V(4).Infof("Calling handler.OnServiceUpdate")
		c.eventHandlers[i].OnServiceUpdate(oldService, service)
	}
}

func (c *ServiceConfig) handleDeleteService(obj interface{}) {
	service, ok := obj.(*api.Service)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
			return
		}
		if service, ok = tombstone.Obj.(*api.Service); !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
			return
		}
	}
	for i := range c.eventHandlers {
		glog.V(4).Infof("Calling handler.OnServiceDelete")
		c.eventHandlers[i].OnServiceDelete(service)
	}
}

// NodeConfig tracks a node configurations.
type NodeConfig struct {
	informer      cache.SharedInformer
	eventHandlers []NodeHandler
}

// RegisterEventHandler registers a handler which is called on every node change.
func (c *NodeConfig) RegisterEventHandler(handler NodeHandler) {
	c.eventHandlers = append(c.eventHandlers, handler)
}

func (c *NodeConfig) handleAddNode(obj interface{}) {
	node, ok := obj.(*api.Node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
		return
	}
	for i := range c.eventHandlers {
		glog.V(4).Infof("Calling handler.OnNodeAdd")
		c.eventHandlers[i].OnNodeAdd(node)
	}
}

func (c *NodeConfig) handleUpdateNode(oldObj, newObj interface{}) {
	oldNode, ok := oldObj.(*api.Node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", oldObj))
		return
	}
	node, ok := newObj.(*api.Node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", newObj))
		return
	}
	for i := range c.eventHandlers {
		glog.V(4).Infof("Calling handler.OnNodeUpdate")
		c.eventHandlers[i].OnNodeUpdate(oldNode, node)
	}
}

func (c *NodeConfig) handleDeleteNode(obj interface{}) {
	node, ok := obj.(*api.Node)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
			return
		}
		if node, ok = tombstone.Obj.(*api.Node); !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
			return
		}
	}
	for i := range c.eventHandlers {
		glog.V(4).Infof("Calling handler.OnNodeDelete")
		c.eventHandlers[i].OnNodeDelete(node)
	}
}

// NodeHandler is an abstract interface of objects which receive
// notifications about node object changes.
type NodeHandler interface {
	// OnNodeAdd is called whenever creation of new service object
	// is observed.
	OnNodeAdd(node *api.Node)
	// OnNodeUpdate is called whenever modification of an existing
	// service object is observed.
	OnNodeUpdate(oldNode, node *api.Node)
	// OnNodeDelete is called whenever deletion of an existing node
	// object is observed.
	OnNodeDelete(node *api.Node)
	// OnNodeSynced is called once all the initial even handlers were
	// called and the state is fully propagated to local cache.
	OnNodeSynced()
}

// NewNodeConfig creates a new NodeConfig.
func NewNodeConfig(client clientset.Interface, nodeName string, resyncPeriod time.Duration) *NodeConfig {
	// select nodes by name
	fieldselector := fields.OneTermEqualSelector("metadata.name", nodeName)

	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (kuberuntime.Object, error) {
			return client.Core().Nodes().List(metav1.ListOptions{
				FieldSelector: fieldselector.String(),
			})
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			return client.Core().Nodes().Watch(metav1.ListOptions{
				FieldSelector:   fieldselector.String(),
				ResourceVersion: options.ResourceVersion,
			})
		},
	}

	result := &NodeConfig{}

	handler := cache.ResourceEventHandlerFuncs{
		AddFunc:    result.handleAddNode,
		UpdateFunc: result.handleUpdateNode,
		DeleteFunc: result.handleDeleteNode,
	}

	informer := cache.NewSharedInformer(lw, &api.Node{}, resyncPeriod)
	informer.AddEventHandler(handler)

	result.informer = informer

	return result
}

// Run starts the NodeConfig controller.
func (c *NodeConfig) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	glog.Info("Starting node config controller")
	defer glog.Info("Shutting down node config controller")

	c.informer.Run(stopCh)

	<-stopCh
}
