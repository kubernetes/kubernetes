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

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	coreinformers "k8s.io/client-go/informers/core/v1"
	discoveryinformers "k8s.io/client-go/informers/discovery/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

// ServiceHandler is an abstract interface of objects which receive
// notifications about service object changes.
type ServiceHandler interface {
	// OnServiceAdd is called whenever creation of new service object
	// is observed.
	OnServiceAdd(service *v1.Service)
	// OnServiceUpdate is called whenever modification of an existing
	// service object is observed.
	OnServiceUpdate(oldService, service *v1.Service)
	// OnServiceDelete is called whenever deletion of an existing service
	// object is observed.
	OnServiceDelete(service *v1.Service)
	// OnServiceSynced is called once all the initial event handlers were
	// called and the state is fully propagated to local cache.
	OnServiceSynced()
}

// EndpointSliceHandler is an abstract interface of objects which receive
// notifications about endpoint slice object changes.
type EndpointSliceHandler interface {
	// OnEndpointSliceAdd is called whenever creation of new endpoint slice
	// object is observed.
	OnEndpointSliceAdd(endpointSlice *discovery.EndpointSlice)
	// OnEndpointSliceUpdate is called whenever modification of an existing
	// endpoint slice object is observed.
	OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice *discovery.EndpointSlice)
	// OnEndpointSliceDelete is called whenever deletion of an existing
	// endpoint slice object is observed.
	OnEndpointSliceDelete(endpointSlice *discovery.EndpointSlice)
	// OnEndpointSlicesSynced is called once all the initial event handlers were
	// called and the state is fully propagated to local cache.
	OnEndpointSlicesSynced()
}

// EndpointSliceConfig tracks a set of endpoints configurations.
type EndpointSliceConfig struct {
	listerSynced  cache.InformerSynced
	eventHandlers []EndpointSliceHandler
}

// NewEndpointSliceConfig creates a new EndpointSliceConfig.
func NewEndpointSliceConfig(endpointSliceInformer discoveryinformers.EndpointSliceInformer, resyncPeriod time.Duration) *EndpointSliceConfig {
	result := &EndpointSliceConfig{}

	handlerRegistration, _ := endpointSliceInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    result.handleAddEndpointSlice,
			UpdateFunc: result.handleUpdateEndpointSlice,
			DeleteFunc: result.handleDeleteEndpointSlice,
		},
		resyncPeriod,
	)

	result.listerSynced = handlerRegistration.HasSynced

	return result
}

// RegisterEventHandler registers a handler which is called on every endpoint slice change.
func (c *EndpointSliceConfig) RegisterEventHandler(handler EndpointSliceHandler) {
	c.eventHandlers = append(c.eventHandlers, handler)
}

// Run waits for cache synced and invokes handlers after syncing.
func (c *EndpointSliceConfig) Run(stopCh <-chan struct{}) {
	klog.InfoS("Starting endpoint slice config controller")

	if !cache.WaitForNamedCacheSync("endpoint slice config", stopCh, c.listerSynced) {
		return
	}

	for _, h := range c.eventHandlers {
		klog.V(3).InfoS("Calling handler.OnEndpointSlicesSynced()")
		h.OnEndpointSlicesSynced()
	}
}

func (c *EndpointSliceConfig) handleAddEndpointSlice(obj interface{}) {
	endpointSlice, ok := obj.(*discovery.EndpointSlice)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %T", obj))
		return
	}
	for _, h := range c.eventHandlers {
		klog.V(4).InfoS("Calling handler.OnEndpointSliceAdd", "endpoints", klog.KObj(endpointSlice))
		h.OnEndpointSliceAdd(endpointSlice)
	}
}

func (c *EndpointSliceConfig) handleUpdateEndpointSlice(oldObj, newObj interface{}) {
	oldEndpointSlice, ok := oldObj.(*discovery.EndpointSlice)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %T", newObj))
		return
	}
	newEndpointSlice, ok := newObj.(*discovery.EndpointSlice)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %T", newObj))
		return
	}
	for _, h := range c.eventHandlers {
		klog.V(4).InfoS("Calling handler.OnEndpointSliceUpdate")
		h.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice)
	}
}

func (c *EndpointSliceConfig) handleDeleteEndpointSlice(obj interface{}) {
	endpointSlice, ok := obj.(*discovery.EndpointSlice)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %T", obj))
			return
		}
		if endpointSlice, ok = tombstone.Obj.(*discovery.EndpointSlice); !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %T", obj))
			return
		}
	}
	for _, h := range c.eventHandlers {
		klog.V(4).InfoS("Calling handler.OnEndpointsDelete")
		h.OnEndpointSliceDelete(endpointSlice)
	}
}

// ServiceConfig tracks a set of service configurations.
type ServiceConfig struct {
	listerSynced  cache.InformerSynced
	eventHandlers []ServiceHandler
}

// NewServiceConfig creates a new ServiceConfig.
func NewServiceConfig(serviceInformer coreinformers.ServiceInformer, resyncPeriod time.Duration) *ServiceConfig {
	result := &ServiceConfig{}

	handlerRegistration, _ := serviceInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    result.handleAddService,
			UpdateFunc: result.handleUpdateService,
			DeleteFunc: result.handleDeleteService,
		},
		resyncPeriod,
	)

	result.listerSynced = handlerRegistration.HasSynced

	return result
}

// RegisterEventHandler registers a handler which is called on every service change.
func (c *ServiceConfig) RegisterEventHandler(handler ServiceHandler) {
	c.eventHandlers = append(c.eventHandlers, handler)
}

// Run waits for cache synced and invokes handlers after syncing.
func (c *ServiceConfig) Run(stopCh <-chan struct{}) {
	klog.InfoS("Starting service config controller")

	if !cache.WaitForNamedCacheSync("service config", stopCh, c.listerSynced) {
		return
	}

	for i := range c.eventHandlers {
		klog.V(3).InfoS("Calling handler.OnServiceSynced()")
		c.eventHandlers[i].OnServiceSynced()
	}
}

func (c *ServiceConfig) handleAddService(obj interface{}) {
	service, ok := obj.(*v1.Service)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
		return
	}
	for i := range c.eventHandlers {
		klog.V(4).InfoS("Calling handler.OnServiceAdd")
		c.eventHandlers[i].OnServiceAdd(service)
	}
}

func (c *ServiceConfig) handleUpdateService(oldObj, newObj interface{}) {
	oldService, ok := oldObj.(*v1.Service)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", oldObj))
		return
	}
	service, ok := newObj.(*v1.Service)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", newObj))
		return
	}
	for i := range c.eventHandlers {
		klog.V(4).InfoS("Calling handler.OnServiceUpdate")
		c.eventHandlers[i].OnServiceUpdate(oldService, service)
	}
}

func (c *ServiceConfig) handleDeleteService(obj interface{}) {
	service, ok := obj.(*v1.Service)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
			return
		}
		if service, ok = tombstone.Obj.(*v1.Service); !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
			return
		}
	}
	for i := range c.eventHandlers {
		klog.V(4).InfoS("Calling handler.OnServiceDelete")
		c.eventHandlers[i].OnServiceDelete(service)
	}
}

// NodeHandler is an abstract interface of objects which receive
// notifications about node object changes.
type NodeHandler interface {
	// OnNodeAdd is called whenever creation of new node object
	// is observed.
	OnNodeAdd(node *v1.Node)
	// OnNodeUpdate is called whenever modification of an existing
	// node object is observed.
	OnNodeUpdate(oldNode, node *v1.Node)
	// OnNodeDelete is called whenever deletion of an existing node
	// object is observed.
	OnNodeDelete(node *v1.Node)
	// OnNodeSynced is called once all the initial event handlers were
	// called and the state is fully propagated to local cache.
	OnNodeSynced()
}

// NoopNodeHandler is a noop handler for proxiers that have not yet
// implemented a full NodeHandler.
type NoopNodeHandler struct{}

// OnNodeAdd is a noop handler for Node creates.
func (*NoopNodeHandler) OnNodeAdd(node *v1.Node) {}

// OnNodeUpdate is a noop handler for Node updates.
func (*NoopNodeHandler) OnNodeUpdate(oldNode, node *v1.Node) {}

// OnNodeDelete is a noop handler for Node deletes.
func (*NoopNodeHandler) OnNodeDelete(node *v1.Node) {}

// OnNodeSynced is a noop handler for Node syncs.
func (*NoopNodeHandler) OnNodeSynced() {}

var _ NodeHandler = &NoopNodeHandler{}

// NodeConfig tracks a set of node configurations.
// It accepts "set", "add" and "remove" operations of node via channels, and invokes registered handlers on change.
type NodeConfig struct {
	listerSynced  cache.InformerSynced
	eventHandlers []NodeHandler
}

// NewNodeConfig creates a new NodeConfig.
func NewNodeConfig(nodeInformer coreinformers.NodeInformer, resyncPeriod time.Duration) *NodeConfig {
	result := &NodeConfig{
		listerSynced: nodeInformer.Informer().HasSynced,
	}

	nodeInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    result.handleAddNode,
			UpdateFunc: result.handleUpdateNode,
			DeleteFunc: result.handleDeleteNode,
		},
		resyncPeriod,
	)

	return result
}

// RegisterEventHandler registers a handler which is called on every node change.
func (c *NodeConfig) RegisterEventHandler(handler NodeHandler) {
	c.eventHandlers = append(c.eventHandlers, handler)
}

// Run starts the goroutine responsible for calling registered handlers.
func (c *NodeConfig) Run(stopCh <-chan struct{}) {
	klog.InfoS("Starting node config controller")

	if !cache.WaitForNamedCacheSync("node config", stopCh, c.listerSynced) {
		return
	}

	for i := range c.eventHandlers {
		klog.V(3).InfoS("Calling handler.OnNodeSynced()")
		c.eventHandlers[i].OnNodeSynced()
	}
}

func (c *NodeConfig) handleAddNode(obj interface{}) {
	node, ok := obj.(*v1.Node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
		return
	}
	for i := range c.eventHandlers {
		klog.V(4).InfoS("Calling handler.OnNodeAdd")
		c.eventHandlers[i].OnNodeAdd(node)
	}
}

func (c *NodeConfig) handleUpdateNode(oldObj, newObj interface{}) {
	oldNode, ok := oldObj.(*v1.Node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", oldObj))
		return
	}
	node, ok := newObj.(*v1.Node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", newObj))
		return
	}
	for i := range c.eventHandlers {
		klog.V(5).InfoS("Calling handler.OnNodeUpdate")
		c.eventHandlers[i].OnNodeUpdate(oldNode, node)
	}
}

func (c *NodeConfig) handleDeleteNode(obj interface{}) {
	node, ok := obj.(*v1.Node)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
			return
		}
		if node, ok = tombstone.Obj.(*v1.Node); !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
			return
		}
	}
	for i := range c.eventHandlers {
		klog.V(4).InfoS("Calling handler.OnNodeDelete")
		c.eventHandlers[i].OnNodeDelete(node)
	}
}
