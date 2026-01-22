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
	"context"
	"fmt"
	"reflect"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	networkingv1 "k8s.io/api/networking/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	v1informers "k8s.io/client-go/informers/core/v1"
	discoveryv1informers "k8s.io/client-go/informers/discovery/v1"
	networkingv1informers "k8s.io/client-go/informers/networking/v1"
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
	OnEndpointSliceAdd(endpointSlice *discoveryv1.EndpointSlice)
	// OnEndpointSliceUpdate is called whenever modification of an existing
	// endpoint slice object is observed.
	OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice *discoveryv1.EndpointSlice)
	// OnEndpointSliceDelete is called whenever deletion of an existing
	// endpoint slice object is observed.
	OnEndpointSliceDelete(endpointSlice *discoveryv1.EndpointSlice)
	// OnEndpointSlicesSynced is called once all the initial event handlers were
	// called and the state is fully propagated to local cache.
	OnEndpointSlicesSynced()
}

// EndpointSliceConfig tracks a set of endpoints configurations.
type EndpointSliceConfig struct {
	listerSynced  cache.InformerSynced
	eventHandlers []EndpointSliceHandler
	logger        klog.Logger
}

// NewEndpointSliceConfig creates a new EndpointSliceConfig.
func NewEndpointSliceConfig(ctx context.Context, endpointSliceInformer discoveryv1informers.EndpointSliceInformer, resyncPeriod time.Duration) *EndpointSliceConfig {
	result := &EndpointSliceConfig{
		logger: klog.FromContext(ctx),
	}

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
	c.logger.Info("Starting endpoint slice config controller")

	if !cache.WaitForNamedCacheSync("endpoint slice config", stopCh, c.listerSynced) {
		return
	}

	for _, h := range c.eventHandlers {
		c.logger.V(3).Info("Calling handler.OnEndpointSlicesSynced()")
		h.OnEndpointSlicesSynced()
	}
}

func (c *EndpointSliceConfig) handleAddEndpointSlice(obj interface{}) {
	endpointSlice, ok := obj.(*discoveryv1.EndpointSlice)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %T", obj))
		return
	}
	for _, h := range c.eventHandlers {
		c.logger.V(4).Info("Calling handler.OnEndpointSliceAdd", "endpoints", klog.KObj(endpointSlice))
		h.OnEndpointSliceAdd(endpointSlice)
	}
}

func (c *EndpointSliceConfig) handleUpdateEndpointSlice(oldObj, newObj interface{}) {
	oldEndpointSlice, ok := oldObj.(*discoveryv1.EndpointSlice)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %T", oldObj))
		return
	}
	newEndpointSlice, ok := newObj.(*discoveryv1.EndpointSlice)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %T", newObj))
		return
	}
	for _, h := range c.eventHandlers {
		c.logger.V(4).Info("Calling handler.OnEndpointSliceUpdate")
		h.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice)
	}
}

func (c *EndpointSliceConfig) handleDeleteEndpointSlice(obj interface{}) {
	endpointSlice, ok := obj.(*discoveryv1.EndpointSlice)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %T", obj))
			return
		}
		if endpointSlice, ok = tombstone.Obj.(*discoveryv1.EndpointSlice); !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %T", obj))
			return
		}
	}
	for _, h := range c.eventHandlers {
		c.logger.V(4).Info("Calling handler.OnEndpointsDelete")
		h.OnEndpointSliceDelete(endpointSlice)
	}
}

// ServiceConfig tracks a set of service configurations.
type ServiceConfig struct {
	listerSynced  cache.InformerSynced
	eventHandlers []ServiceHandler
	logger        klog.Logger
}

// NewServiceConfig creates a new ServiceConfig.
func NewServiceConfig(ctx context.Context, serviceInformer v1informers.ServiceInformer, resyncPeriod time.Duration) *ServiceConfig {
	result := &ServiceConfig{
		logger: klog.FromContext(ctx),
	}

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
	c.logger.Info("Starting service config controller")

	if !cache.WaitForNamedCacheSync("service config", stopCh, c.listerSynced) {
		return
	}

	for i := range c.eventHandlers {
		c.logger.V(3).Info("Calling handler.OnServiceSynced()")
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
		c.logger.V(4).Info("Calling handler.OnServiceAdd")
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
		c.logger.V(4).Info("Calling handler.OnServiceUpdate")
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
		c.logger.V(4).Info("Calling handler.OnServiceDelete")
		c.eventHandlers[i].OnServiceDelete(service)
	}
}

// NodeHandler is an abstract interface of objects which receive
// notifications about node object changes.
type NodeHandler interface {
	// OnNodeChange is called whenever creation or modification
	// of node object is observed.
	OnNodeChange(node *v1.Node)
	// OnNodeDelete is called whenever deletion of an existing node
	// object is observed.
	OnNodeDelete(node *v1.Node)
	// OnNodeSynced is called once all the initial event handlers were
	// called and the state is fully propagated to local cache.
	OnNodeSynced()
}

// NodeConfig tracks a set of node configurations.
// It accepts "set", "add" and "remove" operations of node via channels, and invokes registered handlers on change.
type NodeConfig struct {
	listerSynced  cache.InformerSynced
	eventHandlers []NodeHandler
	logger        klog.Logger
}

// NewNodeConfig creates a new NodeConfig.
func NewNodeConfig(ctx context.Context, nodeInformer v1informers.NodeInformer, resyncPeriod time.Duration) *NodeConfig {
	result := &NodeConfig{
		logger: klog.FromContext(ctx),
	}

	handlerRegistration, _ := nodeInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    func(obj interface{}) { result.handleChangeNode(obj) },
			UpdateFunc: func(_, newObj interface{}) { result.handleChangeNode(newObj) },
			DeleteFunc: result.handleDeleteNode,
		},
		resyncPeriod,
	)

	result.listerSynced = handlerRegistration.HasSynced

	return result
}

// RegisterEventHandler registers a handler which is called on every node change.
func (c *NodeConfig) RegisterEventHandler(handler NodeHandler) {
	c.eventHandlers = append(c.eventHandlers, handler)
}

// Run starts the goroutine responsible for calling registered handlers.
func (c *NodeConfig) Run(stopCh <-chan struct{}) {
	c.logger.Info("Starting node config controller")

	if !cache.WaitForNamedCacheSync("node config", stopCh, c.listerSynced) {
		return
	}

	for i := range c.eventHandlers {
		c.logger.V(3).Info("Calling handler.OnNodeSynced()")
		c.eventHandlers[i].OnNodeSynced()
	}
}

func (c *NodeConfig) handleChangeNode(obj interface{}) {
	node, ok := obj.(*v1.Node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
		return
	}
	for i := range c.eventHandlers {
		c.logger.V(4).Info("Calling handler.OnNodeChange")
		c.eventHandlers[i].OnNodeChange(node)
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
		c.logger.V(4).Info("Calling handler.OnNodeDelete")
		c.eventHandlers[i].OnNodeDelete(node)
	}
}

// ServiceCIDRHandler is an abstract interface of objects which receive
// notifications about ServiceCIDR object changes.
type ServiceCIDRHandler interface {
	// OnServiceCIDRsChanged is called whenever a change is observed
	// in any of the ServiceCIDRs, and provides complete list of service cidrs.
	OnServiceCIDRsChanged(cidrs []string)
}

// ServiceCIDRConfig tracks a set of service configurations.
type ServiceCIDRConfig struct {
	listerSynced  cache.InformerSynced
	eventHandlers []ServiceCIDRHandler
	mu            sync.Mutex
	cidrs         sets.Set[string]
	logger        klog.Logger
}

// NewServiceCIDRConfig creates a new ServiceCIDRConfig.
func NewServiceCIDRConfig(ctx context.Context, serviceCIDRInformer networkingv1informers.ServiceCIDRInformer, resyncPeriod time.Duration) *ServiceCIDRConfig {
	result := &ServiceCIDRConfig{
		cidrs:  sets.New[string](),
		logger: klog.FromContext(ctx),
	}

	handlerRegistration, _ := serviceCIDRInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				result.handleServiceCIDREvent(nil, obj)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				result.handleServiceCIDREvent(oldObj, newObj)
			},
			DeleteFunc: func(obj interface{}) {
				result.handleServiceCIDREvent(obj, nil)
			},
		},
		resyncPeriod,
	)

	result.listerSynced = handlerRegistration.HasSynced

	return result
}

// RegisterEventHandler registers a handler which is called on every ServiceCIDR change.
func (c *ServiceCIDRConfig) RegisterEventHandler(handler ServiceCIDRHandler) {
	c.eventHandlers = append(c.eventHandlers, handler)
}

// Run waits for cache synced and invokes handlers after syncing.
func (c *ServiceCIDRConfig) Run(stopCh <-chan struct{}) {
	c.logger.Info("Starting serviceCIDR config controller")

	if !cache.WaitForNamedCacheSync("serviceCIDR config", stopCh, c.listerSynced) {
		return
	}
	c.handleServiceCIDREvent(nil, nil)
}

// handleServiceCIDREvent is a helper function to handle Add, Update and Delete
// events on ServiceCIDR objects and call downstream event handlers.
func (c *ServiceCIDRConfig) handleServiceCIDREvent(oldObj, newObj interface{}) {
	var oldServiceCIDR, newServiceCIDR *networkingv1.ServiceCIDR
	var ok bool

	if oldObj != nil {
		oldServiceCIDR, ok = oldObj.(*networkingv1.ServiceCIDR)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", oldObj))
			return
		}
	}

	if newObj != nil {
		newServiceCIDR, ok = newObj.(*networkingv1.ServiceCIDR)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", newObj))
			return
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if oldServiceCIDR != nil {
		c.cidrs.Delete(oldServiceCIDR.Spec.CIDRs...)
	}

	if newServiceCIDR != nil {
		c.cidrs.Insert(newServiceCIDR.Spec.CIDRs...)
	}

	for i := range c.eventHandlers {
		c.logger.V(4).Info("Calling handler.OnServiceCIDRsChanged")
		c.eventHandlers[i].OnServiceCIDRsChanged(c.cidrs.UnsortedList())
	}
}

// NodeTopologyHandler is an abstract interface for objects which receive
// notifications about changes in proxy relevant node topology labels.
type NodeTopologyHandler interface {
	// OnTopologyChange is called whenever a change is observed in proxy
	// relevant node topology labels, and provides the observed change.
	OnTopologyChange(topologyLabels map[string]string)
}

// NodeTopologyConfig tracks node topology labels.
type NodeTopologyConfig struct {
	listerSynced   cache.InformerSynced
	eventHandlers  []NodeTopologyHandler
	topologyLabels map[string]string
	logger         klog.Logger
}

// NewNodeTopologyConfig creates a new NodeTopologyConfig.
func NewNodeTopologyConfig(ctx context.Context, nodeInformer v1informers.NodeInformer, resyncPeriod time.Duration) *NodeTopologyConfig {
	return newNodeTopologyConfig(ctx, nodeInformer, resyncPeriod, nil)
}

// newNodeTopologyConfig implements NewNodeTopologyConfig by additionally consuming a callback function which is invoked when
// event handler completes processing and is only used for testing.
func newNodeTopologyConfig(ctx context.Context, nodeInformer v1informers.NodeInformer, resyncPeriod time.Duration, callback func()) *NodeTopologyConfig {
	result := &NodeTopologyConfig{
		logger:         klog.FromContext(ctx),
		topologyLabels: make(map[string]string),
	}

	handlerRegistration, _ := nodeInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				result.handleNodeEvent(obj)
				if callback != nil {
					callback()
				}
			},
			UpdateFunc: func(_, newObj interface{}) {
				result.handleNodeEvent(newObj)
				if callback != nil {
					callback()
				}
			},
			DeleteFunc: func(_ interface{}) {},
		},
		resyncPeriod,
	)
	result.listerSynced = handlerRegistration.HasSynced

	return result
}

// RegisterEventHandler registers a handler which is called on Node object change.
func (n *NodeTopologyConfig) RegisterEventHandler(handler NodeTopologyHandler) {
	n.eventHandlers = append(n.eventHandlers, handler)
}

// handleNodeEvent is a helper function to handle Add, Update and Delete
// events on Node objects and call downstream event handlers.
func (n *NodeTopologyConfig) handleNodeEvent(obj interface{}) {
	node, ok := obj.(*v1.Node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
		return
	}

	topologyLabels := make(map[string]string)
	if _, ok = node.Labels[v1.LabelTopologyZone]; ok {
		topologyLabels[v1.LabelTopologyZone] = node.Labels[v1.LabelTopologyZone]
	}

	// skip calling event handlers when no change in topology labels
	if reflect.DeepEqual(n.topologyLabels, topologyLabels) {
		return
	}

	n.topologyLabels = topologyLabels
	for i := range n.eventHandlers {
		n.logger.V(4).Info("Calling handler.OnTopologyChange")
		n.eventHandlers[i].OnTopologyChange(n.topologyLabels)
	}
}
