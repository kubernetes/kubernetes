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
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api"
	coreinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion/core/internalversion"
	listers "k8s.io/kubernetes/pkg/client/listers/core/internalversion"
	"k8s.io/kubernetes/pkg/util/config"
)

// ServiceConfigHandler is an abstract interface of objects which receive update notifications for the set of services.
type ServiceConfigHandler interface {
	// OnServiceUpdate gets called when a service is created, removed or changed
	// on any of the configuration sources. An example is when a new service
	// comes up.
	//
	// NOTE: For efficiency, services are being passed by reference, thus,
	// OnServiceUpdate should NOT modify pointers of a given slice.
	// Those service objects are shared with other layers of the system and
	// are guaranteed to be immutable with the assumption that are also
	// not mutated by those handlers. Make a deep copy if you need to modify
	// them in your code.
	OnServiceUpdate(services []*api.Service)
}

// EndpointsConfigHandler is an abstract interface of objects which receive update notifications for the set of endpoints.
type EndpointsConfigHandler interface {
	// OnEndpointsUpdate gets called when endpoints configuration is changed for a given
	// service on any of the configuration sources. An example is when a new
	// service comes up, or when containers come up or down for an existing service.
	//
	// NOTE: For efficiency, endpoints are being passed by reference, thus,
	// OnEndpointsUpdate should NOT modify pointers of a given slice.
	// Those endpoints objects are shared with other layers of the system and
	// are guaranteed to be immutable with the assumption that are also
	// not mutated by those handlers. Make a deep copy if you need to modify
	// them in your code.
	OnEndpointsUpdate(endpoints []*api.Endpoints)
}

// EndpointsHandler is an abstract interface o objects which receive
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
	// TODO: Remove handlers by switching them to eventHandlers.
	handlers []EndpointsConfigHandler
	// updates channel is used to trigger registered handlers.
	updates chan struct{}
	stop    chan struct{}
}

// NewEndpointsConfig creates a new EndpointsConfig.
func NewEndpointsConfig(endpointsInformer coreinformers.EndpointsInformer, resyncPeriod time.Duration) *EndpointsConfig {
	result := &EndpointsConfig{
		lister:       endpointsInformer.Lister(),
		listerSynced: endpointsInformer.Informer().HasSynced,
		// The updates channel is used to send interrupts to the Endpoints handler.
		// It's buffered because we never want to block for as long as there is a
		// pending interrupt, but don't want to drop them if the handler is doing
		// work.
		updates: make(chan struct{}, 1),
		stop:    make(chan struct{}),
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

// RegisterHandler registers a handler which is called on every endpoints change.
func (c *EndpointsConfig) RegisterHandler(handler EndpointsConfigHandler) {
	c.handlers = append(c.handlers, handler)
}

// RegisterEventHandler registers a handler which is called on every endpoints change.
func (c *EndpointsConfig) RegisterEventHandler(handler EndpointsHandler) {
	c.eventHandlers = append(c.eventHandlers, handler)
}

// Run starts the goroutine responsible for calling registered handlers.
func (c *EndpointsConfig) Run(stopCh <-chan struct{}) {
	if !cache.WaitForCacheSync(stopCh, c.listerSynced) {
		utilruntime.HandleError(fmt.Errorf("endpoint controller not synced"))
		return
	}

	// We have synced informers. Now we can start delivering updates
	// to the registered handler.
	go func() {
		for i := range c.eventHandlers {
			glog.V(3).Infof("Calling handler.OnEndpointsSynced()")
			c.eventHandlers[i].OnEndpointsSynced()
		}
		for {
			select {
			case <-c.updates:
				endpoints, err := c.lister.List(labels.Everything())
				if err != nil {
					glog.Errorf("Error while listing endpoints from cache: %v", err)
					// This will cause a retry (if there isn't any other trigger in-flight).
					c.dispatchUpdate()
					continue
				}
				if endpoints == nil {
					endpoints = []*api.Endpoints{}
				}
				for i := range c.handlers {
					glog.V(3).Infof("Calling handler.OnEndpointsUpdate()")
					c.handlers[i].OnEndpointsUpdate(endpoints)
				}
			case <-c.stop:
				return
			}
		}
	}()
	// Close updates channel when stopCh is closed.
	go func() {
		<-stopCh
		close(c.stop)
	}()
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
	c.dispatchUpdate()
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
	c.dispatchUpdate()
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
		glog.V(4).Infof("Calling handler.OnEndpointsUpdate")
		c.eventHandlers[i].OnEndpointsDelete(endpoints)
	}
	c.dispatchUpdate()
}

func (c *EndpointsConfig) dispatchUpdate() {
	select {
	case c.updates <- struct{}{}:
		// Work enqueued successfully
	case <-c.stop:
		// We're shut down / avoid logging the message below
	default:
		glog.V(4).Infof("Endpoints handler already has a pending interrupt.")
	}
}

// ServiceConfig tracks a set of service configurations.
// It accepts "set", "add" and "remove" operations of services via channels, and invokes registered handlers on change.
type ServiceConfig struct {
	lister       listers.ServiceLister
	listerSynced cache.InformerSynced
	handlers     []ServiceConfigHandler
	// updates channel is used to trigger registered handlers
	updates chan struct{}
	stop    chan struct{}
}

// NewServiceConfig creates a new ServiceConfig.
func NewServiceConfig(serviceInformer coreinformers.ServiceInformer, resyncPeriod time.Duration) *ServiceConfig {
	result := &ServiceConfig{
		lister:       serviceInformer.Lister(),
		listerSynced: serviceInformer.Informer().HasSynced,
		// The updates channel is used to send interrupts to the Services handler.
		// It's buffered because we never want to block for as long as there is a
		// pending interrupt, but don't want to drop them if the handler is doing
		// work.
		updates: make(chan struct{}, 1),
		stop:    make(chan struct{}),
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

// RegisterHandler registers a handler which is called on every services change.
func (c *ServiceConfig) RegisterHandler(handler ServiceConfigHandler) {
	c.handlers = append(c.handlers, handler)
}

// Run starts the goroutine responsible for calling
// registered handlers.
func (c *ServiceConfig) Run(stopCh <-chan struct{}) {
	if !cache.WaitForCacheSync(stopCh, c.listerSynced) {
		utilruntime.HandleError(fmt.Errorf("service controller not synced"))
		return
	}

	// We have synced informers. Now we can start delivering updates
	// to the registered handler.
	go func() {
		for {
			select {
			case <-c.updates:
				services, err := c.lister.List(labels.Everything())
				if err != nil {
					glog.Errorf("Error while listing services from cache: %v", err)
					// This will cause a retry (if there isnt' any other trigger in-flight).
					c.dispatchUpdate()
					continue
				}
				if services == nil {
					services = []*api.Service{}
				}
				for i := range c.handlers {
					glog.V(3).Infof("Calling handler.OnServiceUpdate()")
					c.handlers[i].OnServiceUpdate(services)
				}
			case <-c.stop:
				return
			}
		}
	}()
	// Close updates channel when stopCh is closed.
	go func() {
		<-stopCh
		close(c.stop)
	}()
}

func (c *ServiceConfig) handleAddService(_ interface{}) {
	c.dispatchUpdate()
}

func (c *ServiceConfig) handleUpdateService(_, _ interface{}) {
	c.dispatchUpdate()
}

func (c *ServiceConfig) handleDeleteService(_ interface{}) {
	c.dispatchUpdate()
}

func (c *ServiceConfig) dispatchUpdate() {
	select {
	case c.updates <- struct{}{}:
		// Work enqueued successfully
	case <-c.stop:
		// We're shut down / avoid logging the message below
	default:
		glog.V(4).Infof("Service handler already has a pending interrupt.")
	}
}

// watchForUpdates invokes bcaster.Notify() with the latest version of an object
// when changes occur.
func watchForUpdates(bcaster *config.Broadcaster, accessor config.Accessor, updates <-chan struct{}) {
	for true {
		<-updates
		bcaster.Notify(accessor.MergedState())
	}
}
