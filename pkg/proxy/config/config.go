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
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api"
	listers "k8s.io/kubernetes/pkg/client/listers/core/internalversion"
	"k8s.io/kubernetes/pkg/util/config"
)

// ServiceConfigHandler is an abstract interface of objects which receive update notifications for the set of services.
type ServiceConfigHandler interface {
	// OnServiceUpdate gets called when a configuration has been changed by one of the sources.
	// This is the union of all the configuration sources.
	OnServiceUpdate(services []api.Service)
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

// EndpointsConfig tracks a set of endpoints configurations.
// It accepts "set", "add" and "remove" operations of endpoints via channels, and invokes registered handlers on change.
type EndpointsConfig struct {
	informer cache.Controller
	lister   listers.EndpointsLister
	handlers []EndpointsConfigHandler
	// updates channel is used to trigger registered handlers.
	updates chan struct{}
}

// NewEndpointsConfig creates a new EndpointsConfig.
func NewEndpointsConfig(c cache.Getter, period time.Duration) *EndpointsConfig {
	endpointsLW := cache.NewListWatchFromClient(c, "endpoints", metav1.NamespaceAll, fields.Everything())
	return newEndpointsConfig(endpointsLW, period)
}

func newEndpointsConfig(lw cache.ListerWatcher, period time.Duration) *EndpointsConfig {
	result := &EndpointsConfig{}

	store, informer := cache.NewIndexerInformer(
		lw,
		&api.Endpoints{},
		period,
		cache.ResourceEventHandlerFuncs{
			AddFunc:    result.handleAddEndpoints,
			UpdateFunc: result.handleUpdateEndpoints,
			DeleteFunc: result.handleDeleteEndpoints,
		},
		cache.Indexers{},
	)
	result.informer = informer
	result.lister = listers.NewEndpointsLister(store)
	return result
}

// RegisterHandler registers a handler which is called on every endpoints change.
func (c *EndpointsConfig) RegisterHandler(handler EndpointsConfigHandler) {
	c.handlers = append(c.handlers, handler)
}

// Run starts the underlying informer and goroutine responsible for calling
// registered handlers.
func (c *EndpointsConfig) Run(stopCh <-chan struct{}) {
	// The updates channel is used to send interrupts to the Endpoints handler.
	// It's buffered because we never want to block for as long as there is a
	// pending interrupt, but don't want to drop them if the handler is doing
	// work.
	c.updates = make(chan struct{}, 1)
	go c.informer.Run(stopCh)
	if !cache.WaitForCacheSync(stopCh, c.informer.HasSynced) {
		utilruntime.HandleError(fmt.Errorf("endpoint controller not synced"))
		return
	}

	// We have synced informers. Now we can start delivering updates
	// to the registered handler.
	go func() {
		for range c.updates {
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
		}
	}()
	// Close updates channel when stopCh is closed.
	go func() {
		<-stopCh
		close(c.updates)
	}()
}

func (c *EndpointsConfig) handleAddEndpoints(_ interface{}) {
	c.dispatchUpdate()
}

func (c *EndpointsConfig) handleUpdateEndpoints(_, _ interface{}) {
	c.dispatchUpdate()
}

func (c *EndpointsConfig) handleDeleteEndpoints(_ interface{}) {
	c.dispatchUpdate()
}

func (c *EndpointsConfig) dispatchUpdate() {
	select {
	case c.updates <- struct{}{}:
	default:
		glog.V(4).Infof("Endpoints handler already has a pending interrupt.")
	}
}

// ServiceConfig tracks a set of service configurations.
// It accepts "set", "add" and "remove" operations of services via channels, and invokes registered handlers on change.
type ServiceConfig struct {
	informer cache.Controller
	lister   listers.ServiceLister
	handlers []ServiceConfigHandler
	// updates channel is used to trigger registered handlers
	updates chan struct{}
}

// NewServiceConfig creates a new ServiceConfig.
func NewServiceConfig(c cache.Getter, period time.Duration) *ServiceConfig {
	servicesLW := cache.NewListWatchFromClient(c, "services", metav1.NamespaceAll, fields.Everything())
	return newServiceConfig(servicesLW, period)
}

func newServiceConfig(lw cache.ListerWatcher, period time.Duration) *ServiceConfig {
	result := &ServiceConfig{}

	store, informer := cache.NewIndexerInformer(
		lw,
		&api.Service{},
		period,
		cache.ResourceEventHandlerFuncs{
			AddFunc:    result.handleAddService,
			UpdateFunc: result.handleUpdateService,
			DeleteFunc: result.handleDeleteService,
		},
		cache.Indexers{},
	)
	result.informer = informer
	result.lister = listers.NewServiceLister(store)
	return result
}

// RegisterHandler registers a handler which is called on every services change.
func (c *ServiceConfig) RegisterHandler(handler ServiceConfigHandler) {
	c.handlers = append(c.handlers, handler)
}

// Run starts the underlying informer and goroutine responsible for calling
// registered handlers.
func (c *ServiceConfig) Run(stopCh <-chan struct{}) {
	// The updates channel is used to send interrupts to the Services handler.
	// It's buffered because we never want to block for as long as there is a
	// pending interrupt, but don't want to drop them if the handler is doing
	// work.
	c.updates = make(chan struct{}, 1)
	go c.informer.Run(stopCh)
	if !cache.WaitForCacheSync(stopCh, c.informer.HasSynced) {
		utilruntime.HandleError(fmt.Errorf("service controller not synced"))
		return
	}

	// We hanve synced informers. Now we can start delivering updates
	// to the registered handler.
	go func() {
		for range c.updates {
			services, err := c.lister.List(labels.Everything())
			if err != nil {
				glog.Errorf("Error while listing services from cache: %v", err)
				// This will cause a retry (if there isnt' any other trigger in-flight).
				c.dispatchUpdate()
				continue
			}
			svcs := make([]api.Service, 0, len(services))
			for i := range services {
				svcs = append(svcs, *services[i])
			}
			for i := range c.handlers {
				glog.V(3).Infof("Calling handler.OnServiceUpdate()")
				c.handlers[i].OnServiceUpdate(svcs)
			}
		}
	}()
	// Close updates channel when stopCh is closed.
	go func() {
		<-stopCh
		close(c.updates)
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
	default:
		glog.V(4).Infof("Service handler alread has a pending interrupt.")
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
