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
	"sync"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/golang/glog"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api"
	listers "k8s.io/kubernetes/pkg/client/listers/core/internalversion"
	"k8s.io/kubernetes/pkg/util/config"
)

// Operation is a type of operation of services or endpoints.
type Operation int

// These are the available operation types.
const (
	ADD Operation = iota
	UPDATE
	REMOVE
	SYNCED
)

// ServiceUpdate describes an operation of services, sent on the channel.
// You can add, update or remove single service by setting Op == ADD|UPDATE|REMOVE.
type ServiceUpdate struct {
	Service *api.Service
	Op      Operation
}

// EndpointsUpdate describes an operation of endpoints, sent on the channel.
// You can add, update or remove single endpoints by setting Op == ADD|UPDATE|REMOVE.
type EndpointsUpdate struct {
	Endpoints *api.Endpoints
	Op        Operation
}

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
	OnEndpointsUpdate(endpoints []api.Endpoints)
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
			// TODO: This will no longer be needed with #42989
			eps := make([]api.Endpoints, 0, len(endpoints))
			for i := range endpoints {
				eps = append(eps, *endpoints[i])
			}
			for i := range c.handlers {
				glog.V(3).Infof("Calling handler.OnEndpointsUpdate()")
				c.handlers[i].OnEndpointsUpdate(eps)
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
	mux     *config.Mux
	bcaster *config.Broadcaster
	store   *serviceStore
}

// NewServiceConfig creates a new ServiceConfig.
// It immediately runs the created ServiceConfig.
func NewServiceConfig() *ServiceConfig {
	// The updates channel is used to send interrupts to the Services handler.
	// It's buffered because we never want to block for as long as there is a
	// pending interrupt, but don't want to drop them if the handler is doing
	// work.
	updates := make(chan struct{}, 1)
	store := &serviceStore{updates: updates, services: make(map[string]map[types.NamespacedName]*api.Service)}
	mux := config.NewMux(store)
	bcaster := config.NewBroadcaster()
	go watchForUpdates(bcaster, store, updates)
	return &ServiceConfig{mux, bcaster, store}
}

// RegisterHandler registers a handler which is called on every services change.
func (c *ServiceConfig) RegisterHandler(handler ServiceConfigHandler) {
	c.bcaster.Add(config.ListenerFunc(func(instance interface{}) {
		glog.V(3).Infof("Calling handler.OnServiceUpdate()")
		handler.OnServiceUpdate(instance.([]api.Service))
	}))
}

// Channel returns a channel to which services updates should be delivered.
func (c *ServiceConfig) Channel(source string) chan ServiceUpdate {
	ch := c.mux.Channel(source)
	serviceCh := make(chan ServiceUpdate)
	go func() {
		for update := range serviceCh {
			ch <- update
		}
	}()
	return serviceCh
}

// Config returns list of all services from underlying store.
func (c *ServiceConfig) Config() []api.Service {
	return c.store.MergedState().([]api.Service)
}

type serviceStore struct {
	serviceLock sync.RWMutex
	services    map[string]map[types.NamespacedName]*api.Service
	synced      bool
	updates     chan<- struct{}
}

func (s *serviceStore) Merge(source string, change interface{}) error {
	s.serviceLock.Lock()
	services := s.services[source]
	if services == nil {
		services = make(map[types.NamespacedName]*api.Service)
	}
	update := change.(ServiceUpdate)
	switch update.Op {
	case ADD, UPDATE:
		glog.V(5).Infof("Adding new service from source %s : %s", source, spew.Sdump(update.Service))
		name := types.NamespacedName{Namespace: update.Service.Namespace, Name: update.Service.Name}
		services[name] = update.Service
	case REMOVE:
		glog.V(5).Infof("Removing a service %s", spew.Sdump(update.Service))
		name := types.NamespacedName{Namespace: update.Service.Namespace, Name: update.Service.Name}
		delete(services, name)
	case SYNCED:
		s.synced = true
	default:
		glog.V(4).Infof("Received invalid update type: %s", spew.Sdump(update))
	}
	s.services[source] = services
	synced := s.synced
	s.serviceLock.Unlock()
	if s.updates != nil && synced {
		select {
		case s.updates <- struct{}{}:
		default:
			glog.V(4).Infof("Service handler already has a pending interrupt.")
		}
	}
	return nil
}

func (s *serviceStore) MergedState() interface{} {
	s.serviceLock.RLock()
	defer s.serviceLock.RUnlock()
	services := make([]api.Service, 0)
	for _, sourceServices := range s.services {
		for _, value := range sourceServices {
			services = append(services, *value)
		}
	}
	return services
}

// watchForUpdates invokes bcaster.Notify() with the latest version of an object
// when changes occur.
func watchForUpdates(bcaster *config.Broadcaster, accessor config.Accessor, updates <-chan struct{}) {
	for true {
		<-updates
		bcaster.Notify(accessor.MergedState())
	}
}
