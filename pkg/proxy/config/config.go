/*
Copyright 2014 Google Inc. All rights reserved.

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
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/golang/glog"
)

// Operation is a type of operation of services or endpoints.
type Operation int

// These are the available operation types.
const (
	SET Operation = iota
	ADD
	REMOVE
)

// ServiceUpdate describes an operation of services, sent on the channel.
// You can add or remove single services by sending an array of size one and Op == ADD|REMOVE.
// For setting the state of the system to a given state for this source configuration, set Services as desired and Op to SET,
// which will reset the system state to that specified in this operation for this source channel.
// To remove all services, set Services to empty array and Op to SET
type ServiceUpdate struct {
	Services []api.Service
	Op       Operation
}

// EndpointsUpdate describes an operation of endpoints, sent on the channel.
// You can add or remove single endpoints by sending an array of size one and Op == ADD|REMOVE.
// For setting the state of the system to a given state for this source configuration, set Endpoints as desired and Op to SET,
// which will reset the system state to that specified in this operation for this source channel.
// To remove all endpoints, set Endpoints to empty array and Op to SET
type EndpointsUpdate struct {
	Endpoints []api.Endpoints
	Op        Operation
}

// ServiceConfigHandler is an abstract interface of objects which receive update notifications for the set of services.
type ServiceConfigHandler interface {
	// OnUpdate gets called when a configuration has been changed by one of the sources.
	// This is the union of all the configuration sources.
	OnUpdate(services []api.Service)
}

// EndpointsConfigHandler is an abstract interface of objects which receive update notifications for the set of endpoints.
type EndpointsConfigHandler interface {
	// OnUpdate gets called when endpoints configuration is changed for a given
	// service on any of the configuration sources. An example is when a new
	// service comes up, or when containers come up or down for an existing service.
	OnUpdate(endpoints []api.Endpoints)
}

// ServiceConfig tracks a set of service configurations and their endpoint configurations.
// It accepts "set", "add" and "remove" operations of services and endpoints via channels, and invokes registered handlers on change.
type ServiceConfig struct {
	// Configuration sources and their lock.
	configSourceLock       sync.RWMutex
	serviceConfigSources   map[string]chan ServiceUpdate
	endpointsConfigSources map[string]chan EndpointsUpdate

	// Handlers for changes to services and endpoints and their lock.
	handlerLock      sync.RWMutex
	serviceHandlers  []ServiceConfigHandler
	endpointHandlers []EndpointsConfigHandler

	// Last known configuration for union of the sources and the locks. Map goes
	// from each source to array of services/endpoints that have been configured
	// through that channel.
	configLock     sync.RWMutex
	serviceConfig  map[string]map[string]api.Service
	endpointConfig map[string]map[string]api.Endpoints

	// Channel that service configuration source listeners use to signal of new
	// configurations.
	// Value written is the source of the change.
	serviceNotifyChannel chan string

	// Channel that endpoint configuration source listeners use to signal of new
	// configurations.
	// Value written is the source of the change.
	endpointsNotifyChannel chan string
}

// NewServiceConfig creates a new ServiceConfig.
// It immediately runs the created ServiceConfig.
func NewServiceConfig() *ServiceConfig {
	config := &ServiceConfig{
		serviceConfigSources:   make(map[string]chan ServiceUpdate),
		endpointsConfigSources: make(map[string]chan EndpointsUpdate),
		serviceHandlers:        make([]ServiceConfigHandler, 10),
		endpointHandlers:       make([]EndpointsConfigHandler, 10),
		serviceConfig:          make(map[string]map[string]api.Service),
		endpointConfig:         make(map[string]map[string]api.Endpoints),
		serviceNotifyChannel:   make(chan string),
		endpointsNotifyChannel: make(chan string),
	}
	go config.Run()
	return config
}

// Run begins a loop to accept new service configurations and new endpoint configurations.
// It never returns.
func (impl *ServiceConfig) Run() {
	glog.Infof("Starting the config Run loop")
	for {
		select {
		case source := <-impl.serviceNotifyChannel:
			glog.Infof("Got new service configuration from source %s", source)
			impl.notifyServiceUpdate()
		case source := <-impl.endpointsNotifyChannel:
			glog.Infof("Got new endpoint configuration from source %s", source)
			impl.notifyEndpointsUpdate()
		case <-time.After(1 * time.Second):
		}
	}
}

// serviceChannelListener begins a loop to handle incoming ServiceUpdate notifications from the channel.
// It never returns.
func (impl *ServiceConfig) serviceChannelListener(source string, listenChannel chan ServiceUpdate) {
	// Represents the current services configuration for this channel.
	serviceMap := make(map[string]api.Service)
	for {
		select {
		case update := <-listenChannel:
			impl.configLock.Lock()
			switch update.Op {
			case ADD:
				glog.Infof("Adding new service from source %s : %v", source, update.Services)
				for _, value := range update.Services {
					serviceMap[value.ID] = value
				}
			case REMOVE:
				glog.Infof("Removing a service %v", update)
				for _, value := range update.Services {
					delete(serviceMap, value.ID)
				}
			case SET:
				glog.Infof("Setting services %v", update)
				// Clear the old map entries by just creating a new map
				serviceMap = make(map[string]api.Service)
				for _, value := range update.Services {
					serviceMap[value.ID] = value
				}
			default:
				glog.Infof("Received invalid update type: %v", update)
				continue
			}
			impl.serviceConfig[source] = serviceMap
			impl.configLock.Unlock()
			impl.serviceNotifyChannel <- source
		}
	}
}

// endpointsChannelListener begins a loop to handle incoming EndpointsUpdate notifications from the channel.
// It never returns.
func (impl *ServiceConfig) endpointsChannelListener(source string, listenChannel chan EndpointsUpdate) {
	endpointMap := make(map[string]api.Endpoints)
	for {
		select {
		case update := <-listenChannel:
			impl.configLock.Lock()
			switch update.Op {
			case ADD:
				glog.Infof("Adding a new endpoint %v", update)
				for _, value := range update.Endpoints {
					endpointMap[value.Name] = value
				}
			case REMOVE:
				glog.Infof("Removing an endpoint %v", update)
				for _, value := range update.Endpoints {
					delete(endpointMap, value.Name)
				}

			case SET:
				glog.Infof("Setting services %v", update)
				// Clear the old map entries by just creating a new map
				endpointMap = make(map[string]api.Endpoints)
				for _, value := range update.Endpoints {
					endpointMap[value.Name] = value
				}
			default:
				glog.Infof("Received invalid update type: %v", update)
				continue
			}
			impl.endpointConfig[source] = endpointMap
			impl.configLock.Unlock()
			impl.endpointsNotifyChannel <- source
		}

	}
}

// GetServiceConfigurationChannel returns a channel where a configuration source
// can send updates of new service configurations. Multiple calls with the same
// source will return the same channel. This allows change and state based sources
// to use the same channel. Difference source names however will be treated as a
// union.
func (impl *ServiceConfig) GetServiceConfigurationChannel(source string) chan ServiceUpdate {
	if len(source) == 0 {
		panic("GetServiceConfigurationChannel given an empty service name")
	}
	impl.configSourceLock.Lock()
	defer impl.configSourceLock.Unlock()
	channel, exists := impl.serviceConfigSources[source]
	if exists {
		return channel
	}
	newChannel := make(chan ServiceUpdate)
	impl.serviceConfigSources[source] = newChannel
	go impl.serviceChannelListener(source, newChannel)
	return newChannel
}

// GetEndpointsConfigurationChannel returns a channel where a configuration source
// can send updates of new endpoint configurations. Multiple calls with the same
// source will return the same channel. This allows change and state based sources
// to use the same channel. Difference source names however will be treated as a
// union.
func (impl *ServiceConfig) GetEndpointsConfigurationChannel(source string) chan EndpointsUpdate {
	if len(source) == 0 {
		panic("GetEndpointConfigurationChannel given an empty service name")
	}
	impl.configSourceLock.Lock()
	defer impl.configSourceLock.Unlock()
	channel, exists := impl.endpointsConfigSources[source]
	if exists {
		return channel
	}
	newChannel := make(chan EndpointsUpdate)
	impl.endpointsConfigSources[source] = newChannel
	go impl.endpointsChannelListener(source, newChannel)
	return newChannel
}

// RegisterServiceHandler registers the ServiceConfigHandler to receive updates of changes to services.
func (impl *ServiceConfig) RegisterServiceHandler(handler ServiceConfigHandler) {
	impl.handlerLock.Lock()
	defer impl.handlerLock.Unlock()
	for i, h := range impl.serviceHandlers {
		if h == nil {
			impl.serviceHandlers[i] = handler
			return
		}
	}
	// TODO(vaikas): Grow the array here instead of panic.
	// In practice we are expecting there to be 1 handler anyways,
	// so not a big deal for now
	panic("Only up to 10 service handlers supported for now")
}

// RegisterEndpointsHandler registers the EndpointsConfigHandler to receive updates of changes to services.
func (impl *ServiceConfig) RegisterEndpointsHandler(handler EndpointsConfigHandler) {
	impl.handlerLock.Lock()
	defer impl.handlerLock.Unlock()
	for i, h := range impl.endpointHandlers {
		if h == nil {
			impl.endpointHandlers[i] = handler
			return
		}
	}
	// TODO(vaikas): Grow the array here instead of panic.
	// In practice we are expecting there to be 1 handler anyways,
	// so not a big deal for now
	panic("Only up to 10 endpoint handlers supported for now")
}

// notifyServiceUpdate calls the registered ServiceConfigHandlers with the current states of services.
func (impl *ServiceConfig) notifyServiceUpdate() {
	services := []api.Service{}
	impl.configLock.RLock()
	for _, sourceServices := range impl.serviceConfig {
		for _, value := range sourceServices {
			services = append(services, value)
		}
	}
	impl.configLock.RUnlock()
	glog.Infof("Unified configuration %+v", services)
	impl.handlerLock.RLock()
	handlers := impl.serviceHandlers
	impl.handlerLock.RUnlock()
	for _, handler := range handlers {
		if handler != nil {
			handler.OnUpdate(services)
		}
	}
}

// notifyEndpointsUpdate calls the registered EndpointsConfigHandlers with the current states of endpoints.
func (impl *ServiceConfig) notifyEndpointsUpdate() {
	endpoints := []api.Endpoints{}
	impl.configLock.RLock()
	for _, sourceEndpoints := range impl.endpointConfig {
		for _, value := range sourceEndpoints {
			endpoints = append(endpoints, value)
		}
	}
	impl.configLock.RUnlock()
	glog.Infof("Unified configuration %+v", endpoints)
	impl.handlerLock.RLock()
	handlers := impl.endpointHandlers
	impl.handlerLock.RUnlock()
	for _, handler := range handlers {
		if handler != nil {
			handler.OnUpdate(endpoints)
		}
	}
}
