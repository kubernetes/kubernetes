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
	"log"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

type Operation int

const (
	SET Operation = iota
	ADD
	REMOVE
)

// Defines an operation sent on the channel. You can add or remove single services by
// sending an array of size one and Op == ADD|REMOVE. For setting the state of the system
// to a given state for this source configuration, set Services as desired and Op to SET,
// which will reset the system state to that specified in this operation for this source
// channel. To remove all services, set Services to empty array and Op to SET
type ServiceUpdate struct {
	Services []api.Service
	Op       Operation
}

// Defines an operation sent on the channel. You can add or remove single endpoints by
// sending an array of size one and Op == ADD|REMOVE. For setting the state of the system
// to a given state for this source configuration, set Endpoints as desired and Op to SET,
// which will reset the system state to that specified in this operation for this source
// channel. To remove all endpoints, set Endpoints to empty array and Op to SET
type EndpointsUpdate struct {
	Endpoints []api.Endpoints
	Op        Operation
}

type ServiceConfigHandler interface {
	// Sent when a configuration has been changed by one of the sources. This is the
	// union of all the configuration sources.
	OnUpdate(services []api.Service)
}

type EndpointsConfigHandler interface {
	// OnUpdate gets called when endpoints configuration is changed for a given
	// service on any of the configuration sources. An example is when a new
	// service comes up, or when containers come up or down for an existing service.
	OnUpdate(endpoints []api.Endpoints)
}

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

func NewServiceConfig() ServiceConfig {
	config := ServiceConfig{
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

func (impl *ServiceConfig) Run() {
	log.Printf("Starting the config Run loop")
	for {
		select {
		case source := <-impl.serviceNotifyChannel:
			log.Printf("Got new service configuration from source %s", source)
			impl.NotifyServiceUpdate()
		case source := <-impl.endpointsNotifyChannel:
			log.Printf("Got new endpoint configuration from source %s", source)
			impl.NotifyEndpointsUpdate()
		case <-time.After(1 * time.Second):
		}
	}
}

func (impl *ServiceConfig) ServiceChannelListener(source string, listenChannel chan ServiceUpdate) {
	// Represents the current services configuration for this channel.
	serviceMap := make(map[string]api.Service)
	for {
		select {
		case update := <-listenChannel:
			switch update.Op {
			case ADD:
				log.Printf("Adding new service from source %s : %v", source, update.Services)
				for _, value := range update.Services {
					serviceMap[value.ID] = value
				}
			case REMOVE:
				log.Printf("Removing a service %v", update)
				for _, value := range update.Services {
					delete(serviceMap, value.ID)
				}
			case SET:
				log.Printf("Setting services %v", update)
				// Clear the old map entries by just creating a new map
				serviceMap = make(map[string]api.Service)
				for _, value := range update.Services {
					serviceMap[value.ID] = value
				}
			default:
				log.Printf("Received invalid update type: %v", update)
				continue
			}
			impl.configLock.Lock()
			impl.serviceConfig[source] = serviceMap
			impl.configLock.Unlock()
			impl.serviceNotifyChannel <- source
		}
	}
}

func (impl *ServiceConfig) EndpointsChannelListener(source string, listenChannel chan EndpointsUpdate) {
	endpointMap := make(map[string]api.Endpoints)
	for {
		select {
		case update := <-listenChannel:
			switch update.Op {
			case ADD:
				log.Printf("Adding a new endpoint %v", update)
				for _, value := range update.Endpoints {
					endpointMap[value.Name] = value
				}
			case REMOVE:
				log.Printf("Removing an endpoint %v", update)
				for _, value := range update.Endpoints {
					delete(endpointMap, value.Name)
				}

			case SET:
				log.Printf("Setting services %v", update)
				// Clear the old map entries by just creating a new map
				endpointMap = make(map[string]api.Endpoints)
				for _, value := range update.Endpoints {
					endpointMap[value.Name] = value
				}
			default:
				log.Printf("Received invalid update type: %v", update)
				continue
			}
			impl.configLock.Lock()
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
	go impl.ServiceChannelListener(source, newChannel)
	return newChannel
}

// GetEndpointConfigurationChannel returns a channel where a configuration source
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
	go impl.EndpointsChannelListener(source, newChannel)
	return newChannel
}

// Register ServiceConfigHandler to receive updates of changes to services.
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

// Register ServiceConfigHandler to receive updates of changes to services.
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

func (impl *ServiceConfig) NotifyServiceUpdate() {
	services := make([]api.Service, 0)
	impl.configLock.RLock()
	for _, sourceServices := range impl.serviceConfig {
		for _, value := range sourceServices {
			services = append(services, value)
		}
	}
	impl.configLock.RUnlock()
	log.Printf("Unified configuration %+v", services)
	impl.handlerLock.RLock()
	handlers := impl.serviceHandlers
	impl.handlerLock.RUnlock()
	for _, handler := range handlers {
		if handler != nil {
			handler.OnUpdate(services)
		}
	}
}

func (impl *ServiceConfig) NotifyEndpointsUpdate() {
	endpoints := make([]api.Endpoints, 0)
	impl.configLock.RLock()
	for _, sourceEndpoints := range impl.endpointConfig {
		for _, value := range sourceEndpoints {
			endpoints = append(endpoints, value)
		}
	}
	impl.configLock.RUnlock()
	log.Printf("Unified configuration %+v", endpoints)
	impl.handlerLock.RLock()
	handlers := impl.endpointHandlers
	impl.handlerLock.RUnlock()
	for _, handler := range handlers {
		if handler != nil {
			handler.OnUpdate(endpoints)
		}
	}
}
