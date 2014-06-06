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
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

const TomcatPort int = 8080
const TomcatName = "tomcat"

var TomcatEndpoints = map[string]string{"c0": "1.1.1.1:18080", "c1": "2.2.2.2:18081"}

const MysqlPort int = 3306
const MysqlName = "mysql"

var MysqlEndpoints = map[string]string{"c0": "1.1.1.1:13306", "c3": "2.2.2.2:13306"}

type ServiceHandlerMock struct {
	services []api.Service
}

func NewServiceHandlerMock() ServiceHandlerMock {
	return ServiceHandlerMock{services: make([]api.Service, 0)}
}

func (impl ServiceHandlerMock) OnUpdate(services []api.Service) {
	impl.services = services
}

func (impl ServiceHandlerMock) ValidateServices(t *testing.T, expectedServices []api.Service) {
	if reflect.DeepEqual(impl.services, expectedServices) {
		t.Errorf("Services don't match %+v expected: %+v", impl.services, expectedServices)
	}
}

type EndpointsHandlerMock struct {
	endpoints []api.Endpoints
}

func NewEndpointsHandlerMock() EndpointsHandlerMock {
	return EndpointsHandlerMock{endpoints: make([]api.Endpoints, 0)}
}

func (impl EndpointsHandlerMock) OnUpdate(endpoints []api.Endpoints) {
	impl.endpoints = endpoints
}

func (impl EndpointsHandlerMock) ValidateEndpoints(t *testing.T, expectedEndpoints []api.Endpoints) {
	if reflect.DeepEqual(impl.endpoints, expectedEndpoints) {
		t.Errorf("Endpoints don't match %+v", impl.endpoints, expectedEndpoints)
	}
}

func CreateServiceUpdate(op Operation, services ...api.Service) ServiceUpdate {
	ret := ServiceUpdate{Op: op}
	ret.Services = make([]api.Service, len(services))
	for i, value := range services {
		ret.Services[i] = value
	}
	return ret
}

func CreateEndpointsUpdate(op Operation, endpoints ...api.Endpoints) EndpointsUpdate {
	ret := EndpointsUpdate{Op: op}
	ret.Endpoints = make([]api.Endpoints, len(endpoints))
	for i, value := range endpoints {
		ret.Endpoints[i] = value
	}
	return ret
}

func TestServiceConfigurationChannels(t *testing.T) {
	config := NewServiceConfig()
	channelOne := config.GetServiceConfigurationChannel("one")
	if channelOne != config.GetServiceConfigurationChannel("one") {
		t.Error("Didn't get the same service configuration channel back with the same name")
	}
	channelTwo := config.GetServiceConfigurationChannel("two")
	if channelOne == channelTwo {
		t.Error("Got back the same service configuration channel for different names")
	}
}

func TestEndpointConfigurationChannels(t *testing.T) {
	config := NewServiceConfig()
	channelOne := config.GetEndpointsConfigurationChannel("one")
	if channelOne != config.GetEndpointsConfigurationChannel("one") {
		t.Error("Didn't get the same endpoint configuration channel back with the same name")
	}
	channelTwo := config.GetEndpointsConfigurationChannel("two")
	if channelOne == channelTwo {
		t.Error("Got back the same endpoint configuration channel for different names")
	}
}

func TestNewServiceAddedAndNotified(t *testing.T) {
	config := NewServiceConfig()
	channel := config.GetServiceConfigurationChannel("one")
	handler := NewServiceHandlerMock()
	config.RegisterServiceHandler(&handler)
	serviceUpdate := CreateServiceUpdate(ADD, api.Service{JSONBase: api.JSONBase{ID: "foo"}, Port: 10})
	channel <- serviceUpdate
	handler.ValidateServices(t, serviceUpdate.Services)

}

func TestServiceAddedRemovedSetAndNotified(t *testing.T) {
	config := NewServiceConfig()
	channel := config.GetServiceConfigurationChannel("one")
	handler := NewServiceHandlerMock()
	config.RegisterServiceHandler(&handler)
	serviceUpdate := CreateServiceUpdate(ADD, api.Service{JSONBase: api.JSONBase{ID: "foo"}, Port: 10})
	channel <- serviceUpdate
	handler.ValidateServices(t, serviceUpdate.Services)

	serviceUpdate2 := CreateServiceUpdate(ADD, api.Service{JSONBase: api.JSONBase{ID: "bar"}, Port: 20})
	channel <- serviceUpdate2
	services := []api.Service{serviceUpdate.Services[0], serviceUpdate2.Services[0]}
	handler.ValidateServices(t, services)

	serviceUpdate3 := CreateServiceUpdate(REMOVE, api.Service{JSONBase: api.JSONBase{ID: "foo"}})
	channel <- serviceUpdate3
	services = []api.Service{serviceUpdate2.Services[0]}
	handler.ValidateServices(t, services)

	serviceUpdate4 := CreateServiceUpdate(SET, api.Service{JSONBase: api.JSONBase{ID: "foobar"}, Port: 99})
	channel <- serviceUpdate4
	services = []api.Service{serviceUpdate4.Services[0]}
	handler.ValidateServices(t, services)
}

func TestNewMultipleSourcesServicesAddedAndNotified(t *testing.T) {
	config := NewServiceConfig()
	channelOne := config.GetServiceConfigurationChannel("one")
	channelTwo := config.GetServiceConfigurationChannel("two")
	if channelOne == channelTwo {
		t.Error("Same channel handed back for one and two")
	}
	handler := NewServiceHandlerMock()
	config.RegisterServiceHandler(handler)
	serviceUpdate1 := CreateServiceUpdate(ADD, api.Service{JSONBase: api.JSONBase{ID: "foo"}, Port: 10})
	serviceUpdate2 := CreateServiceUpdate(ADD, api.Service{JSONBase: api.JSONBase{ID: "bar"}, Port: 20})
	channelOne <- serviceUpdate1
	channelTwo <- serviceUpdate2
	services := []api.Service{serviceUpdate1.Services[0], serviceUpdate2.Services[0]}
	handler.ValidateServices(t, services)
}

func TestNewMultipleSourcesServicesMultipleHandlersAddedAndNotified(t *testing.T) {
	config := NewServiceConfig()
	channelOne := config.GetServiceConfigurationChannel("one")
	channelTwo := config.GetServiceConfigurationChannel("two")
	handler := NewServiceHandlerMock()
	handler2 := NewServiceHandlerMock()
	config.RegisterServiceHandler(handler)
	config.RegisterServiceHandler(handler2)
	serviceUpdate1 := CreateServiceUpdate(ADD, api.Service{JSONBase: api.JSONBase{ID: "foo"}, Port: 10})
	serviceUpdate2 := CreateServiceUpdate(ADD, api.Service{JSONBase: api.JSONBase{ID: "bar"}, Port: 20})
	channelOne <- serviceUpdate1
	channelTwo <- serviceUpdate2
	services := []api.Service{serviceUpdate1.Services[0], serviceUpdate2.Services[0]}
	handler.ValidateServices(t, services)
	handler2.ValidateServices(t, services)
}

func TestNewMultipleSourcesEndpointsMultipleHandlersAddedAndNotified(t *testing.T) {
	config := NewServiceConfig()
	channelOne := config.GetEndpointsConfigurationChannel("one")
	channelTwo := config.GetEndpointsConfigurationChannel("two")
	handler := NewEndpointsHandlerMock()
	handler2 := NewEndpointsHandlerMock()
	config.RegisterEndpointsHandler(handler)
	config.RegisterEndpointsHandler(handler2)
	endpointsUpdate1 := CreateEndpointsUpdate(ADD, api.Endpoints{Name: "foo", Endpoints: []string{"endpoint1", "endpoint2"}})
	endpointsUpdate2 := CreateEndpointsUpdate(ADD, api.Endpoints{Name: "bar", Endpoints: []string{"endpoint3", "endpoint4"}})
	channelOne <- endpointsUpdate1
	channelTwo <- endpointsUpdate2

	endpoints := []api.Endpoints{endpointsUpdate1.Endpoints[0], endpointsUpdate2.Endpoints[0]}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)
}

func TestNewMultipleSourcesEndpointsMultipleHandlersAddRemoveSetAndNotified(t *testing.T) {
	config := NewServiceConfig()
	channelOne := config.GetEndpointsConfigurationChannel("one")
	channelTwo := config.GetEndpointsConfigurationChannel("two")
	handler := NewEndpointsHandlerMock()
	handler2 := NewEndpointsHandlerMock()
	config.RegisterEndpointsHandler(handler)
	config.RegisterEndpointsHandler(handler2)
	endpointsUpdate1 := CreateEndpointsUpdate(ADD, api.Endpoints{Name: "foo", Endpoints: []string{"endpoint1", "endpoint2"}})
	endpointsUpdate2 := CreateEndpointsUpdate(ADD, api.Endpoints{Name: "bar", Endpoints: []string{"endpoint3", "endpoint4"}})
	channelOne <- endpointsUpdate1
	channelTwo <- endpointsUpdate2

	endpoints := []api.Endpoints{endpointsUpdate1.Endpoints[0], endpointsUpdate2.Endpoints[0]}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)

	// Add one more
	endpointsUpdate3 := CreateEndpointsUpdate(ADD, api.Endpoints{Name: "foobar", Endpoints: []string{"endpoint5", "endpoint6"}})
	channelTwo <- endpointsUpdate3
	endpoints = []api.Endpoints{endpointsUpdate1.Endpoints[0], endpointsUpdate2.Endpoints[0], endpointsUpdate3.Endpoints[0]}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)

	// Update the "foo" service with new endpoints
	endpointsUpdate1 = CreateEndpointsUpdate(ADD, api.Endpoints{Name: "foo", Endpoints: []string{"endpoint77"}})
	channelOne <- endpointsUpdate1
	endpoints = []api.Endpoints{endpointsUpdate1.Endpoints[0], endpointsUpdate2.Endpoints[0], endpointsUpdate3.Endpoints[0]}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)

	// Remove "bar" service
	endpointsUpdate2 = CreateEndpointsUpdate(REMOVE, api.Endpoints{Name: "bar"})
	channelTwo <- endpointsUpdate2

	endpoints = []api.Endpoints{endpointsUpdate1.Endpoints[0], endpointsUpdate3.Endpoints[0]}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)
}
