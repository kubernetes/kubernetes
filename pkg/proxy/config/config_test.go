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

package config_test

import (
	"reflect"
	"sort"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	. "k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/util/wait"
)

const TomcatPort int = 8080
const TomcatName = "tomcat"

var TomcatEndpoints = map[string]string{"c0": "1.1.1.1:18080", "c1": "2.2.2.2:18081"}

const MysqlPort int = 3306
const MysqlName = "mysql"

var MysqlEndpoints = map[string]string{"c0": "1.1.1.1:13306", "c3": "2.2.2.2:13306"}

type sortedServices []api.Service

func (s sortedServices) Len() int {
	return len(s)
}
func (s sortedServices) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
func (s sortedServices) Less(i, j int) bool {
	return s[i].Name < s[j].Name
}

type ServiceHandlerMock struct {
	updated chan []api.Service
	waits   int
}

func NewServiceHandlerMock() *ServiceHandlerMock {
	return &ServiceHandlerMock{updated: make(chan []api.Service, 5)}
}

func (h *ServiceHandlerMock) OnServiceUpdate(services []api.Service) {
	sort.Sort(sortedServices(services))
	h.updated <- services
}

func (h *ServiceHandlerMock) ValidateServices(t *testing.T, expectedServices []api.Service) {
	// We might get 1 or more updates for N service updates, because we
	// over write older snapshots of services from the producer go-routine
	// if the consumer falls behind.
	var services []api.Service
	for {
		select {
		case services = <-h.updated:
			if reflect.DeepEqual(services, expectedServices) {
				return
			}
		// Unittests will hard timeout in 5m with a stack trace, prevent that
		// and surface a clearer reason for failure.
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Timed out. Expected %#v, Got %#v", expectedServices, services)
			return
		}
	}
}

type sortedEndpoints []api.Endpoints

func (s sortedEndpoints) Len() int {
	return len(s)
}
func (s sortedEndpoints) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
func (s sortedEndpoints) Less(i, j int) bool {
	return s[i].Name < s[j].Name
}

type EndpointsHandlerMock struct {
	updated chan []api.Endpoints
	waits   int
}

func NewEndpointsHandlerMock() *EndpointsHandlerMock {
	return &EndpointsHandlerMock{updated: make(chan []api.Endpoints, 5)}
}

func (h *EndpointsHandlerMock) OnEndpointsUpdate(endpoints []api.Endpoints) {
	sort.Sort(sortedEndpoints(endpoints))
	h.updated <- endpoints
}

func (h *EndpointsHandlerMock) ValidateEndpoints(t *testing.T, expectedEndpoints []api.Endpoints) {
	// We might get 1 or more updates for N endpoint updates, because we
	// over write older snapshots of endpoints from the producer go-routine
	// if the consumer falls behind. Unittests will hard timeout in 5m.
	var endpoints []api.Endpoints
	for {
		select {
		case endpoints = <-h.updated:
			if reflect.DeepEqual(endpoints, expectedEndpoints) {
				return
			}
		// Unittests will hard timeout in 5m with a stack trace, prevent that
		// and surface a clearer reason for failure.
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Timed out. Expected %#v, Got %#v", expectedEndpoints, endpoints)
			return
		}
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

func TestNewServiceAddedAndNotified(t *testing.T) {
	config := NewServiceConfig()
	channel := config.Channel("one")
	handler := NewServiceHandlerMock()
	config.RegisterHandler(handler)
	serviceUpdate := CreateServiceUpdate(ADD, api.Service{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 10}}},
	})
	channel <- serviceUpdate
	handler.ValidateServices(t, serviceUpdate.Services)

}

func TestServiceAddedRemovedSetAndNotified(t *testing.T) {
	config := NewServiceConfig()
	channel := config.Channel("one")
	handler := NewServiceHandlerMock()
	config.RegisterHandler(handler)
	serviceUpdate := CreateServiceUpdate(ADD, api.Service{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 10}}},
	})
	channel <- serviceUpdate
	handler.ValidateServices(t, serviceUpdate.Services)

	serviceUpdate2 := CreateServiceUpdate(ADD, api.Service{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 20}}},
	})
	channel <- serviceUpdate2
	services := []api.Service{serviceUpdate2.Services[0], serviceUpdate.Services[0]}
	handler.ValidateServices(t, services)

	serviceUpdate3 := CreateServiceUpdate(REMOVE, api.Service{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
	})
	channel <- serviceUpdate3
	services = []api.Service{serviceUpdate2.Services[0]}
	handler.ValidateServices(t, services)

	serviceUpdate4 := CreateServiceUpdate(SET, api.Service{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "foobar"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 99}}},
	})
	channel <- serviceUpdate4
	services = []api.Service{serviceUpdate4.Services[0]}
	handler.ValidateServices(t, services)
}

func TestNewMultipleSourcesServicesAddedAndNotified(t *testing.T) {
	config := NewServiceConfig()
	channelOne := config.Channel("one")
	channelTwo := config.Channel("two")
	if channelOne == channelTwo {
		t.Error("Same channel handed back for one and two")
	}
	handler := NewServiceHandlerMock()
	config.RegisterHandler(handler)
	serviceUpdate1 := CreateServiceUpdate(ADD, api.Service{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 10}}},
	})
	serviceUpdate2 := CreateServiceUpdate(ADD, api.Service{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 20}}},
	})
	channelOne <- serviceUpdate1
	channelTwo <- serviceUpdate2
	services := []api.Service{serviceUpdate2.Services[0], serviceUpdate1.Services[0]}
	handler.ValidateServices(t, services)
}

func TestNewMultipleSourcesServicesMultipleHandlersAddedAndNotified(t *testing.T) {
	config := NewServiceConfig()
	channelOne := config.Channel("one")
	channelTwo := config.Channel("two")
	handler := NewServiceHandlerMock()
	handler2 := NewServiceHandlerMock()
	config.RegisterHandler(handler)
	config.RegisterHandler(handler2)
	serviceUpdate1 := CreateServiceUpdate(ADD, api.Service{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 10}}},
	})
	serviceUpdate2 := CreateServiceUpdate(ADD, api.Service{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 20}}},
	})
	channelOne <- serviceUpdate1
	channelTwo <- serviceUpdate2
	services := []api.Service{serviceUpdate2.Services[0], serviceUpdate1.Services[0]}
	handler.ValidateServices(t, services)
	handler2.ValidateServices(t, services)
}

func TestNewMultipleSourcesEndpointsMultipleHandlersAddedAndNotified(t *testing.T) {
	config := NewEndpointsConfig()
	channelOne := config.Channel("one")
	channelTwo := config.Channel("two")
	handler := NewEndpointsHandlerMock()
	handler2 := NewEndpointsHandlerMock()
	config.RegisterHandler(handler)
	config.RegisterHandler(handler2)
	endpointsUpdate1 := CreateEndpointsUpdate(ADD, api.Endpoints{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "1.1.1.1"}, {IP: "2.2.2.2"}},
			Ports:     []api.EndpointPort{{Port: 80}},
		}},
	})
	endpointsUpdate2 := CreateEndpointsUpdate(ADD, api.Endpoints{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "3.3.3.3"}, {IP: "4.4.4.4"}},
			Ports:     []api.EndpointPort{{Port: 80}},
		}},
	})
	channelOne <- endpointsUpdate1
	channelTwo <- endpointsUpdate2

	endpoints := []api.Endpoints{endpointsUpdate2.Endpoints[0], endpointsUpdate1.Endpoints[0]}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)
}

func TestNewMultipleSourcesEndpointsMultipleHandlersAddRemoveSetAndNotified(t *testing.T) {
	config := NewEndpointsConfig()
	channelOne := config.Channel("one")
	channelTwo := config.Channel("two")
	handler := NewEndpointsHandlerMock()
	handler2 := NewEndpointsHandlerMock()
	config.RegisterHandler(handler)
	config.RegisterHandler(handler2)
	endpointsUpdate1 := CreateEndpointsUpdate(ADD, api.Endpoints{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "1.1.1.1"}, {IP: "2.2.2.2"}},
			Ports:     []api.EndpointPort{{Port: 80}},
		}},
	})
	endpointsUpdate2 := CreateEndpointsUpdate(ADD, api.Endpoints{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "3.3.3.3"}, {IP: "4.4.4.4"}},
			Ports:     []api.EndpointPort{{Port: 80}},
		}},
	})
	channelOne <- endpointsUpdate1
	channelTwo <- endpointsUpdate2

	endpoints := []api.Endpoints{endpointsUpdate2.Endpoints[0], endpointsUpdate1.Endpoints[0]}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)

	// Add one more
	endpointsUpdate3 := CreateEndpointsUpdate(ADD, api.Endpoints{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "foobar"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "5.5.5.5"}, {IP: "6.6.6.6"}},
			Ports:     []api.EndpointPort{{Port: 80}},
		}},
	})
	channelTwo <- endpointsUpdate3
	endpoints = []api.Endpoints{endpointsUpdate2.Endpoints[0], endpointsUpdate1.Endpoints[0], endpointsUpdate3.Endpoints[0]}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)

	// Update the "foo" service with new endpoints
	endpointsUpdate1 = CreateEndpointsUpdate(ADD, api.Endpoints{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "7.7.7.7"}},
			Ports:     []api.EndpointPort{{Port: 80}},
		}},
	})
	channelOne <- endpointsUpdate1
	endpoints = []api.Endpoints{endpointsUpdate2.Endpoints[0], endpointsUpdate1.Endpoints[0], endpointsUpdate3.Endpoints[0]}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)

	// Remove "bar" service
	endpointsUpdate2 = CreateEndpointsUpdate(REMOVE, api.Endpoints{ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "bar"}})
	channelTwo <- endpointsUpdate2

	endpoints = []api.Endpoints{endpointsUpdate1.Endpoints[0], endpointsUpdate3.Endpoints[0]}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)
}

// TODO: Add a unittest for interrupts getting processed in a timely manner.
// Currently this module has a circular dependency with config, and so it's
// named config_test, which means even test methods need to be public. This
// is refactoring that we can avoid by resolving the dependency.
