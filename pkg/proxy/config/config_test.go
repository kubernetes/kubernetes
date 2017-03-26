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
	"reflect"
	"sort"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api"
)

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

type sortedEndpoints []*api.Endpoints

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
	updated chan []*api.Endpoints
	waits   int
}

func NewEndpointsHandlerMock() *EndpointsHandlerMock {
	return &EndpointsHandlerMock{updated: make(chan []*api.Endpoints, 5)}
}

func (h *EndpointsHandlerMock) OnEndpointsUpdate(endpoints []*api.Endpoints) {
	sort.Sort(sortedEndpoints(endpoints))
	h.updated <- endpoints
}

func (h *EndpointsHandlerMock) ValidateEndpoints(t *testing.T, expectedEndpoints []*api.Endpoints) {
	// We might get 1 or more updates for N endpoint updates, because we
	// over write older snapshots of endpoints from the producer go-routine
	// if the consumer falls behind. Unittests will hard timeout in 5m.
	var endpoints []*api.Endpoints
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

func TestNewServiceAddedAndNotified(t *testing.T) {
	fakeWatch := watch.NewFake()
	lw := fakeLW{
		listResp:  &api.ServiceList{Items: []api.Service{}},
		watchResp: fakeWatch,
	}
	stopCh := make(chan struct{})
	defer close(stopCh)

	config := newServiceConfig(lw, time.Minute)
	handler := NewServiceHandlerMock()
	config.RegisterHandler(handler)
	go config.Run(stopCh)

	service := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 10}}},
	}
	fakeWatch.Add(service)
	handler.ValidateServices(t, []api.Service{*service})
}

func TestServiceAddedRemovedSetAndNotified(t *testing.T) {
	fakeWatch := watch.NewFake()
	lw := fakeLW{
		listResp:  &api.ServiceList{Items: []api.Service{}},
		watchResp: fakeWatch,
	}
	stopCh := make(chan struct{})
	defer close(stopCh)

	config := newServiceConfig(lw, time.Minute)
	handler := NewServiceHandlerMock()
	config.RegisterHandler(handler)
	go config.Run(stopCh)

	service1 := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 10}}},
	}
	fakeWatch.Add(service1)
	handler.ValidateServices(t, []api.Service{*service1})

	service2 := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 20}}},
	}
	fakeWatch.Add(service2)
	services := []api.Service{*service2, *service1}
	handler.ValidateServices(t, services)

	fakeWatch.Delete(service1)
	services = []api.Service{*service2}
	handler.ValidateServices(t, services)
}

func TestNewServicesMultipleHandlersAddedAndNotified(t *testing.T) {
	fakeWatch := watch.NewFake()
	lw := fakeLW{
		listResp:  &api.ServiceList{Items: []api.Service{}},
		watchResp: fakeWatch,
	}
	stopCh := make(chan struct{})
	defer close(stopCh)

	config := newServiceConfig(lw, time.Minute)
	handler := NewServiceHandlerMock()
	handler2 := NewServiceHandlerMock()
	config.RegisterHandler(handler)
	config.RegisterHandler(handler2)
	go config.Run(stopCh)

	service1 := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 10}}},
	}
	service2 := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 20}}},
	}
	fakeWatch.Add(service1)
	fakeWatch.Add(service2)

	services := []api.Service{*service2, *service1}
	handler.ValidateServices(t, services)
	handler2.ValidateServices(t, services)
}

func TestNewEndpointsMultipleHandlersAddedAndNotified(t *testing.T) {
	fakeWatch := watch.NewFake()
	lw := fakeLW{
		listResp:  &api.EndpointsList{Items: []api.Endpoints{}},
		watchResp: fakeWatch,
	}
	stopCh := make(chan struct{})
	defer close(stopCh)

	config := newEndpointsConfig(lw, time.Minute)
	handler := NewEndpointsHandlerMock()
	handler2 := NewEndpointsHandlerMock()
	config.RegisterHandler(handler)
	config.RegisterHandler(handler2)
	go config.Run(stopCh)

	endpoints1 := &api.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "1.1.1.1"}, {IP: "2.2.2.2"}},
			Ports:     []api.EndpointPort{{Port: 80}},
		}},
	}
	endpoints2 := &api.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "3.3.3.3"}, {IP: "4.4.4.4"}},
			Ports:     []api.EndpointPort{{Port: 80}},
		}},
	}
	fakeWatch.Add(endpoints1)
	fakeWatch.Add(endpoints2)

	endpoints := []*api.Endpoints{endpoints2, endpoints1}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)
}

func TestNewEndpointsMultipleHandlersAddRemoveSetAndNotified(t *testing.T) {
	fakeWatch := watch.NewFake()
	lw := fakeLW{
		listResp:  &api.EndpointsList{Items: []api.Endpoints{}},
		watchResp: fakeWatch,
	}
	stopCh := make(chan struct{})
	defer close(stopCh)

	config := newEndpointsConfig(lw, time.Minute)
	handler := NewEndpointsHandlerMock()
	handler2 := NewEndpointsHandlerMock()
	config.RegisterHandler(handler)
	config.RegisterHandler(handler2)
	go config.Run(stopCh)

	endpoints1 := &api.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "1.1.1.1"}, {IP: "2.2.2.2"}},
			Ports:     []api.EndpointPort{{Port: 80}},
		}},
	}
	endpoints2 := &api.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "3.3.3.3"}, {IP: "4.4.4.4"}},
			Ports:     []api.EndpointPort{{Port: 80}},
		}},
	}
	fakeWatch.Add(endpoints1)
	fakeWatch.Add(endpoints2)

	endpoints := []*api.Endpoints{endpoints2, endpoints1}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)

	// Add one more
	endpoints3 := &api.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foobar"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "5.5.5.5"}, {IP: "6.6.6.6"}},
			Ports:     []api.EndpointPort{{Port: 80}},
		}},
	}
	fakeWatch.Add(endpoints3)
	endpoints = []*api.Endpoints{endpoints2, endpoints1, endpoints3}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)

	// Update the "foo" service with new endpoints
	endpoints1v2 := &api.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "7.7.7.7"}},
			Ports:     []api.EndpointPort{{Port: 80}},
		}},
	}
	fakeWatch.Modify(endpoints1v2)
	endpoints = []*api.Endpoints{endpoints2, endpoints1v2, endpoints3}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)

	// Remove "bar" endpoints
	fakeWatch.Delete(endpoints2)
	endpoints = []*api.Endpoints{endpoints1v2, endpoints3}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)
}

// TODO: Add a unittest for interrupts getting processed in a timely manner.
// Currently this module has a circular dependency with config, and so it's
// named config_test, which means even test methods need to be public. This
// is refactoring that we can avoid by resolving the dependency.
