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
	"sync"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	informers "k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	ktesting "k8s.io/client-go/testing"
)

type sortedServices []*v1.Service

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
	lock sync.Mutex

	state   map[types.NamespacedName]*v1.Service
	synced  bool
	updated chan []*v1.Service
	process func([]*v1.Service)
}

func NewServiceHandlerMock() *ServiceHandlerMock {
	shm := &ServiceHandlerMock{
		state:   make(map[types.NamespacedName]*v1.Service),
		updated: make(chan []*v1.Service, 5),
	}
	shm.process = func(services []*v1.Service) {
		shm.updated <- services
	}
	return shm
}

func (h *ServiceHandlerMock) OnServiceAdd(service *v1.Service) {
	h.lock.Lock()
	defer h.lock.Unlock()
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	h.state[namespacedName] = service
	h.sendServices()
}

func (h *ServiceHandlerMock) OnServiceUpdate(oldService, service *v1.Service) {
	h.lock.Lock()
	defer h.lock.Unlock()
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	h.state[namespacedName] = service
	h.sendServices()
}

func (h *ServiceHandlerMock) OnServiceDelete(service *v1.Service) {
	h.lock.Lock()
	defer h.lock.Unlock()
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	delete(h.state, namespacedName)
	h.sendServices()
}

func (h *ServiceHandlerMock) OnServiceSynced() {
	h.lock.Lock()
	defer h.lock.Unlock()
	h.synced = true
	h.sendServices()
}

func (h *ServiceHandlerMock) sendServices() {
	if !h.synced {
		return
	}
	services := make([]*v1.Service, 0, len(h.state))
	for _, svc := range h.state {
		services = append(services, svc)
	}
	sort.Sort(sortedServices(services))
	h.process(services)
}

func (h *ServiceHandlerMock) ValidateServices(t *testing.T, expectedServices []*v1.Service) {
	// We might get 1 or more updates for N service updates, because we
	// over write older snapshots of services from the producer go-routine
	// if the consumer falls behind.
	var services []*v1.Service
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

type sortedEndpoints []*v1.Endpoints

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
	lock sync.Mutex

	state   map[types.NamespacedName]*v1.Endpoints
	synced  bool
	updated chan []*v1.Endpoints
	process func([]*v1.Endpoints)
}

func NewEndpointsHandlerMock() *EndpointsHandlerMock {
	ehm := &EndpointsHandlerMock{
		state:   make(map[types.NamespacedName]*v1.Endpoints),
		updated: make(chan []*v1.Endpoints, 5),
	}
	ehm.process = func(endpoints []*v1.Endpoints) {
		ehm.updated <- endpoints
	}
	return ehm
}

func (h *EndpointsHandlerMock) OnEndpointsAdd(endpoints *v1.Endpoints) {
	h.lock.Lock()
	defer h.lock.Unlock()
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	h.state[namespacedName] = endpoints
	h.sendEndpoints()
}

func (h *EndpointsHandlerMock) OnEndpointsUpdate(oldEndpoints, endpoints *v1.Endpoints) {
	h.lock.Lock()
	defer h.lock.Unlock()
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	h.state[namespacedName] = endpoints
	h.sendEndpoints()
}

func (h *EndpointsHandlerMock) OnEndpointsDelete(endpoints *v1.Endpoints) {
	h.lock.Lock()
	defer h.lock.Unlock()
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	delete(h.state, namespacedName)
	h.sendEndpoints()
}

func (h *EndpointsHandlerMock) OnEndpointsSynced() {
	h.lock.Lock()
	defer h.lock.Unlock()
	h.synced = true
	h.sendEndpoints()
}

func (h *EndpointsHandlerMock) sendEndpoints() {
	if !h.synced {
		return
	}
	endpoints := make([]*v1.Endpoints, 0, len(h.state))
	for _, eps := range h.state {
		endpoints = append(endpoints, eps)
	}
	sort.Sort(sortedEndpoints(endpoints))
	h.process(endpoints)
}

func (h *EndpointsHandlerMock) ValidateEndpoints(t *testing.T, expectedEndpoints []*v1.Endpoints) {
	// We might get 1 or more updates for N endpoint updates, because we
	// over write older snapshots of endpoints from the producer go-routine
	// if the consumer falls behind. Unittests will hard timeout in 5m.
	var endpoints []*v1.Endpoints
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
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("services", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	config := NewServiceConfig(sharedInformers.Core().V1().Services(), time.Minute)
	handler := NewServiceHandlerMock()
	config.RegisterEventHandler(handler)
	go sharedInformers.Start(stopCh)
	go config.Run(stopCh)

	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 10}}},
	}
	fakeWatch.Add(service)
	handler.ValidateServices(t, []*v1.Service{service})
}

func TestServiceAddedRemovedSetAndNotified(t *testing.T) {
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("services", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	config := NewServiceConfig(sharedInformers.Core().V1().Services(), time.Minute)
	handler := NewServiceHandlerMock()
	config.RegisterEventHandler(handler)
	go sharedInformers.Start(stopCh)
	go config.Run(stopCh)

	service1 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 10}}},
	}
	fakeWatch.Add(service1)
	handler.ValidateServices(t, []*v1.Service{service1})

	service2 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 20}}},
	}
	fakeWatch.Add(service2)
	services := []*v1.Service{service2, service1}
	handler.ValidateServices(t, services)

	fakeWatch.Delete(service1)
	services = []*v1.Service{service2}
	handler.ValidateServices(t, services)
}

func TestNewServicesMultipleHandlersAddedAndNotified(t *testing.T) {
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("services", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	config := NewServiceConfig(sharedInformers.Core().V1().Services(), time.Minute)
	handler := NewServiceHandlerMock()
	handler2 := NewServiceHandlerMock()
	config.RegisterEventHandler(handler)
	config.RegisterEventHandler(handler2)
	go sharedInformers.Start(stopCh)
	go config.Run(stopCh)

	service1 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 10}}},
	}
	service2 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 20}}},
	}
	fakeWatch.Add(service1)
	fakeWatch.Add(service2)

	services := []*v1.Service{service2, service1}
	handler.ValidateServices(t, services)
	handler2.ValidateServices(t, services)
}

func TestNewEndpointsMultipleHandlersAddedAndNotified(t *testing.T) {
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("endpoints", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	config := NewEndpointsConfig(sharedInformers.Core().V1().Endpoints(), time.Minute)
	handler := NewEndpointsHandlerMock()
	handler2 := NewEndpointsHandlerMock()
	config.RegisterEventHandler(handler)
	config.RegisterEventHandler(handler2)
	go sharedInformers.Start(stopCh)
	go config.Run(stopCh)

	endpoints1 := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "1.1.1.1"}, {IP: "2.2.2.2"}},
			Ports:     []v1.EndpointPort{{Port: 80}},
		}},
	}
	endpoints2 := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "3.3.3.3"}, {IP: "4.4.4.4"}},
			Ports:     []v1.EndpointPort{{Port: 80}},
		}},
	}
	fakeWatch.Add(endpoints1)
	fakeWatch.Add(endpoints2)

	endpoints := []*v1.Endpoints{endpoints2, endpoints1}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)
}

func TestNewEndpointsMultipleHandlersAddRemoveSetAndNotified(t *testing.T) {
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("endpoints", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	config := NewEndpointsConfig(sharedInformers.Core().V1().Endpoints(), time.Minute)
	handler := NewEndpointsHandlerMock()
	handler2 := NewEndpointsHandlerMock()
	config.RegisterEventHandler(handler)
	config.RegisterEventHandler(handler2)
	go sharedInformers.Start(stopCh)
	go config.Run(stopCh)

	endpoints1 := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "1.1.1.1"}, {IP: "2.2.2.2"}},
			Ports:     []v1.EndpointPort{{Port: 80}},
		}},
	}
	endpoints2 := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "3.3.3.3"}, {IP: "4.4.4.4"}},
			Ports:     []v1.EndpointPort{{Port: 80}},
		}},
	}
	fakeWatch.Add(endpoints1)
	fakeWatch.Add(endpoints2)

	endpoints := []*v1.Endpoints{endpoints2, endpoints1}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)

	// Add one more
	endpoints3 := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foobar"},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "5.5.5.5"}, {IP: "6.6.6.6"}},
			Ports:     []v1.EndpointPort{{Port: 80}},
		}},
	}
	fakeWatch.Add(endpoints3)
	endpoints = []*v1.Endpoints{endpoints2, endpoints1, endpoints3}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)

	// Update the "foo" service with new endpoints
	endpoints1v2 := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "7.7.7.7"}},
			Ports:     []v1.EndpointPort{{Port: 80}},
		}},
	}
	fakeWatch.Modify(endpoints1v2)
	endpoints = []*v1.Endpoints{endpoints2, endpoints1v2, endpoints3}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)

	// Remove "bar" endpoints
	fakeWatch.Delete(endpoints2)
	endpoints = []*v1.Endpoints{endpoints1v2, endpoints3}
	handler.ValidateEndpoints(t, endpoints)
	handler2.ValidateEndpoints(t, endpoints)
}

// TODO: Add a unittest for interrupts getting processed in a timely manner.
// Currently this module has a circular dependency with config, and so it's
// named config_test, which means even test methods need to be public. This
// is refactoring that we can avoid by resolving the dependency.
