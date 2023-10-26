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
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	informers "k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	ktesting "k8s.io/client-go/testing"
	"k8s.io/utils/ptr"
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

type sortedEndpointSlices []*discoveryv1.EndpointSlice

func (s sortedEndpointSlices) Len() int {
	return len(s)
}
func (s sortedEndpointSlices) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
func (s sortedEndpointSlices) Less(i, j int) bool {
	return s[i].Name < s[j].Name
}

type EndpointSliceHandlerMock struct {
	lock sync.Mutex

	state   map[types.NamespacedName]*discoveryv1.EndpointSlice
	synced  bool
	updated chan []*discoveryv1.EndpointSlice
	process func([]*discoveryv1.EndpointSlice)
}

func NewEndpointSliceHandlerMock() *EndpointSliceHandlerMock {
	ehm := &EndpointSliceHandlerMock{
		state:   make(map[types.NamespacedName]*discoveryv1.EndpointSlice),
		updated: make(chan []*discoveryv1.EndpointSlice, 5),
	}
	ehm.process = func(endpoints []*discoveryv1.EndpointSlice) {
		ehm.updated <- endpoints
	}
	return ehm
}

func (h *EndpointSliceHandlerMock) OnEndpointSliceAdd(slice *discoveryv1.EndpointSlice) {
	h.lock.Lock()
	defer h.lock.Unlock()
	namespacedName := types.NamespacedName{Namespace: slice.Namespace, Name: slice.Name}
	h.state[namespacedName] = slice
	h.sendEndpointSlices()
}

func (h *EndpointSliceHandlerMock) OnEndpointSliceUpdate(oldSlice, slice *discoveryv1.EndpointSlice) {
	h.lock.Lock()
	defer h.lock.Unlock()
	namespacedName := types.NamespacedName{Namespace: slice.Namespace, Name: slice.Name}
	h.state[namespacedName] = slice
	h.sendEndpointSlices()
}

func (h *EndpointSliceHandlerMock) OnEndpointSliceDelete(slice *discoveryv1.EndpointSlice) {
	h.lock.Lock()
	defer h.lock.Unlock()
	namespacedName := types.NamespacedName{Namespace: slice.Namespace, Name: slice.Name}
	delete(h.state, namespacedName)
	h.sendEndpointSlices()
}

func (h *EndpointSliceHandlerMock) OnEndpointSlicesSynced() {
	h.lock.Lock()
	defer h.lock.Unlock()
	h.synced = true
	h.sendEndpointSlices()
}

func (h *EndpointSliceHandlerMock) sendEndpointSlices() {
	if !h.synced {
		return
	}
	slices := make([]*discoveryv1.EndpointSlice, 0, len(h.state))
	for _, eps := range h.state {
		slices = append(slices, eps)
	}
	sort.Sort(sortedEndpointSlices(slices))
	h.process(slices)
}

func (h *EndpointSliceHandlerMock) ValidateEndpointSlices(t *testing.T, expectedSlices []*discoveryv1.EndpointSlice) {
	// We might get 1 or more updates for N endpointslice updates, because we
	// over write older snapshots of endpointslices from the producer go-routine
	// if the consumer falls behind. Unittests will hard timeout in 5m.
	var slices []*discoveryv1.EndpointSlice
	for {
		select {
		case slices = <-h.updated:
			if reflect.DeepEqual(slices, expectedSlices) {
				return
			}
		// Unittests will hard timeout in 5m with a stack trace, prevent that
		// and surface a clearer reason for failure.
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Timed out. Expected %#v, Got %#v", expectedSlices, slices)
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
	client.PrependWatchReactor("endpointslices", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	config := NewEndpointSliceConfig(sharedInformers.Discovery().V1().EndpointSlices(), time.Minute)
	handler := NewEndpointSliceHandlerMock()
	handler2 := NewEndpointSliceHandlerMock()
	config.RegisterEventHandler(handler)
	config.RegisterEventHandler(handler2)
	go sharedInformers.Start(stopCh)
	go config.Run(stopCh)

	endpoints1 := &discoveryv1.EndpointSlice{
		ObjectMeta:  metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		AddressType: discoveryv1.AddressTypeIPv4,
		Endpoints: []discoveryv1.Endpoint{{
			Addresses: []string{"1.1.1.1"},
		}, {
			Addresses: []string{"2.2.2.2"},
		}},
		Ports: []discoveryv1.EndpointPort{{Port: ptr.To[int32](80)}},
	}
	endpoints2 := &discoveryv1.EndpointSlice{
		ObjectMeta:  metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		AddressType: discoveryv1.AddressTypeIPv4,
		Endpoints: []discoveryv1.Endpoint{{
			Addresses: []string{"3.3.3.3"},
		}, {
			Addresses: []string{"4.4.4.4"},
		}},
		Ports: []discoveryv1.EndpointPort{{Port: ptr.To[int32](80)}},
	}
	fakeWatch.Add(endpoints1)
	fakeWatch.Add(endpoints2)

	endpoints := []*discoveryv1.EndpointSlice{endpoints2, endpoints1}
	handler.ValidateEndpointSlices(t, endpoints)
	handler2.ValidateEndpointSlices(t, endpoints)
}

func TestNewEndpointsMultipleHandlersAddRemoveSetAndNotified(t *testing.T) {
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("endpointslices", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	config := NewEndpointSliceConfig(sharedInformers.Discovery().V1().EndpointSlices(), time.Minute)
	handler := NewEndpointSliceHandlerMock()
	handler2 := NewEndpointSliceHandlerMock()
	config.RegisterEventHandler(handler)
	config.RegisterEventHandler(handler2)
	go sharedInformers.Start(stopCh)
	go config.Run(stopCh)

	endpoints1 := &discoveryv1.EndpointSlice{
		ObjectMeta:  metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		AddressType: discoveryv1.AddressTypeIPv4,
		Endpoints: []discoveryv1.Endpoint{{
			Addresses: []string{"1.1.1.1"},
		}, {
			Addresses: []string{"2.2.2.2"},
		}},
		Ports: []discoveryv1.EndpointPort{{Port: ptr.To[int32](80)}},
	}
	endpoints2 := &discoveryv1.EndpointSlice{
		ObjectMeta:  metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		AddressType: discoveryv1.AddressTypeIPv4,
		Endpoints: []discoveryv1.Endpoint{{
			Addresses: []string{"3.3.3.3"},
		}, {
			Addresses: []string{"4.4.4.4"},
		}},
		Ports: []discoveryv1.EndpointPort{{Port: ptr.To[int32](80)}},
	}
	fakeWatch.Add(endpoints1)
	fakeWatch.Add(endpoints2)

	endpoints := []*discoveryv1.EndpointSlice{endpoints2, endpoints1}
	handler.ValidateEndpointSlices(t, endpoints)
	handler2.ValidateEndpointSlices(t, endpoints)

	// Add one more
	endpoints3 := &discoveryv1.EndpointSlice{
		ObjectMeta:  metav1.ObjectMeta{Namespace: "testnamespace", Name: "foobar"},
		AddressType: discoveryv1.AddressTypeIPv4,
		Endpoints: []discoveryv1.Endpoint{{
			Addresses: []string{"5.5.5.5"},
		}, {
			Addresses: []string{"6.6.6.6"},
		}},
		Ports: []discoveryv1.EndpointPort{{Port: ptr.To[int32](80)}},
	}
	fakeWatch.Add(endpoints3)
	endpoints = []*discoveryv1.EndpointSlice{endpoints2, endpoints1, endpoints3}
	handler.ValidateEndpointSlices(t, endpoints)
	handler2.ValidateEndpointSlices(t, endpoints)

	// Update the "foo" service with new endpoints
	endpoints1v2 := &discoveryv1.EndpointSlice{
		ObjectMeta:  metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		AddressType: discoveryv1.AddressTypeIPv4,
		Endpoints: []discoveryv1.Endpoint{{
			Addresses: []string{"7.7.7.7"},
		}},
		Ports: []discoveryv1.EndpointPort{{Port: ptr.To[int32](80)}},
	}
	fakeWatch.Modify(endpoints1v2)
	endpoints = []*discoveryv1.EndpointSlice{endpoints2, endpoints1v2, endpoints3}
	handler.ValidateEndpointSlices(t, endpoints)
	handler2.ValidateEndpointSlices(t, endpoints)

	// Remove "bar" endpoints
	fakeWatch.Delete(endpoints2)
	endpoints = []*discoveryv1.EndpointSlice{endpoints1v2, endpoints3}
	handler.ValidateEndpointSlices(t, endpoints)
	handler2.ValidateEndpointSlices(t, endpoints)
}

// TODO: Add a unittest for interrupts getting processed in a timely manner.
// Currently this module has a circular dependency with config, and so it's
// named config_test, which means even test methods need to be public. This
// is refactoring that we can avoid by resolving the dependency.
