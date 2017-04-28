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
	"sync"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	ktesting "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
)

func TestNewServicesSourceApi_UpdatesAndMultipleServices(t *testing.T) {
	service1v1 := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "s1"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 10}}}}
	service1v2 := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "s1"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 20}}}}
	service2 := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "s2"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 30}}}}

	// Setup fake api client.
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("services", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	handler := NewServiceHandlerMock()

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	serviceConfig := NewServiceConfig(sharedInformers.Core().InternalVersion().Services(), time.Minute)
	serviceConfig.RegisterEventHandler(handler)
	go sharedInformers.Start(stopCh)
	go serviceConfig.Run(stopCh)

	// Add the first service
	fakeWatch.Add(service1v1)
	handler.ValidateServices(t, []*api.Service{service1v1})

	// Add another service
	fakeWatch.Add(service2)
	handler.ValidateServices(t, []*api.Service{service1v1, service2})

	// Modify service1
	fakeWatch.Modify(service1v2)
	handler.ValidateServices(t, []*api.Service{service1v2, service2})

	// Delete service1
	fakeWatch.Delete(service1v2)
	handler.ValidateServices(t, []*api.Service{service2})

	// Delete service2
	fakeWatch.Delete(service2)
	handler.ValidateServices(t, []*api.Service{})
}

func TestNewEndpointsSourceApi_UpdatesAndMultipleEndpoints(t *testing.T) {
	endpoints1v1 := &api.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "e1"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{
				{IP: "1.2.3.4"},
			},
			Ports: []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	}
	endpoints1v2 := &api.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "e1"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{
				{IP: "1.2.3.4"},
				{IP: "4.3.2.1"},
			},
			Ports: []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	}
	endpoints2 := &api.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "e2"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{
				{IP: "5.6.7.8"},
			},
			Ports: []api.EndpointPort{{Port: 80, Protocol: "TCP"}},
		}},
	}

	// Setup fake api client.
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("endpoints", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	handler := NewEndpointsHandlerMock()

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	endpointsConfig := NewEndpointsConfig(sharedInformers.Core().InternalVersion().Endpoints(), time.Minute)
	endpointsConfig.RegisterEventHandler(handler)
	go sharedInformers.Start(stopCh)
	go endpointsConfig.Run(stopCh)

	// Add the first endpoints
	fakeWatch.Add(endpoints1v1)
	handler.ValidateEndpoints(t, []*api.Endpoints{endpoints1v1})

	// Add another endpoints
	fakeWatch.Add(endpoints2)
	handler.ValidateEndpoints(t, []*api.Endpoints{endpoints1v1, endpoints2})

	// Modify endpoints1
	fakeWatch.Modify(endpoints1v2)
	handler.ValidateEndpoints(t, []*api.Endpoints{endpoints1v2, endpoints2})

	// Delete endpoints1
	fakeWatch.Delete(endpoints1v2)
	handler.ValidateEndpoints(t, []*api.Endpoints{endpoints2})

	// Delete endpoints2
	fakeWatch.Delete(endpoints2)
	handler.ValidateEndpoints(t, []*api.Endpoints{})
}

func newSvcHandler(t *testing.T, svcs []*api.Service, done func()) ServiceHandler {
	shm := &ServiceHandlerMock{
		state: make(map[types.NamespacedName]*api.Service),
	}
	shm.process = func(services []*api.Service) {
		defer done()
		if !reflect.DeepEqual(services, svcs) {
			t.Errorf("Unexpected services: %#v, expected: %#v", services, svcs)
		}
	}
	return shm
}

func newEpsHandler(t *testing.T, eps []*api.Endpoints, done func()) EndpointsHandler {
	ehm := &EndpointsHandlerMock{
		state: make(map[types.NamespacedName]*api.Endpoints),
	}
	ehm.process = func(endpoints []*api.Endpoints) {
		defer done()
		if !reflect.DeepEqual(eps, endpoints) {
			t.Errorf("Unexpected endpoints: %#v, expected: %#v", endpoints, eps)
		}
	}
	return ehm
}

func TestInitialSync(t *testing.T) {
	svc1 := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 10}}},
	}
	svc2 := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 10}}},
	}
	eps1 := &api.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
	}
	eps2 := &api.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
	}

	var wg sync.WaitGroup
	// Wait for both services and endpoints handler.
	wg.Add(2)

	// Setup fake api client.
	client := fake.NewSimpleClientset(svc1, svc2, eps2, eps1)
	sharedInformers := informers.NewSharedInformerFactory(client, 0)

	svcConfig := NewServiceConfig(sharedInformers.Core().InternalVersion().Services(), 0)
	epsConfig := NewEndpointsConfig(sharedInformers.Core().InternalVersion().Endpoints(), 0)
	svcHandler := newSvcHandler(t, []*api.Service{svc2, svc1}, wg.Done)
	svcConfig.RegisterEventHandler(svcHandler)
	epsHandler := newEpsHandler(t, []*api.Endpoints{eps2, eps1}, wg.Done)
	epsConfig.RegisterEventHandler(epsHandler)

	stopCh := make(chan struct{})
	defer close(stopCh)
	go sharedInformers.Start(stopCh)
	go svcConfig.Run(stopCh)
	go epsConfig.Run(stopCh)
	wg.Wait()
}
