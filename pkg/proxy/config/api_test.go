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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	informers "k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	ktesting "k8s.io/client-go/testing"
)

func TestNewServicesSourceApi_UpdatesAndMultipleServices(t *testing.T) {
	service1v1 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "s1"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 10}}}}
	service1v2 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "s1"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 20}}}}
	service2 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "s2"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 30}}}}

	// Setup fake api client.
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("services", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	handler := NewServiceHandlerMock()

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	serviceConfig := NewServiceConfig(sharedInformers.Core().V1().Services(), time.Minute)
	serviceConfig.RegisterEventHandler(handler)
	go sharedInformers.Start(stopCh)
	go serviceConfig.Run(stopCh)

	// Add the first service
	fakeWatch.Add(service1v1)
	handler.ValidateServices(t, []*v1.Service{service1v1})

	// Add another service
	fakeWatch.Add(service2)
	handler.ValidateServices(t, []*v1.Service{service1v1, service2})

	// Modify service1
	fakeWatch.Modify(service1v2)
	handler.ValidateServices(t, []*v1.Service{service1v2, service2})

	// Delete service1
	fakeWatch.Delete(service1v2)
	handler.ValidateServices(t, []*v1.Service{service2})

	// Delete service2
	fakeWatch.Delete(service2)
	handler.ValidateServices(t, []*v1.Service{})
}

func TestNewEndpointsSourceApi_UpdatesAndMultipleEndpoints(t *testing.T) {
	endpoints1v1 := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "e1"},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{
				{IP: "1.2.3.4"},
			},
			Ports: []v1.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	}
	endpoints1v2 := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "e1"},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{
				{IP: "1.2.3.4"},
				{IP: "4.3.2.1"},
			},
			Ports: []v1.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	}
	endpoints2 := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "e2"},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{
				{IP: "5.6.7.8"},
			},
			Ports: []v1.EndpointPort{{Port: 80, Protocol: "TCP"}},
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

	endpointsConfig := NewEndpointsConfig(sharedInformers.Core().V1().Endpoints(), time.Minute)
	endpointsConfig.RegisterEventHandler(handler)
	go sharedInformers.Start(stopCh)
	go endpointsConfig.Run(stopCh)

	// Add the first endpoints
	fakeWatch.Add(endpoints1v1)
	handler.ValidateEndpoints(t, []*v1.Endpoints{endpoints1v1})

	// Add another endpoints
	fakeWatch.Add(endpoints2)
	handler.ValidateEndpoints(t, []*v1.Endpoints{endpoints1v1, endpoints2})

	// Modify endpoints1
	fakeWatch.Modify(endpoints1v2)
	handler.ValidateEndpoints(t, []*v1.Endpoints{endpoints1v2, endpoints2})

	// Delete endpoints1
	fakeWatch.Delete(endpoints1v2)
	handler.ValidateEndpoints(t, []*v1.Endpoints{endpoints2})

	// Delete endpoints2
	fakeWatch.Delete(endpoints2)
	handler.ValidateEndpoints(t, []*v1.Endpoints{})
}

func newSvcHandler(t *testing.T, svcs []*v1.Service, done func()) ServiceHandler {
	shm := &ServiceHandlerMock{
		state: make(map[types.NamespacedName]*v1.Service),
	}
	shm.process = func(services []*v1.Service) {
		defer done()
		if !reflect.DeepEqual(services, svcs) {
			t.Errorf("Unexpected services: %#v, expected: %#v", services, svcs)
		}
	}
	return shm
}

func newEpsHandler(t *testing.T, eps []*v1.Endpoints, done func()) EndpointsHandler {
	ehm := &EndpointsHandlerMock{
		state: make(map[types.NamespacedName]*v1.Endpoints),
	}
	ehm.process = func(endpoints []*v1.Endpoints) {
		defer done()
		if !reflect.DeepEqual(eps, endpoints) {
			t.Errorf("Unexpected endpoints: %#v, expected: %#v", endpoints, eps)
		}
	}
	return ehm
}

func TestInitialSync(t *testing.T) {
	svc1 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 10}}},
	}
	svc2 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 10}}},
	}
	eps1 := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
	}
	eps2 := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
	}

	var wg sync.WaitGroup
	// Wait for both services and endpoints handler.
	wg.Add(2)

	// Setup fake api client.
	client := fake.NewSimpleClientset(svc1, svc2, eps2, eps1)
	sharedInformers := informers.NewSharedInformerFactory(client, 0)

	svcConfig := NewServiceConfig(sharedInformers.Core().V1().Services(), 0)
	epsConfig := NewEndpointsConfig(sharedInformers.Core().V1().Endpoints(), 0)
	svcHandler := newSvcHandler(t, []*v1.Service{svc2, svc1}, wg.Done)
	svcConfig.RegisterEventHandler(svcHandler)
	epsHandler := newEpsHandler(t, []*v1.Endpoints{eps2, eps1}, wg.Done)
	epsConfig.RegisterEventHandler(epsHandler)

	stopCh := make(chan struct{})
	defer close(stopCh)
	go sharedInformers.Start(stopCh)
	go svcConfig.Run(stopCh)
	go epsConfig.Run(stopCh)
	wg.Wait()
}
