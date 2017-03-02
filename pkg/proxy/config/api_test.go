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

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api"
)

type fakeLW struct {
	listResp  runtime.Object
	watchResp watch.Interface
}

func (lw fakeLW) List(options metav1.ListOptions) (runtime.Object, error) {
	return lw.listResp, nil
}

func (lw fakeLW) Watch(options metav1.ListOptions) (watch.Interface, error) {
	return lw.watchResp, nil
}

var _ cache.ListerWatcher = fakeLW{}

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
	fakeWatch := watch.NewFake()
	lw := fakeLW{
		listResp:  &api.ServiceList{Items: []api.Service{}},
		watchResp: fakeWatch,
	}

	ch := make(chan ServiceUpdate)

	serviceController := NewServiceController(lw, 30*time.Second, ch)
	go serviceController.Run(wait.NeverStop)

	// Add the first service
	fakeWatch.Add(service1v1)
	got, ok := <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expected := ServiceUpdate{Op: ADD, Service: service1v1}
	if !apiequality.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v; Got %#v", expected, got)
	}

	// Add another service
	fakeWatch.Add(service2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	// Could be sorted either of these two ways:
	expected = ServiceUpdate{Op: ADD, Service: service2}

	if !apiequality.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v, Got %#v", expected, got)
	}

	// Modify service1
	fakeWatch.Modify(service1v2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expected = ServiceUpdate{Op: UPDATE, Service: service1v2}

	if !apiequality.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v, Got %#v", expected, got)
	}

	// Delete service1
	fakeWatch.Delete(service1v2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expected = ServiceUpdate{Op: REMOVE, Service: service1v2}
	if !apiequality.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v, Got %#v", expected, got)
	}

	// Delete service2
	fakeWatch.Delete(service2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expected = ServiceUpdate{Op: REMOVE, Service: service2}
	if !apiequality.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v, Got %#v", expected, got)
	}
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
	fakeWatch := watch.NewFake()
	lw := fakeLW{
		listResp:  &api.EndpointsList{Items: []api.Endpoints{}},
		watchResp: fakeWatch,
	}

	ch := make(chan EndpointsUpdate)

	endpointsController := NewEndpointsController(lw, 30*time.Second, ch)
	go endpointsController.Run(wait.NeverStop)

	// Add the first endpoints
	fakeWatch.Add(endpoints1v1)
	got, ok := <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expected := EndpointsUpdate{Op: ADD, Endpoints: endpoints1v1}
	if !apiequality.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v; Got %#v", expected, got)
	}

	// Add another endpoints
	fakeWatch.Add(endpoints2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	// Could be sorted either of these two ways:
	expected = EndpointsUpdate{Op: ADD, Endpoints: endpoints2}

	if !apiequality.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v, Got %#v", expected, got)
	}

	// Modify endpoints1
	fakeWatch.Modify(endpoints1v2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expected = EndpointsUpdate{Op: UPDATE, Endpoints: endpoints1v2}

	if !apiequality.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v, Got %#v", expected, got)
	}

	// Delete endpoints1
	fakeWatch.Delete(endpoints1v2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expected = EndpointsUpdate{Op: REMOVE, Endpoints: endpoints1v2}
	if !apiequality.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v, Got %#v", expected, got)
	}

	// Delete endpoints2
	fakeWatch.Delete(endpoints2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expected = EndpointsUpdate{Op: REMOVE, Endpoints: endpoints2}
	if !apiequality.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v, Got %#v", expected, got)
	}
}

type svcHandler struct {
	t        *testing.T
	expected []api.Service
	done     func()
}

func newSvcHandler(t *testing.T, svcs []api.Service, done func()) *svcHandler {
	return &svcHandler{t: t, expected: svcs, done: done}
}

func (s *svcHandler) OnServiceUpdate(services []api.Service) {
	defer s.done()
	sort.Sort(sortedServices(services))
	if !reflect.DeepEqual(s.expected, services) {
		s.t.Errorf("Unexpected services: %#v, expected: %#v", services, s.expected)
	}
}

type epsHandler struct {
	t        *testing.T
	expected []api.Endpoints
	done     func()
}

func newEpsHandler(t *testing.T, eps []api.Endpoints, done func()) *epsHandler {
	return &epsHandler{t: t, expected: eps, done: done}
}

func (e *epsHandler) OnEndpointsUpdate(endpoints []api.Endpoints) {
	defer e.done()
	sort.Sort(sortedEndpoints(endpoints))
	if !reflect.DeepEqual(e.expected, endpoints) {
		e.t.Errorf("Unexpected endpoints: %#v, expected: %#v", endpoints, e.expected)
	}
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

	svcConfig := NewServiceConfig()
	epsConfig := NewEndpointsConfig()
	svcHandler := newSvcHandler(t, []api.Service{*svc2, *svc1}, wg.Done)
	svcConfig.RegisterHandler(svcHandler)
	epsHandler := newEpsHandler(t, []api.Endpoints{*eps2, *eps1}, wg.Done)
	epsConfig.RegisterHandler(epsHandler)

	// Setup fake api client.
	fakeSvcWatch := watch.NewFake()
	svcLW := fakeLW{
		listResp:  &api.ServiceList{Items: []api.Service{*svc1, *svc2}},
		watchResp: fakeSvcWatch,
	}
	fakeEpsWatch := watch.NewFake()
	epsLW := fakeLW{
		listResp:  &api.EndpointsList{Items: []api.Endpoints{*eps2, *eps1}},
		watchResp: fakeEpsWatch,
	}

	stopCh := make(chan struct{})
	defer close(stopCh)
	newSourceAPI(svcLW, epsLW, time.Minute, svcConfig.Channel("one"), epsConfig.Channel("two"), stopCh)
	wg.Wait()
}
