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
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

type fakeLW struct {
	listResp  runtime.Object
	watchResp watch.Interface
}

func (lw fakeLW) List(options api.ListOptions) (runtime.Object, error) {
	return lw.listResp, nil
}

func (lw fakeLW) Watch(options api.ListOptions) (watch.Interface, error) {
	return lw.watchResp, nil
}

var _ cache.ListerWatcher = fakeLW{}

func TestNewServicesSourceApi_UpdatesAndMultipleServices(t *testing.T) {
	service1v1 := &api.Service{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "s1"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 10}}}}
	service1v2 := &api.Service{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "s1"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 20}}}}
	service2 := &api.Service{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "s2"},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Protocol: "TCP", Port: 30}}}}

	// Setup fake api client.
	fakeWatch := watch.NewFake()
	lw := fakeLW{
		listResp:  &api.ServiceList{Items: []api.Service{}},
		watchResp: fakeWatch,
	}

	ch := make(chan ServiceUpdate)

	cache.NewReflector(lw, &api.Service{}, NewServiceStore(nil, ch), 30*time.Second).Run()

	got, ok := <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expected := ServiceUpdate{Op: SET, Services: []api.Service{}}
	if !api.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v; Got %#v", expected, got)
	}

	// Add the first service
	fakeWatch.Add(service1v1)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expected = ServiceUpdate{Op: SET, Services: []api.Service{*service1v1}}
	if !api.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v; Got %#v", expected, got)
	}

	// Add another service
	fakeWatch.Add(service2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	// Could be sorted either of these two ways:
	expectedA := ServiceUpdate{Op: SET, Services: []api.Service{*service1v1, *service2}}
	expectedB := ServiceUpdate{Op: SET, Services: []api.Service{*service2, *service1v1}}

	if !api.Semantic.DeepEqual(expectedA, got) && !api.Semantic.DeepEqual(expectedB, got) {
		t.Errorf("Expected %#v or %#v, Got %#v", expectedA, expectedB, got)
	}

	// Modify service1
	fakeWatch.Modify(service1v2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expectedA = ServiceUpdate{Op: SET, Services: []api.Service{*service1v2, *service2}}
	expectedB = ServiceUpdate{Op: SET, Services: []api.Service{*service2, *service1v2}}

	if !api.Semantic.DeepEqual(expectedA, got) && !api.Semantic.DeepEqual(expectedB, got) {
		t.Errorf("Expected %#v or %#v, Got %#v", expectedA, expectedB, got)
	}

	// Delete service1
	fakeWatch.Delete(service1v2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expected = ServiceUpdate{Op: SET, Services: []api.Service{*service2}}
	if !api.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v, Got %#v", expected, got)
	}

	// Delete service2
	fakeWatch.Delete(service2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expected = ServiceUpdate{Op: SET, Services: []api.Service{}}
	if !api.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v, Got %#v", expected, got)
	}
}

func TestNewEndpointsSourceApi_UpdatesAndMultipleEndpoints(t *testing.T) {
	endpoints1v1 := &api.Endpoints{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "e1"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{
				{IP: "1.2.3.4"},
			},
			Ports: []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	}
	endpoints1v2 := &api.Endpoints{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "e1"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{
				{IP: "1.2.3.4"},
				{IP: "4.3.2.1"},
			},
			Ports: []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	}
	endpoints2 := &api.Endpoints{
		ObjectMeta: api.ObjectMeta{Namespace: "testnamespace", Name: "e2"},
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

	cache.NewReflector(lw, &api.Endpoints{}, NewEndpointsStore(nil, ch), 30*time.Second).Run()

	got, ok := <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expected := EndpointsUpdate{Op: SET, Endpoints: []api.Endpoints{}}
	if !api.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v; Got %#v", expected, got)
	}

	// Add the first endpoints
	fakeWatch.Add(endpoints1v1)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expected = EndpointsUpdate{Op: SET, Endpoints: []api.Endpoints{*endpoints1v1}}
	if !api.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v; Got %#v", expected, got)
	}

	// Add another endpoints
	fakeWatch.Add(endpoints2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	// Could be sorted either of these two ways:
	expectedA := EndpointsUpdate{Op: SET, Endpoints: []api.Endpoints{*endpoints1v1, *endpoints2}}
	expectedB := EndpointsUpdate{Op: SET, Endpoints: []api.Endpoints{*endpoints2, *endpoints1v1}}

	if !api.Semantic.DeepEqual(expectedA, got) && !api.Semantic.DeepEqual(expectedB, got) {
		t.Errorf("Expected %#v or %#v, Got %#v", expectedA, expectedB, got)
	}

	// Modify endpoints1
	fakeWatch.Modify(endpoints1v2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expectedA = EndpointsUpdate{Op: SET, Endpoints: []api.Endpoints{*endpoints1v2, *endpoints2}}
	expectedB = EndpointsUpdate{Op: SET, Endpoints: []api.Endpoints{*endpoints2, *endpoints1v2}}

	if !api.Semantic.DeepEqual(expectedA, got) && !api.Semantic.DeepEqual(expectedB, got) {
		t.Errorf("Expected %#v or %#v, Got %#v", expectedA, expectedB, got)
	}

	// Delete endpoints1
	fakeWatch.Delete(endpoints1v2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expected = EndpointsUpdate{Op: SET, Endpoints: []api.Endpoints{*endpoints2}}
	if !api.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v, Got %#v", expected, got)
	}

	// Delete endpoints2
	fakeWatch.Delete(endpoints2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	expected = EndpointsUpdate{Op: SET, Endpoints: []api.Endpoints{}}
	if !api.Semantic.DeepEqual(expected, got) {
		t.Errorf("Expected %#v, Got %#v", expected, got)
	}
}
