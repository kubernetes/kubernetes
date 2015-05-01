/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"errors"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func TestServices(t *testing.T) {
	service := api.Service{ObjectMeta: api.ObjectMeta{Name: "bar", ResourceVersion: "2"}}

	fakeWatch := watch.NewFake()
	fakeClient := &testclient.Fake{Watch: fakeWatch}
	services := make(chan ServiceUpdate)
	source := SourceAPI{
		s: servicesReflector{watcher: fakeClient.Services(api.NamespaceAll), services: services},
		e: endpointsReflector{watcher: fakeClient.Endpoints(api.NamespaceAll)}}
	resourceVersion := "1"
	go func() {
		// called twice
		source.s.run(&resourceVersion)
		source.s.run(&resourceVersion)
	}()

	// test adding a service to the watch
	fakeWatch.Add(&service)
	if !reflect.DeepEqual(fakeClient.Actions, []testclient.FakeAction{{"watch-services", "1"}}) {
		t.Errorf("expected call to watch-services, got %#v", fakeClient)
	}

	actual := <-services
	expected := ServiceUpdate{Op: ADD, Services: []api.Service{service}}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}

	// verify that a delete results in a config change
	fakeWatch.Delete(&service)
	actual = <-services
	expected = ServiceUpdate{Op: REMOVE, Services: []api.Service{service}}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}

	// verify that closing the channel results in a new call to WatchServices with a higher resource version
	newFakeWatch := watch.NewFake()
	fakeClient.Watch = newFakeWatch
	fakeWatch.Stop()

	newFakeWatch.Add(&service)
	if !reflect.DeepEqual(fakeClient.Actions, []testclient.FakeAction{{"watch-services", "1"}, {"watch-services", "2"}}) {
		t.Errorf("expected call to watch-endpoints, got %#v", fakeClient)
	}
}

func TestServicesFromZero(t *testing.T) {
	service := api.Service{ObjectMeta: api.ObjectMeta{Name: "bar", ResourceVersion: "2"}}

	fakeWatch := watch.NewFake()
	fakeWatch.Stop()
	fakeClient := testclient.NewSimpleFake(&api.ServiceList{
		ListMeta: api.ListMeta{ResourceVersion: "2"},
		Items: []api.Service{
			service,
		},
	})
	fakeClient.Watch = fakeWatch
	services := make(chan ServiceUpdate)
	source := SourceAPI{
		s: servicesReflector{watcher: fakeClient.Services(api.NamespaceAll), services: services},
		e: endpointsReflector{watcher: fakeClient.Endpoints(api.NamespaceAll)}}
	resourceVersion := ""
	ch := make(chan struct{})
	go func() {
		source.s.run(&resourceVersion)
		close(ch)
	}()

	// should get services SET
	actual := <-services
	expected := ServiceUpdate{Op: SET, Services: []api.Service{service}}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}

	// should have listed, then watched
	<-ch
	if resourceVersion != "2" {
		t.Errorf("unexpected resource version, got %#v", resourceVersion)
	}
	if !reflect.DeepEqual(fakeClient.Actions, []testclient.FakeAction{{"list-services", nil}, {"watch-services", "2"}}) {
		t.Errorf("unexpected actions, got %#v", fakeClient)
	}
}

func TestServicesError(t *testing.T) {
	fakeClient := &testclient.Fake{Err: errors.New("test")}
	services := make(chan ServiceUpdate)
	source := SourceAPI{
		s: servicesReflector{watcher: fakeClient.Services(api.NamespaceAll), services: services},
		e: endpointsReflector{watcher: fakeClient.Endpoints(api.NamespaceAll)}}
	resourceVersion := "1"
	ch := make(chan struct{})
	go func() {
		source.s.run(&resourceVersion)
		close(ch)
	}()

	// should have listed only
	<-ch
	if resourceVersion != "" {
		t.Errorf("unexpected resource version, got %#v", resourceVersion)
	}
	if !reflect.DeepEqual(fakeClient.Actions, []testclient.FakeAction{{"watch-services", "1"}}) {
		t.Errorf("unexpected actions, got %#v", fakeClient)
	}
}

func TestServicesErrorTimeout(t *testing.T) {
	fakeClient := &testclient.Fake{Err: errors.New("use of closed network connection")}
	services := make(chan ServiceUpdate)
	source := SourceAPI{
		s: servicesReflector{watcher: fakeClient.Services(api.NamespaceAll), services: services},
		e: endpointsReflector{watcher: fakeClient.Endpoints(api.NamespaceAll)}}
	resourceVersion := "1"
	ch := make(chan struct{})
	go func() {
		source.s.run(&resourceVersion)
		close(ch)
	}()

	// should have listed only
	<-ch
	if resourceVersion != "1" {
		t.Errorf("unexpected resource version, got %#v", resourceVersion)
	}
	if !reflect.DeepEqual(fakeClient.Actions, []testclient.FakeAction{{"watch-services", "1"}}) {
		t.Errorf("unexpected actions, got %#v", fakeClient)
	}
}

func TestServicesFromZeroError(t *testing.T) {
	fakeClient := &testclient.Fake{Err: errors.New("test")}
	services := make(chan ServiceUpdate)
	source := SourceAPI{
		s: servicesReflector{watcher: fakeClient.Services(api.NamespaceAll), services: services},
		e: endpointsReflector{watcher: fakeClient.Endpoints(api.NamespaceAll)}}
	resourceVersion := ""
	ch := make(chan struct{})
	go func() {
		source.s.run(&resourceVersion)
		close(ch)
	}()

	// should have listed only
	<-ch
	if resourceVersion != "" {
		t.Errorf("unexpected resource version, got %#v", resourceVersion)
	}
	if !reflect.DeepEqual(fakeClient.Actions, []testclient.FakeAction{{"list-services", nil}}) {
		t.Errorf("unexpected actions, got %#v", fakeClient)
	}
}

func TestEndpoints(t *testing.T) {
	endpoint := api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: "bar", ResourceVersion: "2"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []api.EndpointPort{{Port: 9000}},
		}},
	}

	fakeWatch := watch.NewFake()
	fakeClient := &testclient.Fake{Watch: fakeWatch}
	endpoints := make(chan EndpointsUpdate)
	source := SourceAPI{
		s: servicesReflector{watcher: fakeClient.Services(api.NamespaceAll)},
		e: endpointsReflector{watcher: fakeClient.Endpoints(api.NamespaceAll), endpoints: endpoints}}
	resourceVersion := "1"
	go func() {
		// called twice
		source.e.run(&resourceVersion)
		source.e.run(&resourceVersion)
	}()

	// test adding an endpoint to the watch
	fakeWatch.Add(&endpoint)
	if !reflect.DeepEqual(fakeClient.Actions, []testclient.FakeAction{{"watch-endpoints", "1"}}) {
		t.Errorf("expected call to watch-endpoints, got %#v", fakeClient)
	}

	actual := <-endpoints
	expected := EndpointsUpdate{Op: ADD, Endpoints: []api.Endpoints{endpoint}}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}

	// verify that a delete results in a config change
	fakeWatch.Delete(&endpoint)
	actual = <-endpoints
	expected = EndpointsUpdate{Op: REMOVE, Endpoints: []api.Endpoints{endpoint}}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}

	// verify that closing the channel results in a new call to WatchEndpoints with a higher resource version
	newFakeWatch := watch.NewFake()
	fakeClient.Watch = newFakeWatch
	fakeWatch.Stop()

	newFakeWatch.Add(&endpoint)
	if !reflect.DeepEqual(fakeClient.Actions, []testclient.FakeAction{{"watch-endpoints", "1"}, {"watch-endpoints", "2"}}) {
		t.Errorf("expected call to watch-endpoints, got %#v", fakeClient)
	}
}

func TestEndpointsFromZero(t *testing.T) {
	endpoint := api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: "bar", ResourceVersion: "2"},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []api.EndpointPort{{Port: 9000}},
		}},
	}

	fakeWatch := watch.NewFake()
	fakeWatch.Stop()
	fakeClient := testclient.NewSimpleFake(&api.EndpointsList{
		ListMeta: api.ListMeta{ResourceVersion: "2"},
		Items: []api.Endpoints{
			endpoint,
		},
	})
	fakeClient.Watch = fakeWatch
	endpoints := make(chan EndpointsUpdate)
	source := SourceAPI{
		s: servicesReflector{watcher: fakeClient.Services(api.NamespaceAll)},
		e: endpointsReflector{watcher: fakeClient.Endpoints(api.NamespaceAll), endpoints: endpoints}}
	resourceVersion := ""
	ch := make(chan struct{})
	go func() {
		source.e.run(&resourceVersion)
		close(ch)
	}()

	// should get endpoints SET
	actual := <-endpoints
	expected := EndpointsUpdate{Op: SET, Endpoints: []api.Endpoints{endpoint}}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}

	// should have listed, then watched
	<-ch
	if resourceVersion != "2" {
		t.Errorf("unexpected resource version, got %#v", resourceVersion)
	}
	if !reflect.DeepEqual(fakeClient.Actions, []testclient.FakeAction{{"list-endpoints", nil}, {"watch-endpoints", "2"}}) {
		t.Errorf("unexpected actions, got %#v", fakeClient)
	}
}

func TestEndpointsError(t *testing.T) {
	fakeClient := &testclient.Fake{Err: errors.New("test")}
	endpoints := make(chan EndpointsUpdate)
	source := SourceAPI{
		s: servicesReflector{watcher: fakeClient.Services(api.NamespaceAll)},
		e: endpointsReflector{watcher: fakeClient.Endpoints(api.NamespaceAll), endpoints: endpoints}}
	resourceVersion := "1"
	ch := make(chan struct{})
	go func() {
		source.e.run(&resourceVersion)
		close(ch)
	}()

	// should have listed only
	<-ch
	if resourceVersion != "" {
		t.Errorf("unexpected resource version, got %#v", resourceVersion)
	}
	if !reflect.DeepEqual(fakeClient.Actions, []testclient.FakeAction{{"watch-endpoints", "1"}}) {
		t.Errorf("unexpected actions, got %#v", fakeClient)
	}
}

func TestEndpointsErrorTimeout(t *testing.T) {
	fakeClient := &testclient.Fake{Err: errors.New("use of closed network connection")}
	endpoints := make(chan EndpointsUpdate)
	source := SourceAPI{
		s: servicesReflector{watcher: fakeClient.Services(api.NamespaceAll)},
		e: endpointsReflector{watcher: fakeClient.Endpoints(api.NamespaceAll), endpoints: endpoints}}
	resourceVersion := "1"
	ch := make(chan struct{})
	go func() {
		source.e.run(&resourceVersion)
		close(ch)
	}()

	// should have listed only
	<-ch
	if resourceVersion != "1" {
		t.Errorf("unexpected resource version, got %#v", resourceVersion)
	}
	if !reflect.DeepEqual(fakeClient.Actions, []testclient.FakeAction{{"watch-endpoints", "1"}}) {
		t.Errorf("unexpected actions, got %#v", fakeClient)
	}
}

func TestEndpointsFromZeroError(t *testing.T) {
	fakeClient := &testclient.Fake{Err: errors.New("test")}
	endpoints := make(chan EndpointsUpdate)
	source := SourceAPI{
		s: servicesReflector{watcher: fakeClient.Services(api.NamespaceAll)},
		e: endpointsReflector{watcher: fakeClient.Endpoints(api.NamespaceAll), endpoints: endpoints}}
	resourceVersion := ""
	ch := make(chan struct{})
	go func() {
		source.e.run(&resourceVersion)
		close(ch)
	}()

	// should have listed only
	<-ch
	if resourceVersion != "" {
		t.Errorf("unexpected resource version, got %#v", resourceVersion)
	}
	if !reflect.DeepEqual(fakeClient.Actions, []testclient.FakeAction{{"list-endpoints", nil}}) {
		t.Errorf("unexpected actions, got %#v", fakeClient)
	}
}
