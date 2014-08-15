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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func TestServices(t *testing.T) {
	service := api.Service{JSONBase: api.JSONBase{ID: "bar", ResourceVersion: uint64(2)}}

	fakeWatch := watch.NewFake()
	fakeClient := &client.Fake{Watch: fakeWatch}
	services := make(chan ServiceUpdate)
	source := SourceAPI{client: fakeClient, services: services}
	resourceVersion := uint64(0)
	go func() {
		// called twice
		source.runServices(&resourceVersion)
		source.runServices(&resourceVersion)
	}()

	// test adding a service to the watch
	fakeWatch.Add(&service)
	if !reflect.DeepEqual(fakeClient.Actions, []client.FakeAction{{"watch-services", uint64(0)}}) {
		t.Errorf("expected call to watch-services, got %#v", fakeClient)
	}

	actual := <-services
	expected := ServiceUpdate{Op: SET, Services: []api.Service{service}}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}

	// verify that a delete results in a config change
	fakeWatch.Delete(&service)
	actual = <-services
	expected = ServiceUpdate{Op: SET}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}

	// verify that closing the channel results in a new call to WatchServices with a higher resource version
	newFakeWatch := watch.NewFake()
	fakeClient.Watch = newFakeWatch
	fakeWatch.Stop()

	newFakeWatch.Add(&service)
	if !reflect.DeepEqual(fakeClient.Actions, []client.FakeAction{{"watch-services", uint64(0)}, {"watch-services", uint64(3)}}) {
		t.Errorf("expected call to watch-endpoints, got %#v", fakeClient)
	}
}

func TestEndpoints(t *testing.T) {
	endpoint := api.Endpoints{JSONBase: api.JSONBase{ID: "bar", ResourceVersion: uint64(2)}, Endpoints: []string{"127.0.0.1:9000"}}

	fakeWatch := watch.NewFake()
	fakeClient := &client.Fake{Watch: fakeWatch}
	endpoints := make(chan EndpointsUpdate)
	source := SourceAPI{client: fakeClient, endpoints: endpoints}
	resourceVersion := uint64(0)
	go func() {
		// called twice
		source.runEndpoints(&resourceVersion)
		source.runEndpoints(&resourceVersion)
	}()

	// test adding an endpoint to the watch
	fakeWatch.Add(&endpoint)
	if !reflect.DeepEqual(fakeClient.Actions, []client.FakeAction{{"watch-endpoints", uint64(0)}}) {
		t.Errorf("expected call to watch-endpoints, got %#v", fakeClient)
	}

	actual := <-endpoints
	expected := EndpointsUpdate{Op: SET, Endpoints: []api.Endpoints{endpoint}}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}

	// verify that a delete results in a config change
	fakeWatch.Delete(&endpoint)
	actual = <-endpoints
	expected = EndpointsUpdate{Op: SET}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}

	// verify that closing the channel results in a new call to WatchEndpoints with a higher resource version
	newFakeWatch := watch.NewFake()
	fakeClient.Watch = newFakeWatch
	fakeWatch.Stop()

	newFakeWatch.Add(&endpoint)
	if !reflect.DeepEqual(fakeClient.Actions, []client.FakeAction{{"watch-endpoints", uint64(0)}, {"watch-endpoints", uint64(3)}}) {
		t.Errorf("expected call to watch-endpoints, got %#v", fakeClient)
	}
}
