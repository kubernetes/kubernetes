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

package endpoint

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
)

func TestGetEndpoints(t *testing.T) {
	registry := &registrytest.ServiceRegistry{
		Endpoints: api.Endpoints{
			JSONBase:  api.JSONBase{ID: "foo"},
			Endpoints: []string{"127.0.0.1:9000"},
		},
	}
	storage := NewStorage(registry)
	obj, err := storage.Get("foo")
	if err != nil {
		t.Fatalf("unexpected error: %#v", err)
	}
	if !reflect.DeepEqual([]string{"127.0.0.1:9000"}, obj.(*api.Endpoints).Endpoints) {
		t.Errorf("unexpected endpoints: %#v", obj)
	}
}

func TestGetEndpointsMissingService(t *testing.T) {
	registry := &registrytest.ServiceRegistry{
		Err: apiserver.NewNotFoundErr("service", "foo"),
	}
	storage := NewStorage(registry)

	// returns service not found
	_, err := storage.Get("foo")
	if !apiserver.IsNotFound(err) || !reflect.DeepEqual(err, apiserver.NewNotFoundErr("service", "foo")) {
		t.Errorf("expected NotFound error, got %#v", err)
	}

	// returns empty endpoints
	registry.Err = nil
	registry.Service = &api.Service{
		JSONBase: api.JSONBase{ID: "foo"},
	}
	obj, err := storage.Get("foo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if obj.(*api.Endpoints).Endpoints != nil {
		t.Errorf("unexpected endpoints: %#v", obj)
	}
}
