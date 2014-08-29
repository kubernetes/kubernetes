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

package service

import (
	"fmt"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	cloud "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/fake"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/minion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestServiceRegistryCreate(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	storage := NewRegistryStorage(registry, fakeCloud, minion.NewRegistry(machines))
	svc := &api.Service{
		Port:     6502,
		JSONBase: api.JSONBase{ID: "foo"},
		Selector: map[string]string{"bar": "baz"},
	}
	c, _ := storage.Create(svc)
	created_svc := <-c
	created_service := created_svc.(*api.Service)
	if created_service.ID != "foo" {
		t.Errorf("Expected foo, but got %v", created_service.ID)
	}
	if created_service.CreationTimestamp.IsZero() {
		t.Errorf("Expected timestamp to be set, got %:v", created_service.CreationTimestamp)
	}
	if len(fakeCloud.Calls) != 0 {
		t.Errorf("Unexpected call(s): %#v", fakeCloud.Calls)
	}
	srv, err := registry.GetService(svc.ID)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if srv == nil {
		t.Errorf("Failed to find service: %s", svc.ID)
	}
}

func TestServiceStorageValidatesCreate(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	storage := NewRegistryStorage(registry, nil, nil)
	failureCases := map[string]api.Service{
		"empty ID": {
			Port:     6502,
			JSONBase: api.JSONBase{ID: ""},
			Selector: map[string]string{"bar": "baz"},
		},
		"empty selector": {
			JSONBase: api.JSONBase{ID: "foo"},
			Selector: map[string]string{},
		},
	}
	for _, failureCase := range failureCases {
		c, err := storage.Create(&failureCase)
		if c != nil {
			t.Errorf("Expected nil channel")
		}
		if !apiserver.IsInvalid(err) {
			t.Errorf("Expected to get an invalid resource error, got %v", err)
		}

	}
}

func TestServiceRegistryUpdate(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	registry.CreateService(api.Service{
		Port:     6502,
		JSONBase: api.JSONBase{ID: "foo"},
		Selector: map[string]string{"bar": "baz1"},
	})
	storage := NewRegistryStorage(registry, nil, nil)
	c, err := storage.Update(&api.Service{
		Port:     6502,
		JSONBase: api.JSONBase{ID: "foo"},
		Selector: map[string]string{"bar": "baz2"},
	})
	if c == nil {
		t.Errorf("Expected non-nil channel")
	}
	if err != nil {
		t.Errorf("Expected no error")
	}
	updated_svc := <-c
	updated_service := updated_svc.(*api.Service)
	if updated_service.ID != "foo" {
		t.Errorf("Expected foo, but got %v", updated_service.ID)
	}
	if e, a := "foo", registry.UpdatedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceStorageValidatesUpdate(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	registry.CreateService(api.Service{
		Port:     6502,
		JSONBase: api.JSONBase{ID: "foo"},
		Selector: map[string]string{"bar": "baz"},
	})
	storage := NewRegistryStorage(registry, nil, nil)
	failureCases := map[string]api.Service{
		"empty ID": {
			Port:     6502,
			JSONBase: api.JSONBase{ID: ""},
			Selector: map[string]string{"bar": "baz"},
		},
		"empty selector": {
			Port:     6502,
			JSONBase: api.JSONBase{ID: "foo"},
			Selector: map[string]string{},
		},
	}
	for _, failureCase := range failureCases {
		c, err := storage.Update(&failureCase)
		if c != nil {
			t.Errorf("Expected nil channel")
		}
		if !apiserver.IsInvalid(err) {
			t.Errorf("Expected to get an invalid resource error, got %v", err)
		}
	}
}

func TestServiceRegistryExternalService(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	storage := NewRegistryStorage(registry, fakeCloud, minion.NewRegistry(machines))
	svc := &api.Service{
		Port:                       6502,
		JSONBase:                   api.JSONBase{ID: "foo"},
		Selector:                   map[string]string{"bar": "baz"},
		CreateExternalLoadBalancer: true,
	}
	c, _ := storage.Create(svc)
	<-c
	if len(fakeCloud.Calls) != 2 || fakeCloud.Calls[0] != "get-zone" || fakeCloud.Calls[1] != "create" {
		t.Errorf("Unexpected call(s): %#v", fakeCloud.Calls)
	}
	srv, err := registry.GetService(svc.ID)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if srv == nil {
		t.Errorf("Failed to find service: %s", svc.ID)
	}
}

func TestServiceRegistryExternalServiceError(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{
		Err: fmt.Errorf("test error"),
	}
	machines := []string{"foo", "bar", "baz"}
	storage := NewRegistryStorage(registry, fakeCloud, minion.NewRegistry(machines))
	svc := &api.Service{
		Port:                       6502,
		JSONBase:                   api.JSONBase{ID: "foo"},
		Selector:                   map[string]string{"bar": "baz"},
		CreateExternalLoadBalancer: true,
	}
	c, _ := storage.Create(svc)
	<-c
	if len(fakeCloud.Calls) != 1 || fakeCloud.Calls[0] != "get-zone" {
		t.Errorf("Unexpected call(s): %#v", fakeCloud.Calls)
	}
	if registry.Service != nil {
		t.Errorf("Expected registry.CreateService to not get called, but it got %#v", registry.Service)
	}
}

func TestServiceRegistryDelete(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	storage := NewRegistryStorage(registry, fakeCloud, minion.NewRegistry(machines))
	svc := api.Service{
		JSONBase: api.JSONBase{ID: "foo"},
		Selector: map[string]string{"bar": "baz"},
	}
	registry.CreateService(svc)
	c, _ := storage.Delete(svc.ID)
	<-c
	if len(fakeCloud.Calls) != 0 {
		t.Errorf("Unexpected call(s): %#v", fakeCloud.Calls)
	}
	if e, a := "foo", registry.DeletedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceRegistryDeleteExternal(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	storage := NewRegistryStorage(registry, fakeCloud, minion.NewRegistry(machines))
	svc := api.Service{
		JSONBase:                   api.JSONBase{ID: "foo"},
		Selector:                   map[string]string{"bar": "baz"},
		CreateExternalLoadBalancer: true,
	}
	registry.CreateService(svc)
	c, _ := storage.Delete(svc.ID)
	<-c
	if len(fakeCloud.Calls) != 2 || fakeCloud.Calls[0] != "get-zone" || fakeCloud.Calls[1] != "delete" {
		t.Errorf("Unexpected call(s): %#v", fakeCloud.Calls)
	}
	if e, a := "foo", registry.DeletedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceRegistryMakeLinkVariables(t *testing.T) {
	service := api.Service{
		JSONBase:      api.JSONBase{ID: "foo-bar"},
		Selector:      map[string]string{"bar": "baz"},
		ContainerPort: util.IntOrString{Kind: util.IntstrString, StrVal: "a-b-c"},
	}
	registry := registrytest.NewServiceRegistry()
	registry.List = api.ServiceList{
		Items: []api.Service{service},
	}
	machine := "machine"
	vars, err := GetServiceEnvironmentVariables(registry, machine)
	if err != nil {
		t.Errorf("Unexpected err: %v", err)
	}
	for _, v := range vars {
		if !util.IsCIdentifier(v.Name) {
			t.Errorf("Environment variable name is not valid: %v", v.Name)
		}
	}
}

func TestServiceRegistryGet(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	storage := NewRegistryStorage(registry, fakeCloud, minion.NewRegistry(machines))
	registry.CreateService(api.Service{
		JSONBase: api.JSONBase{ID: "foo"},
		Selector: map[string]string{"bar": "baz"},
	})
	storage.Get("foo")
	if len(fakeCloud.Calls) != 0 {
		t.Errorf("Unexpected call(s): %#v", fakeCloud.Calls)
	}
	if e, a := "foo", registry.GottenID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceRegistryResourceLocation(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	registry.Endpoints = api.Endpoints{Endpoints: []string{"foo:80"}}
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	storage := NewRegistryStorage(registry, fakeCloud, minion.NewRegistry(machines))
	registry.CreateService(api.Service{
		JSONBase: api.JSONBase{ID: "foo"},
		Selector: map[string]string{"bar": "baz"},
	})
	redirector := storage.(apiserver.Redirector)
	location, err := redirector.ResourceLocation("foo")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := "foo:80", location; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
	if e, a := "foo", registry.GottenID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}

	// Test error path
	registry.Err = fmt.Errorf("fake error")
	if _, err = redirector.ResourceLocation("foo"); err == nil {
		t.Errorf("unexpected nil error")
	}
}

func TestServiceRegistryList(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	storage := NewRegistryStorage(registry, fakeCloud, minion.NewRegistry(machines))
	registry.CreateService(api.Service{
		JSONBase: api.JSONBase{ID: "foo"},
		Selector: map[string]string{"bar": "baz"},
	})
	registry.CreateService(api.Service{
		JSONBase: api.JSONBase{ID: "foo2"},
		Selector: map[string]string{"bar2": "baz2"},
	})
	registry.List.ResourceVersion = 1
	s, _ := storage.List(labels.Everything())
	sl := s.(*api.ServiceList)
	if len(fakeCloud.Calls) != 0 {
		t.Errorf("Unexpected call(s): %#v", fakeCloud.Calls)
	}
	if len(sl.Items) != 2 {
		t.Fatalf("Expected 2 services, but got %v", len(sl.Items))
	}
	if e, a := "foo", sl.Items[0].ID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
	if e, a := "foo2", sl.Items[1].ID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
	if sl.ResourceVersion != 1 {
		t.Errorf("Unexpected resource version: %#v", sl)
	}
}
