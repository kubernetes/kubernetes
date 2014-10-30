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
	"net"
	"reflect"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	cloud "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/fake"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
)

func makeIPNet(t *testing.T) *net.IPNet {
	_, net, err := net.ParseCIDR("1.2.3.0/24")
	if err != nil {
		t.Error(err)
	}
	return net
}

func TestServiceRegistryCreate(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	storage := NewREST(registry, fakeCloud, registrytest.NewMinionRegistry(machines, api.NodeResources{}), makeIPNet(t))
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Port:     6502,
			Selector: map[string]string{"bar": "baz"},
		},
	}
	ctx := api.NewDefaultContext()
	c, _ := storage.Create(ctx, svc)
	created_svc := <-c
	created_service := created_svc.Object.(*api.Service)
	if created_service.Name != "foo" {
		t.Errorf("Expected foo, but got %v", created_service.Name)
	}
	if created_service.CreationTimestamp.IsZero() {
		t.Errorf("Expected timestamp to be set, got: %v", created_service.CreationTimestamp)
	}
	if created_service.Spec.PortalIP != "1.2.3.1" {
		t.Errorf("Unexpected PortalIP: %s", created_service.Spec.PortalIP)
	}
	if created_service.Spec.ProxyPort != 0 {
		t.Errorf("Unexpected ProxyPort: %d", created_service.Spec.ProxyPort)
	}
	if len(fakeCloud.Calls) != 0 {
		t.Errorf("Unexpected call(s): %#v", fakeCloud.Calls)
	}
	srv, err := registry.GetService(ctx, svc.Name)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if srv == nil {
		t.Errorf("Failed to find service: %s", svc.Name)
	}
}

func TestServiceStorageValidatesCreate(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	storage := NewREST(registry, nil, nil, makeIPNet(t))
	failureCases := map[string]api.Service{
		"empty ID": {
			ObjectMeta: api.ObjectMeta{Name: ""},
			Spec: api.ServiceSpec{
				Port:     6502,
				Selector: map[string]string{"bar": "baz"},
			},
		},
		"empty selector": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.ServiceSpec{
				Selector: map[string]string{"bar": "baz"},
			},
		},
	}
	ctx := api.NewDefaultContext()
	for _, failureCase := range failureCases {
		c, err := storage.Create(ctx, &failureCase)
		if c != nil {
			t.Errorf("Expected nil channel")
		}
		if !errors.IsInvalid(err) {
			t.Errorf("Expected to get an invalid resource error, got %v", err)
		}

	}
}

func TestServiceRegistryUpdate(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry := registrytest.NewServiceRegistry()
	registry.CreateService(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Port:     6502,
			Selector: map[string]string{"bar": "baz1"},
		},
	})
	storage := NewREST(registry, nil, nil, makeIPNet(t))
	c, err := storage.Update(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Port:     6502,
			Selector: map[string]string{"bar": "baz2"},
		},
	})
	if c == nil {
		t.Errorf("Expected non-nil channel")
	}
	if err != nil {
		t.Errorf("Expected no error")
	}
	updated_svc := <-c
	updated_service := updated_svc.Object.(*api.Service)
	if updated_service.Name != "foo" {
		t.Errorf("Expected foo, but got %v", updated_service.Name)
	}
	if e, a := "foo", registry.UpdatedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceStorageValidatesUpdate(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry := registrytest.NewServiceRegistry()
	registry.CreateService(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Port:     6502,
			Selector: map[string]string{"bar": "baz"},
		},
	})
	storage := NewREST(registry, nil, nil, makeIPNet(t))
	failureCases := map[string]api.Service{
		"empty ID": {
			ObjectMeta: api.ObjectMeta{Name: ""},
			Spec: api.ServiceSpec{
				Port:     6502,
				Selector: map[string]string{"bar": "baz"},
			},
		},
		"empty selector": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.ServiceSpec{
				Port:     6502,
				Selector: map[string]string{},
			},
		},
	}
	for _, failureCase := range failureCases {
		c, err := storage.Update(ctx, &failureCase)
		if c != nil {
			t.Errorf("Expected nil channel")
		}
		if !errors.IsInvalid(err) {
			t.Errorf("Expected to get an invalid resource error, got %v", err)
		}
	}
}

func TestServiceRegistryExternalService(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	storage := NewREST(registry, fakeCloud, registrytest.NewMinionRegistry(machines, api.NodeResources{}), makeIPNet(t))
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Port:                       6502,
			Selector:                   map[string]string{"bar": "baz"},
			CreateExternalLoadBalancer: true,
		},
	}
	c, _ := storage.Create(ctx, svc)
	<-c
	if len(fakeCloud.Calls) != 2 || fakeCloud.Calls[0] != "get-zone" || fakeCloud.Calls[1] != "create" {
		t.Errorf("Unexpected call(s): %#v", fakeCloud.Calls)
	}
	srv, err := registry.GetService(ctx, svc.Name)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if srv == nil {
		t.Errorf("Failed to find service: %s", svc.Name)
	}
}

func TestServiceRegistryExternalServiceError(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{
		Err: fmt.Errorf("test error"),
	}
	machines := []string{"foo", "bar", "baz"}
	storage := NewREST(registry, fakeCloud, registrytest.NewMinionRegistry(machines, api.NodeResources{}), makeIPNet(t))
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Port:                       6502,
			Selector:                   map[string]string{"bar": "baz"},
			CreateExternalLoadBalancer: true,
		},
	}
	ctx := api.NewDefaultContext()
	c, _ := storage.Create(ctx, svc)
	<-c
	if len(fakeCloud.Calls) != 1 || fakeCloud.Calls[0] != "get-zone" {
		t.Errorf("Unexpected call(s): %#v", fakeCloud.Calls)
	}
	if registry.Service != nil {
		t.Errorf("Expected registry.CreateService to not get called, but it got %#v", registry.Service)
	}
}

func TestServiceRegistryDelete(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	storage := NewREST(registry, fakeCloud, registrytest.NewMinionRegistry(machines, api.NodeResources{}), makeIPNet(t))
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
		},
	}
	registry.CreateService(ctx, svc)
	c, _ := storage.Delete(ctx, svc.Name)
	<-c
	if len(fakeCloud.Calls) != 0 {
		t.Errorf("Unexpected call(s): %#v", fakeCloud.Calls)
	}
	if e, a := "foo", registry.DeletedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceRegistryDeleteExternal(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	storage := NewREST(registry, fakeCloud, registrytest.NewMinionRegistry(machines, api.NodeResources{}), makeIPNet(t))
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector:                   map[string]string{"bar": "baz"},
			CreateExternalLoadBalancer: true,
		},
	}
	registry.CreateService(ctx, svc)
	c, _ := storage.Delete(ctx, svc.Name)
	<-c
	if len(fakeCloud.Calls) != 2 || fakeCloud.Calls[0] != "get-zone" || fakeCloud.Calls[1] != "delete" {
		t.Errorf("Unexpected call(s): %#v", fakeCloud.Calls)
	}
	if e, a := "foo", registry.DeletedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceRegistryMakeLinkVariables(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry := registrytest.NewServiceRegistry()
	registry.List = api.ServiceList{
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo-bar"},
				Spec: api.ServiceSpec{
					Port:     8080,
					Selector: map[string]string{"bar": "baz"},
					Protocol: "TCP",
					PortalIP: "1.2.3.4",
				},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "abc-123"},
				Spec: api.ServiceSpec{
					Port:     8081,
					Selector: map[string]string{"bar": "baz"},
					Protocol: "UDP",
					PortalIP: "5.6.7.8",
				},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "q-u-u-x"},
				Spec: api.ServiceSpec{
					Port:     8082,
					Selector: map[string]string{"bar": "baz"},
					Protocol: "TCP",
					PortalIP: "9.8.7.6",
				},
			},
		},
	}
	machine := "machine"
	vars, err := GetServiceEnvironmentVariables(ctx, registry, machine)
	if err != nil {
		t.Errorf("Unexpected err: %v", err)
	}
	expected := []api.EnvVar{
		{Name: "FOO_BAR_SERVICE_HOST", Value: "1.2.3.4"},
		{Name: "FOO_BAR_SERVICE_PORT", Value: "8080"},
		{Name: "FOO_BAR_PORT", Value: "tcp://1.2.3.4:8080"},
		{Name: "FOO_BAR_PORT_8080_TCP", Value: "tcp://1.2.3.4:8080"},
		{Name: "FOO_BAR_PORT_8080_TCP_PROTO", Value: "tcp"},
		{Name: "FOO_BAR_PORT_8080_TCP_PORT", Value: "8080"},
		{Name: "FOO_BAR_PORT_8080_TCP_ADDR", Value: "1.2.3.4"},
		{Name: "ABC_123_SERVICE_HOST", Value: "5.6.7.8"},
		{Name: "ABC_123_SERVICE_PORT", Value: "8081"},
		{Name: "ABC_123_PORT", Value: "udp://5.6.7.8:8081"},
		{Name: "ABC_123_PORT_8081_UDP", Value: "udp://5.6.7.8:8081"},
		{Name: "ABC_123_PORT_8081_UDP_PROTO", Value: "udp"},
		{Name: "ABC_123_PORT_8081_UDP_PORT", Value: "8081"},
		{Name: "ABC_123_PORT_8081_UDP_ADDR", Value: "5.6.7.8"},
		{Name: "Q_U_U_X_SERVICE_HOST", Value: "9.8.7.6"},
		{Name: "Q_U_U_X_SERVICE_PORT", Value: "8082"},
		{Name: "Q_U_U_X_PORT", Value: "tcp://9.8.7.6:8082"},
		{Name: "Q_U_U_X_PORT_8082_TCP", Value: "tcp://9.8.7.6:8082"},
		{Name: "Q_U_U_X_PORT_8082_TCP_PROTO", Value: "tcp"},
		{Name: "Q_U_U_X_PORT_8082_TCP_PORT", Value: "8082"},
		{Name: "Q_U_U_X_PORT_8082_TCP_ADDR", Value: "9.8.7.6"},
	}
	if len(vars) != len(expected) {
		t.Errorf("Expected %d env vars, got: %+v", len(expected), vars)
		return
	}
	for i := range expected {
		if !reflect.DeepEqual(vars[i], expected[i]) {
			t.Errorf("expected %#v, got %#v", vars[i], expected[i])
		}
	}
}

func TestServiceRegistryGet(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	storage := NewREST(registry, fakeCloud, registrytest.NewMinionRegistry(machines, api.NodeResources{}), makeIPNet(t))
	registry.CreateService(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
		},
	})
	storage.Get(ctx, "foo")
	if len(fakeCloud.Calls) != 0 {
		t.Errorf("Unexpected call(s): %#v", fakeCloud.Calls)
	}
	if e, a := "foo", registry.GottenID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceRegistryResourceLocation(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry := registrytest.NewServiceRegistry()
	registry.Endpoints = api.Endpoints{Endpoints: []string{"foo:80"}}
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	storage := NewREST(registry, fakeCloud, registrytest.NewMinionRegistry(machines, api.NodeResources{}), makeIPNet(t))
	registry.CreateService(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
		},
	})
	redirector := apiserver.Redirector(storage)
	location, err := redirector.ResourceLocation(ctx, "foo")
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
	if _, err = redirector.ResourceLocation(ctx, "foo"); err == nil {
		t.Errorf("unexpected nil error")
	}
}

func TestServiceRegistryList(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	storage := NewREST(registry, fakeCloud, registrytest.NewMinionRegistry(machines, api.NodeResources{}), makeIPNet(t))
	registry.CreateService(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
		},
	})
	registry.CreateService(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo2"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar2": "baz2"},
		},
	})
	registry.List.ResourceVersion = "1"
	s, _ := storage.List(ctx, labels.Everything(), labels.Everything())
	sl := s.(*api.ServiceList)
	if len(fakeCloud.Calls) != 0 {
		t.Errorf("Unexpected call(s): %#v", fakeCloud.Calls)
	}
	if len(sl.Items) != 2 {
		t.Fatalf("Expected 2 services, but got %v", len(sl.Items))
	}
	if e, a := "foo", sl.Items[0].Name; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
	if e, a := "foo2", sl.Items[1].Name; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
	if sl.ResourceVersion != "1" {
		t.Errorf("Unexpected resource version: %#v", sl)
	}
}

func TestServiceRegistryIPAllocation(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	rest := NewREST(registry, fakeCloud, registrytest.NewMinionRegistry(machines, api.NodeResources{}), makeIPNet(t))

	svc1 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
			Port:     6502,
		},
	}
	ctx := api.NewDefaultContext()
	c1, _ := rest.Create(ctx, svc1)
	created_svc1 := <-c1
	created_service_1 := created_svc1.Object.(*api.Service)
	if created_service_1.Name != "foo" {
		t.Errorf("Expected foo, but got %v", created_service_1.Name)
	}
	if created_service_1.Spec.PortalIP != "1.2.3.1" {
		t.Errorf("Unexpected PortalIP: %s", created_service_1.Spec.PortalIP)
	}

	svc2 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "bar"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
			Port:     6502,
		}}
	ctx = api.NewDefaultContext()
	c2, _ := rest.Create(ctx, svc2)
	created_svc2 := <-c2
	created_service_2 := created_svc2.Object.(*api.Service)
	if created_service_2.Name != "bar" {
		t.Errorf("Expected bar, but got %v", created_service_2.Name)
	}
	if created_service_2.Spec.PortalIP != "1.2.3.2" { // new IP
		t.Errorf("Unexpected PortalIP: %s", created_service_2.Spec.PortalIP)
	}

	svc3 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "quux"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
			PortalIP: "1.2.3.93",
			Port:     6502,
		},
	}
	ctx = api.NewDefaultContext()
	c3, _ := rest.Create(ctx, svc3)
	created_svc3 := <-c3
	created_service_3 := created_svc3.Object.(*api.Service)
	if created_service_3.Spec.PortalIP != "1.2.3.93" { // specific IP
		t.Errorf("Unexpected PortalIP: %s", created_service_3.Spec.PortalIP)
	}
}

func TestServiceRegistryIPReallocation(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	rest := NewREST(registry, fakeCloud, registrytest.NewMinionRegistry(machines, api.NodeResources{}), makeIPNet(t))

	svc1 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
			Port:     6502,
		},
	}
	ctx := api.NewDefaultContext()
	c1, _ := rest.Create(ctx, svc1)
	created_svc1 := <-c1
	created_service_1 := created_svc1.Object.(*api.Service)
	if created_service_1.Name != "foo" {
		t.Errorf("Expected foo, but got %v", created_service_1.Name)
	}
	if created_service_1.Spec.PortalIP != "1.2.3.1" {
		t.Errorf("Unexpected PortalIP: %s", created_service_1.Spec.PortalIP)
	}

	c, _ := rest.Delete(ctx, created_service_1.Name)
	<-c

	svc2 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "bar"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
			Port:     6502,
		},
	}
	ctx = api.NewDefaultContext()
	c2, _ := rest.Create(ctx, svc2)
	created_svc2 := <-c2
	created_service_2 := created_svc2.Object.(*api.Service)
	if created_service_2.Name != "bar" {
		t.Errorf("Expected bar, but got %v", created_service_2.Name)
	}
	if created_service_2.Spec.PortalIP != "1.2.3.1" { // same IP as before
		t.Errorf("Unexpected PortalIP: %s", created_service_2.Spec.PortalIP)
	}
}

func TestServiceRegistryIPUpdate(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	rest := NewREST(registry, fakeCloud, registrytest.NewMinionRegistry(machines, api.NodeResources{}), makeIPNet(t))

	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
			Port:     6502,
		},
	}
	ctx := api.NewDefaultContext()
	c, _ := rest.Create(ctx, svc)
	created_svc := <-c
	created_service := created_svc.Object.(*api.Service)
	if created_service.Spec.Port != 6502 {
		t.Errorf("Expected port 6502, but got %v", created_service.Spec.Port)
	}
	if created_service.Spec.PortalIP != "1.2.3.1" {
		t.Errorf("Unexpected PortalIP: %s", created_service.Spec.PortalIP)
	}
	if created_service.Spec.ProxyPort != 0 {
		t.Errorf("Unexpected ProxyPort: %d", created_service.Spec.ProxyPort)
	}

	update := new(api.Service)
	*update = *created_service
	update.Spec.Port = 6503
	update.Spec.ProxyPort = 309 // should be ignored

	c, _ = rest.Update(ctx, update)
	updated_svc := <-c
	updated_service := updated_svc.Object.(*api.Service)
	if updated_service.Spec.Port != 6503 {
		t.Errorf("Expected port 6503, but got %v", updated_service.Spec.Port)
	}
	if updated_service.Spec.ProxyPort != 0 { // unchanged, despite trying
		t.Errorf("Unexpected ProxyPort: %d", updated_service.Spec.ProxyPort)
	}

	*update = *created_service
	update.Spec.Port = 6503
	update.Spec.PortalIP = "1.2.3.76" // error

	c, _ = rest.Update(ctx, update)
	result := <-c
	st := result.Object.(*api.Status)
	if st.Reason != api.StatusReasonInvalid {
		t.Errorf("Expected to get an invalid error, got %v", st)
	}
}

func TestServiceRegistryIPExternalLoadBalancer(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	rest := NewREST(registry, fakeCloud, registrytest.NewMinionRegistry(machines, api.NodeResources{}), makeIPNet(t))

	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
			Port:     6502,
			CreateExternalLoadBalancer: true,
		},
	}
	ctx := api.NewDefaultContext()
	c, _ := rest.Create(ctx, svc)
	created_svc := <-c
	created_service := created_svc.Object.(*api.Service)
	if created_service.Spec.Port != 6502 {
		t.Errorf("Expected port 6502, but got %v", created_service.Spec.Port)
	}
	if created_service.Spec.PortalIP != "1.2.3.1" {
		t.Errorf("Unexpected PortalIP: %s", created_service.Spec.PortalIP)
	}
	if created_service.Spec.ProxyPort != 6502 {
		t.Errorf("Unexpected ProxyPort: %d", created_service.Spec.ProxyPort)
	}
}

func TestServiceRegistryIPReloadFromStorage(t *testing.T) {
	registry := registrytest.NewServiceRegistry()
	fakeCloud := &cloud.FakeCloud{}
	machines := []string{"foo", "bar", "baz"}
	rest1 := NewREST(registry, fakeCloud, registrytest.NewMinionRegistry(machines, api.NodeResources{}), makeIPNet(t))

	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
			Port:     6502,
		},
	}
	ctx := api.NewDefaultContext()
	c, _ := rest1.Create(ctx, svc)
	<-c
	svc = &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
			Port:     6502,
		},
	}
	c, _ = rest1.Create(ctx, svc)
	<-c

	// This will reload from storage, finding the previous 2
	rest2 := NewREST(registry, fakeCloud, registrytest.NewMinionRegistry(machines, api.NodeResources{}), makeIPNet(t))

	svc = &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
			Port:     6502,
		},
	}
	c, _ = rest2.Create(ctx, svc)
	created_svc := <-c
	created_service := created_svc.Object.(*api.Service)
	if created_service.Spec.PortalIP != "1.2.3.3" {
		t.Errorf("Unexpected PortalIP: %s", created_service.Spec.PortalIP)
	}
}

func TestCreateServiceWithConflictingNamespace(t *testing.T) {
	storage := REST{}
	service := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "not-default"},
	}

	ctx := api.NewDefaultContext()
	channel, err := storage.Create(ctx, service)
	if channel != nil {
		t.Error("Expected a nil channel, but we got a value")
	}
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if strings.Index(err.Error(), "Service.Namespace does not match the provided context") == -1 {
		t.Errorf("Expected 'Service.Namespace does not match the provided context' error, got '%v'", err.Error())
	}
}

func TestUpdateServiceWithConflictingNamespace(t *testing.T) {
	storage := REST{}
	service := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "not-default"},
	}

	ctx := api.NewDefaultContext()
	channel, err := storage.Update(ctx, service)
	if channel != nil {
		t.Error("Expected a nil channel, but we got a value")
	}
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if strings.Index(err.Error(), "Service.Namespace does not match the provided context") == -1 {
		t.Errorf("Expected 'Service.Namespace does not match the provided context' error, got '%v'", err.Error())
	}
}
