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

package service

import (
	"net"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest/resttest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service/ipallocator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service/portallocator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func NewTestREST(t *testing.T, endpoints *api.EndpointsList) (*REST, *registrytest.ServiceRegistry) {
	registry := registrytest.NewServiceRegistry()
	machines := []string{"foo", "bar", "baz"}
	endpointRegistry := &registrytest.EndpointRegistry{
		Endpoints: endpoints,
	}
	nodeRegistry := registrytest.NewMinionRegistry(machines, api.NodeResources{})
	r := ipallocator.NewCIDRRange(makeIPNet(t))

	portRange := util.PortRange{Base: 30000, Size: 1000}
	portAllocator := portallocator.NewPortAllocator(portRange)

	storage := NewStorage(registry, nodeRegistry, endpointRegistry, r, portAllocator, "kubernetes")

	return storage, registry
}

func makeIPNet(t *testing.T) *net.IPNet {
	_, net, err := net.ParseCIDR("1.2.3.0/24")
	if err != nil {
		t.Error(err)
	}
	return net
}

func deepCloneService(svc *api.Service) *api.Service {
	value, err := conversion.DeepCopy(svc)
	if err != nil {
		panic("couldn't copy service")
	}
	return value.(*api.Service)
}

func TestServiceRegistryCreate(t *testing.T) {
	storage, registry := NewTestREST(t, nil)

	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}},
		},
	}
	ctx := api.NewDefaultContext()
	created_svc, err := storage.Create(ctx, svc)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	created_service := created_svc.(*api.Service)
	if !api.HasObjectMetaSystemFieldValues(&created_service.ObjectMeta) {
		t.Errorf("storage did not populate object meta field values")
	}
	if created_service.Name != "foo" {
		t.Errorf("Expected foo, but got %v", created_service.Name)
	}
	if created_service.CreationTimestamp.IsZero() {
		t.Errorf("Expected timestamp to be set, got: %v", created_service.CreationTimestamp)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service.Spec.PortalIP)) {
		t.Errorf("Unexpected PortalIP: %s", created_service.Spec.PortalIP)
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
	storage, _ := NewTestREST(t, nil)
	failureCases := map[string]api.Service{
		"empty ID": {
			ObjectMeta: api.ObjectMeta{Name: ""},
			Spec: api.ServiceSpec{
				Selector:        map[string]string{"bar": "baz"},
				SessionAffinity: api.ServiceAffinityNone,
				Type:            api.ServiceTypeClusterIP,
				Ports: []api.ServicePort{{
					Port:     6502,
					Protocol: api.ProtocolTCP,
				}},
			},
		},
		"empty port": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.ServiceSpec{
				Selector:        map[string]string{"bar": "baz"},
				SessionAffinity: api.ServiceAffinityNone,
				Type:            api.ServiceTypeClusterIP,
				Ports: []api.ServicePort{{
					Protocol: api.ProtocolTCP,
				}},
			},
		},
	}
	ctx := api.NewDefaultContext()
	for _, failureCase := range failureCases {
		c, err := storage.Create(ctx, &failureCase)
		if c != nil {
			t.Errorf("Expected nil object")
		}
		if !errors.IsInvalid(err) {
			t.Errorf("Expected to get an invalid resource error, got %v", err)
		}
	}
}

func TestServiceRegistryUpdate(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, registry := NewTestREST(t, nil)
	svc, err := registry.CreateService(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1", Namespace: api.NamespaceDefault},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz1"},
			Ports: []api.ServicePort{{
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}},
		},
	})

	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	updated_svc, created, err := storage.Update(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: svc.ResourceVersion},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz2"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}},
		},
	})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if updated_svc == nil {
		t.Errorf("Expected non-nil object")
	}
	if created {
		t.Errorf("expected not created")
	}
	updated_service := updated_svc.(*api.Service)
	if updated_service.Name != "foo" {
		t.Errorf("Expected foo, but got %v", updated_service.Name)
	}
	if e, a := "foo", registry.UpdatedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceStorageValidatesUpdate(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, registry := NewTestREST(t, nil)
	registry.CreateService(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
			Ports: []api.ServicePort{{
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}},
		},
	})
	failureCases := map[string]api.Service{
		"empty ID": {
			ObjectMeta: api.ObjectMeta{Name: ""},
			Spec: api.ServiceSpec{
				Selector:        map[string]string{"bar": "baz"},
				SessionAffinity: api.ServiceAffinityNone,
				Type:            api.ServiceTypeClusterIP,
				Ports: []api.ServicePort{{
					Port:     6502,
					Protocol: api.ProtocolTCP,
				}},
			},
		},
		"invalid selector": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.ServiceSpec{
				Selector:        map[string]string{"ThisSelectorFailsValidation": "ok"},
				SessionAffinity: api.ServiceAffinityNone,
				Type:            api.ServiceTypeClusterIP,
				Ports: []api.ServicePort{{
					Port:     6502,
					Protocol: api.ProtocolTCP,
				}},
			},
		},
	}
	for _, failureCase := range failureCases {
		c, created, err := storage.Update(ctx, &failureCase)
		if c != nil || created {
			t.Errorf("Expected nil object or created false")
		}
		if !errors.IsInvalid(err) {
			t.Errorf("Expected to get an invalid resource error, got %v", err)
		}
	}
}

func TestServiceRegistryExternalService(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, registry := NewTestREST(t, nil)
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeLoadBalancer,
			Ports: []api.ServicePort{{
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}},
		},
	}
	_, err := storage.Create(ctx, svc)
	if err != nil {
		t.Errorf("Failed to create service: %#v", err)
	}
	srv, err := registry.GetService(ctx, svc.Name)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if srv == nil {
		t.Errorf("Failed to find service: %s", svc.Name)
	}
}

func TestServiceRegistryDelete(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, registry := NewTestREST(t, nil)
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}},
		},
	}
	registry.CreateService(ctx, svc)
	storage.Delete(ctx, svc.Name)
	if e, a := "foo", registry.DeletedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceRegistryDeleteExternal(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, registry := NewTestREST(t, nil)
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeLoadBalancer,
			Ports: []api.ServicePort{{
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}},
		},
	}
	registry.CreateService(ctx, svc)
	storage.Delete(ctx, svc.Name)
	if e, a := "foo", registry.DeletedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceRegistryUpdateExternalService(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, _ := NewTestREST(t, nil)

	// Create non-external load balancer.
	svc1 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}},
		},
	}
	if _, err := storage.Create(ctx, svc1); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Modify load balancer to be external.
	svc2 := deepCloneService(svc1)
	svc2.Spec.Type = api.ServiceTypeLoadBalancer
	if _, _, err := storage.Update(ctx, svc2); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Change port.
	svc3 := deepCloneService(svc2)
	svc3.Spec.Ports[0].Port = 6504
	if _, _, err := storage.Update(ctx, svc3); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
}

func TestServiceRegistryUpdateMultiPortExternalService(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, _ := NewTestREST(t, nil)

	// Create external load balancer.
	svc1 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeLoadBalancer,
			Ports: []api.ServicePort{{
				Name:     "p",
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}, {
				Name:     "q",
				Port:     8086,
				Protocol: api.ProtocolTCP,
			}},
		},
	}
	if _, err := storage.Create(ctx, svc1); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Modify ports
	svc2 := deepCloneService(svc1)
	svc2.Spec.Ports[1].Port = 8088
	if _, _, err := storage.Update(ctx, svc2); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
}

func TestServiceRegistryGet(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, registry := NewTestREST(t, nil)
	registry.CreateService(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
		},
	})
	storage.Get(ctx, "foo")
	if e, a := "foo", registry.GottenID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceRegistryResourceLocation(t *testing.T) {
	ctx := api.NewDefaultContext()
	endpoints := &api.EndpointsList{
		Items: []api.Endpoints{
			{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: api.NamespaceDefault,
				},
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Name: "", Port: 80}, {Name: "p", Port: 93}},
				}},
			},
		},
	}
	storage, registry := NewTestREST(t, endpoints)
	registry.CreateService(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
		},
	})
	redirector := rest.Redirector(storage)

	// Test a simple id.
	location, _, err := redirector.ResourceLocation(ctx, "foo")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if location == nil {
		t.Errorf("Unexpected nil: %v", location)
	}
	if e, a := "//1.2.3.4:80", location.String(); e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}

	// Test a name + port.
	location, _, err = redirector.ResourceLocation(ctx, "foo:p")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if location == nil {
		t.Errorf("Unexpected nil: %v", location)
	}
	if e, a := "//1.2.3.4:93", location.String(); e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}

	// Test a non-existent name + port.
	location, _, err = redirector.ResourceLocation(ctx, "foo:q")
	if err == nil {
		t.Errorf("Unexpected nil error")
	}

	// Test error path
	if _, _, err = redirector.ResourceLocation(ctx, "bar"); err == nil {
		t.Errorf("unexpected nil error")
	}
}

func TestServiceRegistryList(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, registry := NewTestREST(t, nil)
	registry.CreateService(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: api.NamespaceDefault},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
		},
	})
	registry.CreateService(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo2", Namespace: api.NamespaceDefault},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar2": "baz2"},
		},
	})
	registry.List.ResourceVersion = "1"
	s, _ := storage.List(ctx, labels.Everything(), fields.Everything())
	sl := s.(*api.ServiceList)
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
	rest, _ := NewTestREST(t, nil)

	svc1 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}},
		},
	}
	ctx := api.NewDefaultContext()
	created_svc1, _ := rest.Create(ctx, svc1)
	created_service_1 := created_svc1.(*api.Service)
	if created_service_1.Name != "foo" {
		t.Errorf("Expected foo, but got %v", created_service_1.Name)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service_1.Spec.PortalIP)) {
		t.Errorf("Unexpected PortalIP: %s", created_service_1.Spec.PortalIP)
	}

	svc2 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "bar"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}},
		}}
	ctx = api.NewDefaultContext()
	created_svc2, _ := rest.Create(ctx, svc2)
	created_service_2 := created_svc2.(*api.Service)
	if created_service_2.Name != "bar" {
		t.Errorf("Expected bar, but got %v", created_service_2.Name)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service_2.Spec.PortalIP)) {
		t.Errorf("Unexpected PortalIP: %s", created_service_2.Spec.PortalIP)
	}

	testIPs := []string{"1.2.3.93", "1.2.3.94", "1.2.3.95", "1.2.3.96"}
	testIP := ""
	for _, ip := range testIPs {
		if !rest.portals.(*ipallocator.Range).Has(net.ParseIP(ip)) {
			testIP = ip
		}
	}

	svc3 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "quux"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			PortalIP:        testIP,
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}},
		},
	}
	ctx = api.NewDefaultContext()
	created_svc3, err := rest.Create(ctx, svc3)
	if err != nil {
		t.Fatal(err)
	}
	created_service_3 := created_svc3.(*api.Service)
	if created_service_3.Spec.PortalIP != testIP { // specific IP
		t.Errorf("Unexpected PortalIP: %s", created_service_3.Spec.PortalIP)
	}
}

func TestServiceRegistryIPReallocation(t *testing.T) {
	rest, _ := NewTestREST(t, nil)

	svc1 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}},
		},
	}
	ctx := api.NewDefaultContext()
	created_svc1, _ := rest.Create(ctx, svc1)
	created_service_1 := created_svc1.(*api.Service)
	if created_service_1.Name != "foo" {
		t.Errorf("Expected foo, but got %v", created_service_1.Name)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service_1.Spec.PortalIP)) {
		t.Errorf("Unexpected PortalIP: %s", created_service_1.Spec.PortalIP)
	}

	_, err := rest.Delete(ctx, created_service_1.Name)
	if err != nil {
		t.Errorf("Unexpected error deleting service: %v", err)
	}

	svc2 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "bar"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}},
		},
	}
	ctx = api.NewDefaultContext()
	created_svc2, _ := rest.Create(ctx, svc2)
	created_service_2 := created_svc2.(*api.Service)
	if created_service_2.Name != "bar" {
		t.Errorf("Expected bar, but got %v", created_service_2.Name)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service_2.Spec.PortalIP)) {
		t.Errorf("Unexpected PortalIP: %s", created_service_2.Spec.PortalIP)
	}
}

func TestServiceRegistryIPUpdate(t *testing.T) {
	rest, _ := NewTestREST(t, nil)

	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}},
		},
	}
	ctx := api.NewDefaultContext()
	created_svc, _ := rest.Create(ctx, svc)
	created_service := created_svc.(*api.Service)
	if created_service.Spec.Ports[0].Port != 6502 {
		t.Errorf("Expected port 6502, but got %v", created_service.Spec.Ports[0].Port)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service.Spec.PortalIP)) {
		t.Errorf("Unexpected PortalIP: %s", created_service.Spec.PortalIP)
	}

	update := deepCloneService(created_service)
	update.Spec.Ports[0].Port = 6503

	updated_svc, _, _ := rest.Update(ctx, update)
	updated_service := updated_svc.(*api.Service)
	if updated_service.Spec.Ports[0].Port != 6503 {
		t.Errorf("Expected port 6503, but got %v", updated_service.Spec.Ports[0].Port)
	}

	update = deepCloneService(created_service)
	update.Spec.Ports[0].Port = 6503
	update.Spec.PortalIP = "1.2.3.76" // error

	_, _, err := rest.Update(ctx, update)
	if err == nil || !errors.IsInvalid(err) {
		t.Errorf("Unexpected error type: %v", err)
	}
}

func TestServiceRegistryIPLoadBalancer(t *testing.T) {
	rest, _ := NewTestREST(t, nil)

	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeLoadBalancer,
			Ports: []api.ServicePort{{
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}},
		},
	}
	ctx := api.NewDefaultContext()
	created_svc, _ := rest.Create(ctx, svc)
	created_service := created_svc.(*api.Service)
	if created_service.Spec.Ports[0].Port != 6502 {
		t.Errorf("Expected port 6502, but got %v", created_service.Spec.Ports[0].Port)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service.Spec.PortalIP)) {
		t.Errorf("Unexpected PortalIP: %s", created_service.Spec.PortalIP)
	}

	update := deepCloneService(created_service)

	_, _, err := rest.Update(ctx, update)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
}

// TODO: remove, covered by TestCreate
func TestCreateServiceWithConflictingNamespace(t *testing.T) {
	storage := REST{}
	service := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "not-default"},
	}

	ctx := api.NewDefaultContext()
	obj, err := storage.Create(ctx, service)
	if obj != nil {
		t.Error("Expected a nil object, but we got a value")
	}
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if strings.Contains(err.Error(), "Service.Namespace does not match the provided context") {
		t.Errorf("Expected 'Service.Namespace does not match the provided context' error, got '%s'", err.Error())
	}
}

func TestUpdateServiceWithConflictingNamespace(t *testing.T) {
	storage := REST{}
	service := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "not-default"},
	}

	ctx := api.NewDefaultContext()
	obj, created, err := storage.Update(ctx, service)
	if obj != nil || created {
		t.Error("Expected a nil object, but we got a value or created was true")
	}
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if strings.Index(err.Error(), "Service.Namespace does not match the provided context") == -1 {
		t.Errorf("Expected 'Service.Namespace does not match the provided context' error, got '%s'", err.Error())
	}
}

func TestCreate(t *testing.T) {
	rest, registry := NewTestREST(t, nil)

	test := resttest.New(t, rest, registry.SetError)
	test.TestCreate(
		// valid
		&api.Service{
			Spec: api.ServiceSpec{
				Selector:        map[string]string{"bar": "baz"},
				PortalIP:        "None",
				SessionAffinity: "None",
				Type:            api.ServiceTypeClusterIP,
				Ports: []api.ServicePort{{
					Port:     6502,
					Protocol: api.ProtocolTCP,
				}},
			},
		},
		// invalid
		&api.Service{
			Spec: api.ServiceSpec{},
		},
		// invalid
		&api.Service{
			Spec: api.ServiceSpec{
				Selector:        map[string]string{"bar": "baz"},
				PortalIP:        "invalid",
				SessionAffinity: "None",
				Type:            api.ServiceTypeClusterIP,
				Ports: []api.ServicePort{{
					Port:     6502,
					Protocol: api.ProtocolTCP,
				}},
			},
		},
	)
}
