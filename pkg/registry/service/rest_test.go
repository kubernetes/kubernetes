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

package service

import (
	"net"
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/service"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/registry/service/ipallocator"
	"k8s.io/kubernetes/pkg/registry/service/portallocator"
	featuregate "k8s.io/kubernetes/pkg/util/config"
	"k8s.io/kubernetes/pkg/util/intstr"
	utilnet "k8s.io/kubernetes/pkg/util/net"
)

func init() {
	featuregate.DefaultFeatureGate.Set("AllowExtTrafficLocalEndpoints=true")
}

// TODO(wojtek-t): Cleanup this file.
// It is now testing mostly the same things as other resources but
// in a completely different way. We should unify it.

func NewTestREST(t *testing.T, endpoints *api.EndpointsList) (*REST, *registrytest.ServiceRegistry) {
	registry := registrytest.NewServiceRegistry()
	endpointRegistry := &registrytest.EndpointRegistry{
		Endpoints: endpoints,
	}
	r := ipallocator.NewCIDRRange(makeIPNet(t))

	portRange := utilnet.PortRange{Base: 30000, Size: 1000}
	portAllocator := portallocator.NewPortAllocator(portRange)

	storage := NewStorage(registry, endpointRegistry, r, portAllocator, nil)

	return storage.Service, registry
}

func makeIPNet(t *testing.T) *net.IPNet {
	_, net, err := net.ParseCIDR("1.2.3.0/24")
	if err != nil {
		t.Error(err)
	}
	return net
}

func deepCloneService(svc *api.Service) *api.Service {
	value, err := api.Scheme.DeepCopy(svc)
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
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
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
	if !makeIPNet(t).Contains(net.ParseIP(created_service.Spec.ClusterIP)) {
		t.Errorf("Unexpected ClusterIP: %s", created_service.Spec.ClusterIP)
	}
	srv, err := registry.GetService(ctx, svc.Name)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if srv == nil {
		t.Errorf("Failed to find service: %s", svc.Name)
	}
}

func TestServiceRegistryCreateMultiNodePortsService(t *testing.T) {
	storage, registry := NewTestREST(t, nil)
	testCases := []struct {
		svc             *api.Service
		name            string
		expectNodePorts []int
	}{
		{
			svc: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "foo1"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Name:       "port-tcp",
							Port:       53,
							NodePort:   30053,
							TargetPort: intstr.FromInt(6503),
							Protocol:   api.ProtocolTCP,
						},
						{
							Name:       "port-udp",
							Port:       53,
							NodePort:   30053,
							TargetPort: intstr.FromInt(6503),
							Protocol:   api.ProtocolUDP,
						},
					},
				},
			},
			name:            "foo1",
			expectNodePorts: []int{30053, 30053},
		},
		{
			svc: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "foo2"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Name:       "port-tcp",
							Port:       54,
							TargetPort: intstr.FromInt(6504),
							Protocol:   api.ProtocolTCP,
						},
						{
							Name:       "port-udp",
							Port:       54,
							NodePort:   30054,
							TargetPort: intstr.FromInt(6504),
							Protocol:   api.ProtocolUDP,
						},
					},
				},
			},
			name:            "foo2",
			expectNodePorts: []int{30054, 30054},
		},
		{
			svc: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "foo3"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Name:       "port-tcp",
							Port:       55,
							NodePort:   30055,
							TargetPort: intstr.FromInt(6505),
							Protocol:   api.ProtocolTCP,
						},
						{
							Name:       "port-udp",
							Port:       55,
							NodePort:   30056,
							TargetPort: intstr.FromInt(6506),
							Protocol:   api.ProtocolUDP,
						},
					},
				},
			},
			name:            "foo3",
			expectNodePorts: []int{30055, 30056},
		},
	}

	ctx := api.NewDefaultContext()
	for _, test := range testCases {
		created_svc, err := storage.Create(ctx, test.svc)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		created_service := created_svc.(*api.Service)
		if !api.HasObjectMetaSystemFieldValues(&created_service.ObjectMeta) {
			t.Errorf("storage did not populate object meta field values")
		}
		if created_service.Name != test.name {
			t.Errorf("Expected %s, but got %s", test.name, created_service.Name)
		}
		serviceNodePorts := CollectServiceNodePorts(created_service)
		if !reflect.DeepEqual(serviceNodePorts, test.expectNodePorts) {
			t.Errorf("Expected %v, but got %v", test.expectNodePorts, serviceNodePorts)
		}
		srv, err := registry.GetService(ctx, test.name)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if srv == nil {
			t.Errorf("Failed to find service: %s", test.name)
		}
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
					Port:       6502,
					Protocol:   api.ProtocolTCP,
					TargetPort: intstr.FromInt(6502),
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
		"missing targetPort": {
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
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	})

	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	updated_svc, created, err := storage.Update(ctx, "foo", rest.DefaultUpdatedObjectInfo(&api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: svc.ResourceVersion},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz2"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}, api.Scheme))
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
					Port:       6502,
					Protocol:   api.ProtocolTCP,
					TargetPort: intstr.FromInt(6502),
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
					Port:       6502,
					Protocol:   api.ProtocolTCP,
					TargetPort: intstr.FromInt(6502),
				}},
			},
		},
	}
	for _, failureCase := range failureCases {
		c, created, err := storage.Update(ctx, failureCase.Name, rest.DefaultUpdatedObjectInfo(&failureCase, api.Scheme))
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
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
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
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}
	if _, err := storage.Create(ctx, svc1); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Modify load balancer to be external.
	svc2 := deepCloneService(svc1)
	svc2.Spec.Type = api.ServiceTypeLoadBalancer
	if _, _, err := storage.Update(ctx, svc2.Name, rest.DefaultUpdatedObjectInfo(svc2, api.Scheme)); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Change port.
	svc3 := deepCloneService(svc2)
	svc3.Spec.Ports[0].Port = 6504
	if _, _, err := storage.Update(ctx, svc3.Name, rest.DefaultUpdatedObjectInfo(svc3, api.Scheme)); err != nil {
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
				Name:       "p",
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}, {
				Name:       "q",
				Port:       8086,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(8086),
			}},
		},
	}
	if _, err := storage.Create(ctx, svc1); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Modify ports
	svc2 := deepCloneService(svc1)
	svc2.Spec.Ports[1].Port = 8088
	if _, _, err := storage.Update(ctx, svc2.Name, rest.DefaultUpdatedObjectInfo(svc2, api.Scheme)); err != nil {
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
			{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: api.NamespaceDefault,
				},
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{},
					Ports:     []api.EndpointPort{{Name: "", Port: 80}, {Name: "p", Port: 93}},
				}, {
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Name: "", Port: 80}, {Name: "p", Port: 93}},
				}, {
					Addresses: []api.EndpointAddress{{IP: "1.2.3.5"}},
					Ports:     []api.EndpointPort{},
				}},
			},
		},
	}
	storage, registry := NewTestREST(t, endpoints)
	registry.CreateService(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
			Ports: []api.ServicePort{
				// Service port 9393 should route to endpoint port "p", which is port 93
				{Name: "p", Port: 9393, TargetPort: intstr.FromString("p")},

				// Service port 93 should route to unnamed endpoint port, which is port 80
				// This is to test that the service port definition is used when determining resource location
				{Name: "", Port: 93, TargetPort: intstr.FromInt(80)},
			},
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

	// Test a name + port number (service port 93 -> target port 80)
	location, _, err = redirector.ResourceLocation(ctx, "foo:93")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if location == nil {
		t.Errorf("Unexpected nil: %v", location)
	}
	if e, a := "//1.2.3.4:80", location.String(); e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}

	// Test a name + port number (service port 9393 -> target port "p" -> endpoint port 93)
	location, _, err = redirector.ResourceLocation(ctx, "foo:9393")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if location == nil {
		t.Errorf("Unexpected nil: %v", location)
	}
	if e, a := "//1.2.3.4:93", location.String(); e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}

	// Test a scheme + name + port.
	location, _, err = redirector.ResourceLocation(ctx, "https:foo:p")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if location == nil {
		t.Errorf("Unexpected nil: %v", location)
	}
	if e, a := "https://1.2.3.4:93", location.String(); e != a {
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
	s, _ := storage.List(ctx, nil)
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
	storage, _ := NewTestREST(t, nil)

	svc1 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}
	ctx := api.NewDefaultContext()
	created_svc1, _ := storage.Create(ctx, svc1)
	created_service_1 := created_svc1.(*api.Service)
	if created_service_1.Name != "foo" {
		t.Errorf("Expected foo, but got %v", created_service_1.Name)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service_1.Spec.ClusterIP)) {
		t.Errorf("Unexpected ClusterIP: %s", created_service_1.Spec.ClusterIP)
	}

	svc2 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "bar"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		}}
	ctx = api.NewDefaultContext()
	created_svc2, _ := storage.Create(ctx, svc2)
	created_service_2 := created_svc2.(*api.Service)
	if created_service_2.Name != "bar" {
		t.Errorf("Expected bar, but got %v", created_service_2.Name)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service_2.Spec.ClusterIP)) {
		t.Errorf("Unexpected ClusterIP: %s", created_service_2.Spec.ClusterIP)
	}

	testIPs := []string{"1.2.3.93", "1.2.3.94", "1.2.3.95", "1.2.3.96"}
	testIP := ""
	for _, ip := range testIPs {
		if !storage.serviceIPs.(*ipallocator.Range).Has(net.ParseIP(ip)) {
			testIP = ip
			break
		}
	}

	svc3 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "quux"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			ClusterIP:       testIP,
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}
	ctx = api.NewDefaultContext()
	created_svc3, err := storage.Create(ctx, svc3)
	if err != nil {
		t.Fatal(err)
	}
	created_service_3 := created_svc3.(*api.Service)
	if created_service_3.Spec.ClusterIP != testIP { // specific IP
		t.Errorf("Unexpected ClusterIP: %s", created_service_3.Spec.ClusterIP)
	}
}

func TestServiceRegistryIPReallocation(t *testing.T) {
	storage, _ := NewTestREST(t, nil)

	svc1 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}
	ctx := api.NewDefaultContext()
	created_svc1, _ := storage.Create(ctx, svc1)
	created_service_1 := created_svc1.(*api.Service)
	if created_service_1.Name != "foo" {
		t.Errorf("Expected foo, but got %v", created_service_1.Name)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service_1.Spec.ClusterIP)) {
		t.Errorf("Unexpected ClusterIP: %s", created_service_1.Spec.ClusterIP)
	}

	_, err := storage.Delete(ctx, created_service_1.Name)
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
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}
	ctx = api.NewDefaultContext()
	created_svc2, _ := storage.Create(ctx, svc2)
	created_service_2 := created_svc2.(*api.Service)
	if created_service_2.Name != "bar" {
		t.Errorf("Expected bar, but got %v", created_service_2.Name)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service_2.Spec.ClusterIP)) {
		t.Errorf("Unexpected ClusterIP: %s", created_service_2.Spec.ClusterIP)
	}
}

func TestServiceRegistryIPUpdate(t *testing.T) {
	storage, _ := NewTestREST(t, nil)

	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}
	ctx := api.NewDefaultContext()
	created_svc, _ := storage.Create(ctx, svc)
	created_service := created_svc.(*api.Service)
	if created_service.Spec.Ports[0].Port != 6502 {
		t.Errorf("Expected port 6502, but got %v", created_service.Spec.Ports[0].Port)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service.Spec.ClusterIP)) {
		t.Errorf("Unexpected ClusterIP: %s", created_service.Spec.ClusterIP)
	}

	update := deepCloneService(created_service)
	update.Spec.Ports[0].Port = 6503

	updated_svc, _, _ := storage.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(update, api.Scheme))
	updated_service := updated_svc.(*api.Service)
	if updated_service.Spec.Ports[0].Port != 6503 {
		t.Errorf("Expected port 6503, but got %v", updated_service.Spec.Ports[0].Port)
	}

	testIPs := []string{"1.2.3.93", "1.2.3.94", "1.2.3.95", "1.2.3.96"}
	testIP := ""
	for _, ip := range testIPs {
		if !storage.serviceIPs.(*ipallocator.Range).Has(net.ParseIP(ip)) {
			testIP = ip
			break
		}
	}

	update = deepCloneService(created_service)
	update.Spec.Ports[0].Port = 6503
	update.Spec.ClusterIP = testIP // Error: Cluster IP is immutable

	_, _, err := storage.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(update, api.Scheme))
	if err == nil || !errors.IsInvalid(err) {
		t.Errorf("Unexpected error type: %v", err)
	}
}

func TestServiceRegistryIPLoadBalancer(t *testing.T) {
	storage, _ := NewTestREST(t, nil)

	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeLoadBalancer,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}
	ctx := api.NewDefaultContext()
	created_svc, _ := storage.Create(ctx, svc)
	created_service := created_svc.(*api.Service)
	if created_service.Spec.Ports[0].Port != 6502 {
		t.Errorf("Expected port 6502, but got %v", created_service.Spec.Ports[0].Port)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service.Spec.ClusterIP)) {
		t.Errorf("Unexpected ClusterIP: %s", created_service.Spec.ClusterIP)
	}

	update := deepCloneService(created_service)

	_, _, err := storage.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(update, api.Scheme))
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
}

func TestUpdateServiceWithConflictingNamespace(t *testing.T) {
	storage, _ := NewTestREST(t, nil)
	service := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "not-default"},
	}

	ctx := api.NewDefaultContext()
	obj, created, err := storage.Update(ctx, service.Name, rest.DefaultUpdatedObjectInfo(service, api.Scheme))
	if obj != nil || created {
		t.Error("Expected a nil object, but we got a value or created was true")
	}
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if strings.Index(err.Error(), "Service.Namespace does not match the provided context") == -1 {
		t.Errorf("Expected 'Service.Namespace does not match the provided context' error, got '%s'", err.Error())
	}
}

// Validate allocation of a nodePort when the externalTraffic=OnlyLocal annotation is set
// and type is LoadBalancer
func TestServiceRegistryExternalTrafficAnnotationHealthCheckNodePortAllocation(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, _ := NewTestREST(t, nil)
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "external-lb-esipp",
			Annotations: map[string]string{
				service.AnnotationExternalTraffic: service.AnnotationValueExternalTrafficLocal,
			},
		},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeLoadBalancer,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}
	created_svc, err := storage.Create(ctx, svc)
	if created_svc == nil || err != nil {
		t.Errorf("Unexpected failure creating service %v", err)
	}
	created_service := created_svc.(*api.Service)
	if !service.NeedsHealthCheck(created_service) {
		t.Errorf("Unexpected missing annotation %s", service.AnnotationExternalTraffic)
	}
	port := service.GetServiceHealthCheckNodePort(created_service)
	if port == 0 {
		t.Errorf("Failed to allocate and create the health check node port annotation %s", service.AnnotationHealthCheckNodePort)
	}

}

// Validate using the user specified nodePort when the externalTraffic=OnlyLocal annotation is set
// and type is LoadBalancer
func TestServiceRegistryExternalTrafficAnnotationHealthCheckNodePortUserAllocation(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, _ := NewTestREST(t, nil)
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "external-lb-esipp",
			Annotations: map[string]string{
				service.AnnotationExternalTraffic:     service.AnnotationValueExternalTrafficLocal,
				service.AnnotationHealthCheckNodePort: "30200",
			},
		},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeLoadBalancer,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}
	created_svc, err := storage.Create(ctx, svc)
	if created_svc == nil || err != nil {
		t.Errorf("Unexpected failure creating service %v", err)
	}
	created_service := created_svc.(*api.Service)
	if !service.NeedsHealthCheck(created_service) {
		t.Errorf("Unexpected missing annotation %s", service.AnnotationExternalTraffic)
	}
	port := service.GetServiceHealthCheckNodePort(created_service)
	if port == 0 {
		t.Errorf("Failed to allocate and create the health check node port annotation %s", service.AnnotationHealthCheckNodePort)
	}
	if port != 30200 {
		t.Errorf("Failed to allocate requested nodePort expected 30200, got %d", port)
	}
}

// Validate that the service creation fails when the requested port number is -1
func TestServiceRegistryExternalTrafficAnnotationNegative(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, _ := NewTestREST(t, nil)
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "external-lb-esipp",
			Annotations: map[string]string{
				service.AnnotationExternalTraffic:     service.AnnotationValueExternalTrafficLocal,
				service.AnnotationHealthCheckNodePort: "-1",
			},
		},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeLoadBalancer,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}
	created_svc, err := storage.Create(ctx, svc)
	if created_svc == nil || err != nil {
		return
	}
	t.Errorf("Unexpected creation of service with invalid healthCheckNodePort specified")
}

// Validate that the health check nodePort is not allocated when the externalTraffic annotation is !"OnlyLocal"
func TestServiceRegistryExternalTrafficAnnotationGlobal(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, _ := NewTestREST(t, nil)
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "external-lb-esipp",
			Annotations: map[string]string{
				service.AnnotationExternalTraffic: service.AnnotationValueExternalTrafficGlobal,
			},
		},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeLoadBalancer,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}
	created_svc, err := storage.Create(ctx, svc)
	if created_svc == nil || err != nil {
		t.Errorf("Unexpected failure creating service %v", err)
	}
	created_service := created_svc.(*api.Service)
	// Make sure the service does not have the annotation
	if service.NeedsHealthCheck(created_service) {
		t.Errorf("Unexpected value for annotation %s", service.AnnotationExternalTraffic)
	}
	// Make sure the service does not have the health check node port allocated
	port := service.GetServiceHealthCheckNodePort(created_service)
	if port != 0 {
		t.Errorf("Unexpected allocation of health check node port annotation %s", service.AnnotationHealthCheckNodePort)
	}
}

// Validate that the health check nodePort is not allocated when service type is ClusterIP
func TestServiceRegistryExternalTrafficAnnotationClusterIP(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, _ := NewTestREST(t, nil)
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "external-lb-esipp",
			Annotations: map[string]string{
				service.AnnotationExternalTraffic: service.AnnotationValueExternalTrafficGlobal,
			},
		},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}
	created_svc, err := storage.Create(ctx, svc)
	if created_svc == nil || err != nil {
		t.Errorf("Unexpected failure creating service %v", err)
	}
	created_service := created_svc.(*api.Service)
	// Make sure that ClusterIP services do not have the health check node port allocated
	port := service.GetServiceHealthCheckNodePort(created_service)
	if port != 0 {
		t.Errorf("Unexpected allocation of health check node port annotation %s", service.AnnotationHealthCheckNodePort)
	}
}
