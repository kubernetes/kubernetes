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

package storage

import (
	"context"
	"net"
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/util/dryrun"

	"k8s.io/kubernetes/pkg/api/service"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	endpointstore "k8s.io/kubernetes/pkg/registry/core/endpoint/storage"
	podstore "k8s.io/kubernetes/pkg/registry/core/pod/storage"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

// TODO(wojtek-t): Cleanup this file.
// It is now testing mostly the same things as other resources but
// in a completely different way. We should unify it.

type serviceStorage struct {
	GottenID           string
	UpdatedID          string
	CreatedID          string
	DeletedID          string
	Created            bool
	DeletedImmediately bool
	Service            *api.Service
	OldService         *api.Service
	ServiceList        *api.ServiceList
	Err                error
}

func (s *serviceStorage) NamespaceScoped() bool {
	return true
}

func (s *serviceStorage) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	s.GottenID = name
	return s.Service, s.Err
}

func (s *serviceStorage) GetService(ctx context.Context, name string, options *metav1.GetOptions) (*api.Service, error) {
	return s.Service, s.Err
}

func (s *serviceStorage) NewList() runtime.Object {
	panic("not implemented")
}

func (s *serviceStorage) List(ctx context.Context, options *metainternalversion.ListOptions) (runtime.Object, error) {
	ns, _ := genericapirequest.NamespaceFrom(ctx)

	// Copy metadata from internal list into result
	res := new(api.ServiceList)
	res.TypeMeta = s.ServiceList.TypeMeta
	res.ListMeta = s.ServiceList.ListMeta

	if ns != metav1.NamespaceAll {
		for _, service := range s.ServiceList.Items {
			if ns == service.Namespace {
				res.Items = append(res.Items, service)
			}
		}
	} else {
		res.Items = append([]api.Service{}, s.ServiceList.Items...)
	}

	return res, s.Err
}

func (s *serviceStorage) New() runtime.Object {
	panic("not implemented")
}

func (s *serviceStorage) Create(ctx context.Context, obj runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	if dryrun.IsDryRun(options.DryRun) {
		return obj, s.Err
	}
	svc := obj.(*api.Service)
	s.CreatedID = obj.(metav1.Object).GetName()
	s.Service = svc.DeepCopy()

	if s.ServiceList == nil {
		s.ServiceList = &api.ServiceList{}
	}

	s.ServiceList.Items = append(s.ServiceList.Items, *svc)
	return svc, s.Err
}

func (s *serviceStorage) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	obj, err := objInfo.UpdatedObject(ctx, s.OldService)
	if err != nil {
		return nil, false, err
	}
	if !dryrun.IsDryRun(options.DryRun) {
		s.UpdatedID = name
		s.Service = obj.(*api.Service)
	}
	return obj, s.Created, s.Err
}

func (s *serviceStorage) Delete(ctx context.Context, name string, deleteValidation rest.ValidateObjectFunc, options *metav1.DeleteOptions) (runtime.Object, bool, error) {
	if !dryrun.IsDryRun(options.DryRun) {
		s.DeletedID = name
	}
	return s.Service, s.DeletedImmediately, s.Err
}

func (s *serviceStorage) DeleteCollection(ctx context.Context, deleteValidation rest.ValidateObjectFunc, options *metav1.DeleteOptions, listOptions *metainternalversion.ListOptions) (runtime.Object, error) {
	panic("not implemented")
}

func (s *serviceStorage) Watch(ctx context.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	panic("not implemented")
}

func (s *serviceStorage) ConvertToTable(ctx context.Context, object runtime.Object, tableOptions runtime.Object) (*metav1beta1.Table, error) {
	panic("not implemented")
}

func (s *serviceStorage) Export(ctx context.Context, name string, opts metav1.ExportOptions) (runtime.Object, error) {
	panic("not implemented")
}

func (s *serviceStorage) StorageVersion() runtime.GroupVersioner {
	panic("not implemented")
}

func generateRandomNodePort() int32 {
	return int32(rand.IntnRange(30001, 30999))
}

func NewTestREST(t *testing.T, endpoints *api.EndpointsList) (*REST, *serviceStorage, *etcd3testing.EtcdTestServer) {
	return NewTestRESTWithPods(t, endpoints, nil)
}

func NewTestRESTWithPods(t *testing.T, endpoints *api.EndpointsList, pods *api.PodList) (*REST, *serviceStorage, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")

	serviceStorage := &serviceStorage{}

	podStorage, err := podstore.NewStorage(generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 3,
		ResourcePrefix:          "pods",
	}, nil, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	if pods != nil && len(pods.Items) > 0 {
		ctx := genericapirequest.NewDefaultContext()
		for ix := range pods.Items {
			key, _ := podStorage.Pod.KeyFunc(ctx, pods.Items[ix].Name)
			if err := podStorage.Pod.Storage.Create(ctx, key, &pods.Items[ix], nil, 0, false); err != nil {
				t.Fatalf("Couldn't create pod: %v", err)
			}
		}
	}
	endpointStorage, err := endpointstore.NewREST(generic.RESTOptions{
		StorageConfig:  etcdStorage,
		Decorator:      generic.UndecoratedStorage,
		ResourcePrefix: "endpoints",
	})
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	if endpoints != nil && len(endpoints.Items) > 0 {
		ctx := genericapirequest.NewDefaultContext()
		for ix := range endpoints.Items {
			key, _ := endpointStorage.KeyFunc(ctx, endpoints.Items[ix].Name)
			if err := endpointStorage.Store.Storage.Create(ctx, key, &endpoints.Items[ix], nil, 0, false); err != nil {
				t.Fatalf("Couldn't create endpoint: %v", err)
			}
		}
	}

	r, err := ipallocator.NewCIDRRange(makeIPNet(t))
	if err != nil {
		t.Fatalf("cannot create CIDR Range %v", err)
	}

	portRange := utilnet.PortRange{Base: 30000, Size: 1000}
	portAllocator, err := portallocator.NewPortAllocator(portRange)
	if err != nil {
		t.Fatalf("cannot create port allocator %v", err)
	}

	rest, _ := NewREST(serviceStorage, endpointStorage, podStorage.Pod, r, portAllocator, nil)

	return rest, serviceStorage, server
}

func makeIPNet(t *testing.T) *net.IPNet {
	_, net, err := net.ParseCIDR("1.2.3.0/24")
	if err != nil {
		t.Error(err)
	}
	return net
}

func releaseServiceNodePorts(t *testing.T, ctx context.Context, svcName string, rest *REST, registry ServiceStorage) {
	obj, err := registry.Get(ctx, svcName, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	srv := obj.(*api.Service)
	if srv == nil {
		t.Fatalf("Failed to find service: %s", svcName)
	}
	serviceNodePorts := collectServiceNodePorts(srv)
	if len(serviceNodePorts) == 0 {
		t.Errorf("Failed to find NodePorts of service : %s", srv.Name)
	}
	for i := range serviceNodePorts {
		nodePort := serviceNodePorts[i]
		rest.serviceNodePorts.Release(nodePort)
	}
}

func TestServiceRegistryCreate(t *testing.T) {
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)

	svc := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
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
	ctx := genericapirequest.NewDefaultContext()
	created_svc, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	created_service := created_svc.(*api.Service)
	objMeta, err := meta.Accessor(created_service)
	if err != nil {
		t.Fatal(err)
	}
	if !metav1.HasObjectMetaSystemFieldValues(objMeta) {
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
	srv, err := registry.GetService(ctx, svc.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if srv == nil {
		t.Errorf("Failed to find service: %s", svc.Name)
	}
}

func TestServiceRegistryCreateDryRun(t *testing.T) {
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)

	// Test dry run create request with cluster ip
	svc := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			ClusterIP:       "1.2.3.4",
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}
	ctx := genericapirequest.NewDefaultContext()
	_, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if storage.serviceIPs.Has(net.ParseIP("1.2.3.4")) {
		t.Errorf("unexpected side effect: ip allocated")
	}
	srv, err := registry.GetService(ctx, svc.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if srv != nil {
		t.Errorf("unexpected service found: %v", srv)
	}

	// Test dry run create request with a node port
	svc = &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeNodePort,
			Ports: []api.ServicePort{{
				NodePort:   30010,
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}
	_, err = storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if storage.serviceNodePorts.Has(30010) {
		t.Errorf("unexpected side effect: NodePort allocated")
	}
	srv, err = registry.GetService(ctx, svc.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if srv != nil {
		t.Errorf("unexpected service found: %v", srv)
	}

	// Test dry run create request with multi node port
	svc = &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
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
	}
	expectNodePorts := collectServiceNodePorts(svc)
	created_svc, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	created_service := created_svc.(*api.Service)
	serviceNodePorts := collectServiceNodePorts(created_service)
	if !reflect.DeepEqual(serviceNodePorts, expectNodePorts) {
		t.Errorf("Expected %v, but got %v", expectNodePorts, serviceNodePorts)
	}
	if storage.serviceNodePorts.Has(30053) {
		t.Errorf("unexpected side effect: NodePort allocated")
	}
	srv, err = registry.GetService(ctx, svc.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if srv != nil {
		t.Errorf("unexpected service found: %v", srv)
	}

	// Test dry run create request with multiple unspecified node ports,
	// so PortAllocationOperation.AllocateNext() will be called multiple times.
	svc = &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeNodePort,
			Ports: []api.ServicePort{
				{
					Name:       "port-a",
					Port:       53,
					Protocol:   api.ProtocolTCP,
					TargetPort: intstr.FromInt(6503),
				},
				{
					Name:       "port-b",
					Port:       54,
					Protocol:   api.ProtocolTCP,
					TargetPort: intstr.FromInt(6504),
				},
			},
		},
	}
	created_svc, err = storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	created_service = created_svc.(*api.Service)
	serviceNodePorts = collectServiceNodePorts(created_service)
	if len(serviceNodePorts) != 2 {
		t.Errorf("Expected service to have 2 ports, but got %v", serviceNodePorts)
	} else if serviceNodePorts[0] == serviceNodePorts[1] {
		t.Errorf("Expected unique port numbers, but got %v", serviceNodePorts)
	}
}

func TestServiceRegistryCreateMultiNodePortsService(t *testing.T) {
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)

	testCases := []struct {
		svc             *api.Service
		name            string
		expectNodePorts []int
	}{
		{
			svc: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
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
				ObjectMeta: metav1.ObjectMeta{Name: "foo2"},
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
				ObjectMeta: metav1.ObjectMeta{Name: "foo3"},
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

	ctx := genericapirequest.NewDefaultContext()
	for _, test := range testCases {
		created_svc, err := storage.Create(ctx, test.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		created_service := created_svc.(*api.Service)
		objMeta, err := meta.Accessor(created_service)
		if err != nil {
			t.Fatal(err)
		}
		if !metav1.HasObjectMetaSystemFieldValues(objMeta) {
			t.Errorf("storage did not populate object meta field values")
		}
		if created_service.Name != test.name {
			t.Errorf("Expected %s, but got %s", test.name, created_service.Name)
		}
		serviceNodePorts := collectServiceNodePorts(created_service)
		if !reflect.DeepEqual(serviceNodePorts, test.expectNodePorts) {
			t.Errorf("Expected %v, but got %v", test.expectNodePorts, serviceNodePorts)
		}
		srv, err := registry.GetService(ctx, test.name, &metav1.GetOptions{})
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if srv == nil {
			t.Fatalf("Failed to find service: %s", test.name)
		}
		for i := range serviceNodePorts {
			nodePort := serviceNodePorts[i]
			// Release the node port at the end of the test case.
			storage.serviceNodePorts.Release(nodePort)
		}
	}
}

func TestServiceStorageValidatesCreate(t *testing.T) {
	storage, _, server := NewTestREST(t, nil)
	defer server.Terminate(t)
	failureCases := map[string]api.Service{
		"empty ID": {
			ObjectMeta: metav1.ObjectMeta{Name: ""},
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
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
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
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
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
	ctx := genericapirequest.NewDefaultContext()
	for _, failureCase := range failureCases {
		c, err := storage.Create(ctx, &failureCase, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
		if c != nil {
			t.Errorf("Expected nil object")
		}
		if !errors.IsInvalid(err) {
			t.Errorf("Expected to get an invalid resource error, got %v", err)
		}
	}
}

func TestServiceRegistryUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)

	obj, err := registry.Create(ctx, &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "1", Namespace: metav1.NamespaceDefault},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz1"},
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	svc := obj.(*api.Service)
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	updated_svc, created, err := storage.Update(ctx, "foo", rest.DefaultUpdatedObjectInfo(&api.Service{
		ObjectMeta: metav1.ObjectMeta{
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
	}), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
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

func TestServiceRegistryUpdateDryRun(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)

	obj, err := registry.Create(ctx, &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "1", Namespace: metav1.NamespaceDefault},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeExternalName,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	svc := obj.(*api.Service)
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}

	// Test dry run update request external name to node port
	updated_svc, created, err := storage.Update(ctx, svc.Name, rest.DefaultUpdatedObjectInfo(&api.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:            svc.Name,
			ResourceVersion: svc.ResourceVersion},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeNodePort,
			Ports: []api.ServicePort{{
				NodePort:   30020,
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if updated_svc == nil {
		t.Errorf("Expected non-nil object")
	}
	if created {
		t.Errorf("expected not created")
	}
	if storage.serviceNodePorts.Has(30020) {
		t.Errorf("unexpected side effect: NodePort allocated")
	}
	if e, a := "", registry.UpdatedID; e != a {
		t.Errorf("Expected %q, but got %q", e, a)
	}

	// Test dry run update request external name to cluster ip
	_, _, err = storage.Update(ctx, svc.Name, rest.DefaultUpdatedObjectInfo(&api.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:            svc.Name,
			ResourceVersion: svc.ResourceVersion},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			ClusterIP:       "1.2.3.4",
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if storage.serviceIPs.Has(net.ParseIP("1.2.3.4")) {
		t.Errorf("unexpected side effect: ip allocated")
	}

	// Test dry run update request remove node port
	obj, err = storage.Create(ctx, &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo2", ResourceVersion: "1", Namespace: metav1.NamespaceDefault},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeNodePort,
			Ports: []api.ServicePort{{
				NodePort:   30020,
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	svc = obj.(*api.Service)
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	_, _, err = storage.Update(ctx, svc.Name, rest.DefaultUpdatedObjectInfo(&api.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:            svc.Name,
			ResourceVersion: svc.ResourceVersion},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeExternalName,
			ExternalName:    "foo-svc",
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if !storage.serviceNodePorts.Has(30020) {
		t.Errorf("unexpected side effect: NodePort unallocated")
	}

	// Test dry run update request remove cluster ip
	obj, err = storage.Create(ctx, &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo3", ResourceVersion: "1", Namespace: metav1.NamespaceDefault},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			ClusterIP:       "1.2.3.4",
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	svc = obj.(*api.Service)
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	_, _, err = storage.Update(ctx, svc.Name, rest.DefaultUpdatedObjectInfo(&api.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:            svc.Name,
			ResourceVersion: svc.ResourceVersion},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeExternalName,
			ExternalName:    "foo-svc",
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if !storage.serviceIPs.Has(net.ParseIP("1.2.3.4")) {
		t.Errorf("unexpected side effect: ip unallocated")
	}
}

func TestServiceStorageValidatesUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)
	registry.Create(ctx, &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
			Ports: []api.ServicePort{{
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}},
		},
	}, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	failureCases := map[string]api.Service{
		"empty ID": {
			ObjectMeta: metav1.ObjectMeta{Name: ""},
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
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
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
		c, created, err := storage.Update(ctx, failureCase.Name, rest.DefaultUpdatedObjectInfo(&failureCase), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
		if c != nil || created {
			t.Errorf("Expected nil object or created false")
		}
		if !errors.IsInvalid(err) {
			t.Errorf("Expected to get an invalid resource error, got %v", err)
		}
	}
}

func TestServiceRegistryExternalService(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)
	svc := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
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
	_, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Errorf("Failed to create service: %#v", err)
	}
	srv, err := registry.GetService(ctx, svc.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if srv == nil {
		t.Fatalf("Failed to find service: %s", svc.Name)
	}
	serviceNodePorts := collectServiceNodePorts(srv)
	if len(serviceNodePorts) == 0 {
		t.Errorf("Failed to find NodePorts of service : %s", srv.Name)
	}
	for i := range serviceNodePorts {
		nodePort := serviceNodePorts[i]
		// Release the node port at the end of the test case.
		storage.serviceNodePorts.Release(nodePort)
	}
}

func TestServiceRegistryDelete(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)
	svc := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
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
	registry.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	storage.Delete(ctx, svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
	if e, a := "foo", registry.DeletedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceRegistryDeleteDryRun(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)

	// Test dry run delete request with cluster ip
	svc := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
			ClusterIP:       "1.2.3.4",
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}
	_, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	_, _, err = storage.Delete(ctx, svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if e, a := "", registry.DeletedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
	if !storage.serviceIPs.Has(net.ParseIP("1.2.3.4")) {
		t.Errorf("unexpected side effect: ip unallocated")
	}

	// Test dry run delete request with node port
	svc = &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo2"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeNodePort,
			Ports: []api.ServicePort{{
				NodePort:   30030,
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}
	_, err = storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	_, _, err = storage.Delete(ctx, svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if e, a := "", registry.DeletedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
	if !storage.serviceNodePorts.Has(30030) {
		t.Errorf("unexpected side effect: NodePort unallocated")
	}
}

func TestServiceRegistryDeleteExternal(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)
	svc := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
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
	registry.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	storage.Delete(ctx, svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
	if e, a := "foo", registry.DeletedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceRegistryUpdateExternalService(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)

	// Create non-external load balancer.
	svc1 := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "1"},
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
	if _, err := storage.Create(ctx, svc1, rest.ValidateAllObjectFunc, &metav1.CreateOptions{}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Modify load balancer to be external.
	svc2 := svc1.DeepCopy()
	svc2.Spec.Type = api.ServiceTypeLoadBalancer
	if _, _, err := storage.Update(ctx, svc2.Name, rest.DefaultUpdatedObjectInfo(svc2), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer releaseServiceNodePorts(t, ctx, svc2.Name, storage, registry)

	// Change port.
	svc3 := svc2.DeepCopy()
	svc3.Spec.Ports[0].Port = 6504
	if _, _, err := storage.Update(ctx, svc3.Name, rest.DefaultUpdatedObjectInfo(svc3), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
}

func TestServiceRegistryUpdateMultiPortExternalService(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)

	// Create external load balancer.
	svc1 := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "1"},
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
	if _, err := storage.Create(ctx, svc1, rest.ValidateAllObjectFunc, &metav1.CreateOptions{}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer releaseServiceNodePorts(t, ctx, svc1.Name, storage, registry)

	// Modify ports
	svc2 := svc1.DeepCopy()
	svc2.Spec.Ports[1].Port = 8088
	if _, _, err := storage.Update(ctx, svc2.Name, rest.DefaultUpdatedObjectInfo(svc2), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
}

func TestServiceRegistryGet(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)
	registry.Create(ctx, &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
		},
	}, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	storage.Get(ctx, "foo", &metav1.GetOptions{})
	if e, a := "foo", registry.GottenID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceRegistryResourceLocation(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	endpoints := &api.EndpointsList{
		Items: []api.Endpoints{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bad",
					Namespace: metav1.NamespaceDefault,
				},
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{
						{IP: "1.2.3.4", TargetRef: &api.ObjectReference{Name: "foo", Namespace: "doesn't exist"}},
						{IP: "1.2.3.4", TargetRef: &api.ObjectReference{Name: "doesn't exist", Namespace: metav1.NamespaceDefault}},
						{IP: "23.2.3.4", TargetRef: &api.ObjectReference{Name: "foo", Namespace: metav1.NamespaceDefault}},
					},
					Ports: []api.EndpointPort{{Name: "", Port: 80}, {Name: "p", Port: 93}},
				}},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: metav1.NamespaceDefault,
				},
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4", TargetRef: &api.ObjectReference{Name: "foo", Namespace: metav1.NamespaceDefault}}},
					Ports:     []api.EndpointPort{{Name: "", Port: 80}, {Name: "p", Port: 93}},
				}},
			},
		},
	}
	pods := &api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: metav1.NamespaceDefault,
				},
				Spec: api.PodSpec{
					RestartPolicy: "Always",
					DNSPolicy:     "Default",
					Containers:    []api.Container{{Name: "bar", Image: "test", ImagePullPolicy: api.PullIfNotPresent, TerminationMessagePolicy: api.TerminationMessageReadFile}},
				},
				Status: api.PodStatus{
					PodIPs: []api.PodIP{{IP: "1.2.3.4"}, {IP: "2001:db7::"}},
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: metav1.NamespaceDefault,
				},
				Spec: api.PodSpec{
					RestartPolicy: "Always",
					DNSPolicy:     "Default",
					Containers:    []api.Container{{Name: "bar", Image: "test", ImagePullPolicy: api.PullIfNotPresent, TerminationMessagePolicy: api.TerminationMessageReadFile}},
				},
				Status: api.PodStatus{
					PodIPs: []api.PodIP{{IP: "1.2.3.5"}, {IP: "2001:db8::"}},
				},
			},
		},
	}
	storage, registry, server := NewTestRESTWithPods(t, endpoints, pods)
	defer server.Terminate(t)
	for _, name := range []string{"foo", "bad"} {
		registry.Create(ctx, &api.Service{
			ObjectMeta: metav1.ObjectMeta{Name: name},
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
		}, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	}
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

	// Test a simple id.
	_, _, err = redirector.ResourceLocation(ctx, "bad")
	if err == nil {
		t.Errorf("Unexpected nil error")
	}
}

func TestServiceRegistryList(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)
	registry.Create(ctx, &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: metav1.NamespaceDefault},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar": "baz"},
		},
	}, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	registry.Create(ctx, &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo2", Namespace: metav1.NamespaceDefault},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"bar2": "baz2"},
		},
	}, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	registry.ServiceList.ResourceVersion = "1"
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
	storage, _, server := NewTestREST(t, nil)
	defer server.Terminate(t)

	svc1 := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
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
	ctx := genericapirequest.NewDefaultContext()
	created_svc1, _ := storage.Create(ctx, svc1, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	created_service_1 := created_svc1.(*api.Service)
	if created_service_1.Name != "foo" {
		t.Errorf("Expected foo, but got %v", created_service_1.Name)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service_1.Spec.ClusterIP)) {
		t.Errorf("Unexpected ClusterIP: %s", created_service_1.Spec.ClusterIP)
	}

	svc2 := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "bar"},
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
	ctx = genericapirequest.NewDefaultContext()
	created_svc2, _ := storage.Create(ctx, svc2, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
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
		ObjectMeta: metav1.ObjectMeta{Name: "quux"},
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
	ctx = genericapirequest.NewDefaultContext()
	created_svc3, err := storage.Create(ctx, svc3, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	created_service_3 := created_svc3.(*api.Service)
	if created_service_3.Spec.ClusterIP != testIP { // specific IP
		t.Errorf("Unexpected ClusterIP: %s", created_service_3.Spec.ClusterIP)
	}
}

func TestServiceRegistryIPReallocation(t *testing.T) {
	storage, _, server := NewTestREST(t, nil)
	defer server.Terminate(t)

	svc1 := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
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
	ctx := genericapirequest.NewDefaultContext()
	created_svc1, _ := storage.Create(ctx, svc1, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	created_service_1 := created_svc1.(*api.Service)
	if created_service_1.Name != "foo" {
		t.Errorf("Expected foo, but got %v", created_service_1.Name)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service_1.Spec.ClusterIP)) {
		t.Errorf("Unexpected ClusterIP: %s", created_service_1.Spec.ClusterIP)
	}

	_, _, err := storage.Delete(ctx, created_service_1.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
	if err != nil {
		t.Errorf("Unexpected error deleting service: %v", err)
	}

	svc2 := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "bar"},
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
	ctx = genericapirequest.NewDefaultContext()
	created_svc2, _ := storage.Create(ctx, svc2, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	created_service_2 := created_svc2.(*api.Service)
	if created_service_2.Name != "bar" {
		t.Errorf("Expected bar, but got %v", created_service_2.Name)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service_2.Spec.ClusterIP)) {
		t.Errorf("Unexpected ClusterIP: %s", created_service_2.Spec.ClusterIP)
	}
}

func TestServiceRegistryIPUpdate(t *testing.T) {
	storage, _, server := NewTestREST(t, nil)
	defer server.Terminate(t)

	svc := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "1"},
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
	ctx := genericapirequest.NewDefaultContext()
	created_svc, _ := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	created_service := created_svc.(*api.Service)
	if created_service.Spec.Ports[0].Port != 6502 {
		t.Errorf("Expected port 6502, but got %v", created_service.Spec.Ports[0].Port)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service.Spec.ClusterIP)) {
		t.Errorf("Unexpected ClusterIP: %s", created_service.Spec.ClusterIP)
	}

	update := created_service.DeepCopy()
	update.Spec.Ports[0].Port = 6503

	updated_svc, _, _ := storage.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
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

	update = created_service.DeepCopy()
	update.Spec.Ports[0].Port = 6503
	update.Spec.ClusterIP = testIP // Error: Cluster IP is immutable

	_, _, err := storage.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err == nil || !errors.IsInvalid(err) {
		t.Errorf("Unexpected error type: %v", err)
	}
}

func TestServiceRegistryIPLoadBalancer(t *testing.T) {
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)

	svc := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "1"},
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
	ctx := genericapirequest.NewDefaultContext()
	created_svc, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if created_svc == nil || err != nil {
		t.Errorf("Unexpected failure creating service %v", err)
	}
	defer releaseServiceNodePorts(t, ctx, svc.Name, storage, registry)

	created_service := created_svc.(*api.Service)
	if created_service.Spec.Ports[0].Port != 6502 {
		t.Errorf("Expected port 6502, but got %v", created_service.Spec.Ports[0].Port)
	}
	if !makeIPNet(t).Contains(net.ParseIP(created_service.Spec.ClusterIP)) {
		t.Errorf("Unexpected ClusterIP: %s", created_service.Spec.ClusterIP)
	}

	update := created_service.DeepCopy()

	_, _, err = storage.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
}

func TestUpdateServiceWithConflictingNamespace(t *testing.T) {
	storage, _, server := NewTestREST(t, nil)
	defer server.Terminate(t)
	service := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "not-default"},
	}

	ctx := genericapirequest.NewDefaultContext()
	obj, created, err := storage.Update(ctx, service.Name, rest.DefaultUpdatedObjectInfo(service), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if obj != nil || created {
		t.Error("Expected a nil object, but we got a value or created was true")
	}
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if strings.Index(err.Error(), "Service.Namespace does not match the provided context") == -1 {
		t.Errorf("Expected 'Service.Namespace does not match the provided context' error, got '%s'", err.Error())
	}
}

// Validate allocation of a nodePort when ExternalTrafficPolicy is set to Local
// and type is LoadBalancer.
func TestServiceRegistryExternalTrafficHealthCheckNodePortAllocation(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)
	svc := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "external-lb-esipp"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeLoadBalancer,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
			ExternalTrafficPolicy: api.ServiceExternalTrafficPolicyTypeLocal,
		},
	}
	created_svc, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if created_svc == nil || err != nil {
		t.Errorf("Unexpected failure creating service %v", err)
	}
	defer releaseServiceNodePorts(t, ctx, svc.Name, storage, registry)

	created_service := created_svc.(*api.Service)
	if !service.NeedsHealthCheck(created_service) {
		t.Errorf("Expecting health check needed, returned health check not needed instead")
	}
	port := created_service.Spec.HealthCheckNodePort
	if port == 0 {
		t.Errorf("Failed to allocate health check node port and set the HealthCheckNodePort")
	} else {
		// Release the health check node port at the end of the test case.
		storage.serviceNodePorts.Release(int(port))
	}
}

// Validate using the user specified nodePort when ExternalTrafficPolicy is set to Local
// and type is LoadBalancer.
func TestServiceRegistryExternalTrafficHealthCheckNodePortUserAllocation(t *testing.T) {
	randomNodePort := generateRandomNodePort()
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)
	svc := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "external-lb-esipp"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeLoadBalancer,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
			ExternalTrafficPolicy: api.ServiceExternalTrafficPolicyTypeLocal,
			HealthCheckNodePort:   randomNodePort,
		},
	}
	created_svc, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if created_svc == nil || err != nil {
		t.Fatalf("Unexpected failure creating service :%v", err)
	}
	defer releaseServiceNodePorts(t, ctx, svc.Name, storage, registry)

	created_service := created_svc.(*api.Service)
	if !service.NeedsHealthCheck(created_service) {
		t.Errorf("Expecting health check needed, returned health check not needed instead")
	}
	port := created_service.Spec.HealthCheckNodePort
	if port == 0 {
		t.Errorf("Failed to allocate health check node port and set the HealthCheckNodePort")
	}
	if port != randomNodePort {
		t.Errorf("Failed to allocate requested nodePort expected %d, got %d", randomNodePort, port)
	}
	if port != 0 {
		// Release the health check node port at the end of the test case.
		storage.serviceNodePorts.Release(int(port))
	}
}

// Validate that the service creation fails when the requested port number is -1.
func TestServiceRegistryExternalTrafficHealthCheckNodePortNegative(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, _, server := NewTestREST(t, nil)
	defer server.Terminate(t)
	svc := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "external-lb-esipp"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeLoadBalancer,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
			ExternalTrafficPolicy: api.ServiceExternalTrafficPolicyTypeLocal,
			HealthCheckNodePort:   int32(-1),
		},
	}
	created_svc, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if created_svc == nil || err != nil {
		return
	}
	t.Errorf("Unexpected creation of service with invalid HealthCheckNodePort specified")
}

// Validate that the health check nodePort is not allocated when ExternalTrafficPolicy is set to Global.
func TestServiceRegistryExternalTrafficGlobal(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil)
	defer server.Terminate(t)
	svc := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "external-lb-esipp"},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeLoadBalancer,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
			ExternalTrafficPolicy: api.ServiceExternalTrafficPolicyTypeCluster,
		},
	}
	created_svc, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if created_svc == nil || err != nil {
		t.Errorf("Unexpected failure creating service %v", err)
	}
	defer releaseServiceNodePorts(t, ctx, svc.Name, storage, registry)

	created_service := created_svc.(*api.Service)
	if service.NeedsHealthCheck(created_service) {
		t.Errorf("Expecting health check not needed, returned health check needed instead")
	}
	// Make sure the service does not have the health check node port allocated
	port := created_service.Spec.HealthCheckNodePort
	if port != 0 {
		// Release the health check node port at the end of the test case.
		storage.serviceNodePorts.Release(int(port))
		t.Errorf("Unexpected allocation of health check node port: %v", port)
	}
}

func TestInitClusterIP(t *testing.T) {
	storage, _, server := NewTestREST(t, nil)
	defer server.Terminate(t)

	testCases := []struct {
		name            string
		svc             *api.Service
		expectClusterIP bool
	}{
		{
			name: "Allocate new ClusterIP",
			svc: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
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
			expectClusterIP: true,
		},
		{
			name: "Allocate specified ClusterIP",
			svc: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeClusterIP,
					ClusterIP:       "1.2.3.4",
					Ports: []api.ServicePort{{
						Port:       6502,
						Protocol:   api.ProtocolTCP,
						TargetPort: intstr.FromInt(6502),
					}},
				},
			},
			expectClusterIP: true,
		},
		{
			name: "Shouldn't allocate ClusterIP",
			svc: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeClusterIP,
					ClusterIP:       api.ClusterIPNone,
					Ports: []api.ServicePort{{
						Port:       6502,
						Protocol:   api.ProtocolTCP,
						TargetPort: intstr.FromInt(6502),
					}},
				},
			},
			expectClusterIP: false,
		},
	}

	for _, test := range testCases {
		hasAllocatedIP, err := initClusterIP(test.svc, storage.serviceIPs)
		if err != nil {
			t.Errorf("%q: unexpected error: %v", test.name, err)
		}

		if hasAllocatedIP != test.expectClusterIP {
			t.Errorf("%q: expected %v, but got %v", test.name, test.expectClusterIP, hasAllocatedIP)
		}

		if test.expectClusterIP {
			if !storage.serviceIPs.Has(net.ParseIP(test.svc.Spec.ClusterIP)) {
				t.Errorf("%q: unexpected ClusterIP %q, out of range", test.name, test.svc.Spec.ClusterIP)
			}
		}

		if test.name == "Allocate specified ClusterIP" && test.svc.Spec.ClusterIP != "1.2.3.4" {
			t.Errorf("%q: expected ClusterIP %q, but got %q", test.name, "1.2.3.4", test.svc.Spec.ClusterIP)
		}

		if hasAllocatedIP {
			if helper.IsServiceIPSet(test.svc) {
				storage.serviceIPs.Release(net.ParseIP(test.svc.Spec.ClusterIP))
			}
		}
	}
}

func TestInitNodePorts(t *testing.T) {
	storage, _, server := NewTestREST(t, nil)
	defer server.Terminate(t)
	nodePortOp := portallocator.StartOperation(storage.serviceNodePorts, false)
	defer nodePortOp.Finish()

	testCases := []struct {
		name                     string
		service                  *api.Service
		expectSpecifiedNodePorts []int
	}{
		{
			name: "Service doesn't have specified NodePort",
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"bar": "baz"},
					Type:     api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Name:       "port-tcp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolTCP,
						},
					},
				},
			},
			expectSpecifiedNodePorts: []int{},
		},
		{
			name: "Service has one specified NodePort",
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"bar": "baz"},
					Type:     api.ServiceTypeNodePort,
					Ports: []api.ServicePort{{
						Name:       "port-tcp",
						Port:       53,
						TargetPort: intstr.FromInt(6502),
						Protocol:   api.ProtocolTCP,
						NodePort:   30053,
					}},
				},
			},
			expectSpecifiedNodePorts: []int{30053},
		},
		{
			name: "Service has two same ports with different protocols and specifies same NodePorts",
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"bar": "baz"},
					Type:     api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Name:       "port-tcp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolTCP,
							NodePort:   30054,
						},
						{
							Name:       "port-udp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolUDP,
							NodePort:   30054,
						},
					},
				},
			},
			expectSpecifiedNodePorts: []int{30054, 30054},
		},
		{
			name: "Service has two same ports with different protocols and specifies different NodePorts",
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"bar": "baz"},
					Type:     api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Name:       "port-tcp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolTCP,
							NodePort:   30055,
						},
						{
							Name:       "port-udp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolUDP,
							NodePort:   30056,
						},
					},
				},
			},
			expectSpecifiedNodePorts: []int{30055, 30056},
		},
		{
			name: "Service has two different ports with different protocols and specifies different NodePorts",
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"bar": "baz"},
					Type:     api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Name:       "port-tcp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolTCP,
							NodePort:   30057,
						},
						{
							Name:       "port-udp",
							Port:       54,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolUDP,
							NodePort:   30058,
						},
					},
				},
			},
			expectSpecifiedNodePorts: []int{30057, 30058},
		},
		{
			name: "Service has two same ports with different protocols but only specifies one NodePort",
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"bar": "baz"},
					Type:     api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Name:       "port-tcp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolTCP,
							NodePort:   30059,
						},
						{
							Name:       "port-udp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolUDP,
						},
					},
				},
			},
			expectSpecifiedNodePorts: []int{30059, 30059},
		},
	}

	for _, test := range testCases {
		err := initNodePorts(test.service, nodePortOp)
		if err != nil {
			t.Errorf("%q: unexpected error: %v", test.name, err)
			continue
		}

		serviceNodePorts := collectServiceNodePorts(test.service)
		if len(test.expectSpecifiedNodePorts) == 0 {
			for _, nodePort := range serviceNodePorts {
				if !storage.serviceNodePorts.Has(nodePort) {
					t.Errorf("%q: unexpected NodePort %d, out of range", test.name, nodePort)
				}
			}
		} else if !reflect.DeepEqual(serviceNodePorts, test.expectSpecifiedNodePorts) {
			t.Errorf("%q: expected NodePorts %v, but got %v", test.name, test.expectSpecifiedNodePorts, serviceNodePorts)
		}
		for i := range serviceNodePorts {
			nodePort := serviceNodePorts[i]
			// Release the node port at the end of the test case.
			storage.serviceNodePorts.Release(nodePort)
		}
	}
}

func TestUpdateNodePorts(t *testing.T) {
	storage, _, server := NewTestREST(t, nil)
	defer server.Terminate(t)
	nodePortOp := portallocator.StartOperation(storage.serviceNodePorts, false)
	defer nodePortOp.Finish()

	testCases := []struct {
		name                     string
		oldService               *api.Service
		newService               *api.Service
		expectSpecifiedNodePorts []int
	}{
		{
			name: "Old service and new service have the same NodePort",
			oldService: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeNodePort,
					Ports: []api.ServicePort{{
						Port:       6502,
						Protocol:   api.ProtocolTCP,
						TargetPort: intstr.FromInt(6502),
						NodePort:   30053,
					}},
				},
			},
			newService: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeNodePort,
					Ports: []api.ServicePort{{
						Port:       6502,
						Protocol:   api.ProtocolTCP,
						TargetPort: intstr.FromInt(6502),
						NodePort:   30053,
					}},
				},
			},
			expectSpecifiedNodePorts: []int{30053},
		},
		{
			name: "Old service has more NodePorts than new service has",
			oldService: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Name:       "port-tcp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolTCP,
							NodePort:   30053,
						},
						{
							Name:       "port-udp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolUDP,
							NodePort:   30053,
						},
					},
				},
			},
			newService: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Name:       "port-tcp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolTCP,
							NodePort:   30053,
						},
					},
				},
			},
			expectSpecifiedNodePorts: []int{30053},
		},
		{
			name: "Change protocol of ServicePort without changing NodePort",
			oldService: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Name:       "port-tcp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolTCP,
							NodePort:   30053,
						},
					},
				},
			},
			newService: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Name:       "port-udp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolUDP,
							NodePort:   30053,
						},
					},
				},
			},
			expectSpecifiedNodePorts: []int{30053},
		},
		{
			name: "Should allocate NodePort when changing service type to NodePort",
			oldService: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
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
			newService: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeNodePort,
					Ports: []api.ServicePort{{
						Port:       6502,
						Protocol:   api.ProtocolTCP,
						TargetPort: intstr.FromInt(6502),
					}},
				},
			},
			expectSpecifiedNodePorts: []int{},
		},
		{
			name: "Add new ServicePort with a different protocol without changing port numbers",
			oldService: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Name:       "port-tcp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolTCP,
							NodePort:   30053,
						},
					},
				},
			},
			newService: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Name:       "port-tcp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolTCP,
							NodePort:   30053,
						},
						{
							Name:       "port-udp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolUDP,
							NodePort:   30053,
						},
					},
				},
			},
			expectSpecifiedNodePorts: []int{30053, 30053},
		},
		{
			name: "Change service type from ClusterIP to NodePort with same NodePort number but different protocols",
			oldService: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeClusterIP,
					Ports: []api.ServicePort{{
						Port:       53,
						Protocol:   api.ProtocolTCP,
						TargetPort: intstr.FromInt(6502),
					}},
				},
			},
			newService: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Name:       "port-tcp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolTCP,
							NodePort:   30053,
						},
						{
							Name:       "port-udp",
							Port:       53,
							TargetPort: intstr.FromInt(6502),
							Protocol:   api.ProtocolUDP,
							NodePort:   30053,
						},
					},
				},
			},
			expectSpecifiedNodePorts: []int{30053, 30053},
		},
	}

	for _, test := range testCases {
		err := updateNodePorts(test.oldService, test.newService, nodePortOp)
		if err != nil {
			t.Errorf("%q: unexpected error: %v", test.name, err)
			continue
		}

		serviceNodePorts := collectServiceNodePorts(test.newService)
		if len(test.expectSpecifiedNodePorts) == 0 {
			for _, nodePort := range serviceNodePorts {
				if !storage.serviceNodePorts.Has(nodePort) {
					t.Errorf("%q: unexpected NodePort %d, out of range", test.name, nodePort)
				}
			}
		} else if !reflect.DeepEqual(serviceNodePorts, test.expectSpecifiedNodePorts) {
			t.Errorf("%q: expected NodePorts %v, but got %v", test.name, test.expectSpecifiedNodePorts, serviceNodePorts)
		}
		for i := range serviceNodePorts {
			nodePort := serviceNodePorts[i]
			// Release the node port at the end of the test case.
			storage.serviceNodePorts.Release(nodePort)
		}
	}
}
