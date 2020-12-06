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
	"fmt"
	"net"
	"net/http"
	"net/url"
	"reflect"
	"sort"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/util/dryrun"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	svctest "k8s.io/kubernetes/pkg/api/service/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	endpointstore "k8s.io/kubernetes/pkg/registry/core/endpoint/storage"
	podstore "k8s.io/kubernetes/pkg/registry/core/pod/storage"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	netutils "k8s.io/utils/net"
	utilpointer "k8s.io/utils/pointer"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// TODO(wojtek-t): Cleanup this file.
// It is now testing mostly the same things as other resources but
// in a completely different way. We should unify it.

type serviceStorage struct {
	inner    *GenericREST
	Services map[string]*api.Service
}

func (s *serviceStorage) saveService(svc *api.Service) {
	if s.Services == nil {
		s.Services = map[string]*api.Service{}
	}
	s.Services[svc.Name] = svc.DeepCopy()
}

func (s *serviceStorage) NamespaceScoped() bool {
	return true
}

func (s *serviceStorage) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	if s.Services[name] == nil {
		return nil, fmt.Errorf("service %q not found", name)
	}
	return s.Services[name].DeepCopy(), nil
}

func getService(getter rest.Getter, ctx context.Context, name string, options *metav1.GetOptions) (*api.Service, error) {
	obj, err := getter.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	return obj.(*api.Service), nil
}

func (s *serviceStorage) NewList() runtime.Object {
	panic("not implemented")
}

func (s *serviceStorage) List(ctx context.Context, options *metainternalversion.ListOptions) (runtime.Object, error) {
	ns, _ := genericapirequest.NamespaceFrom(ctx)

	keys := make([]string, 0, len(s.Services))
	for k := range s.Services {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	res := new(api.ServiceList)
	for _, k := range keys {
		svc := s.Services[k]
		if ns == metav1.NamespaceAll || ns == svc.Namespace {
			res.Items = append(res.Items, *svc)
		}
	}

	return res, nil
}

func (s *serviceStorage) New() runtime.Object {
	panic("not implemented")
}

func (s *serviceStorage) Create(ctx context.Context, obj runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	ret, err := s.inner.Create(ctx, obj, createValidation, options)
	if err != nil {
		return ret, err
	}

	if dryrun.IsDryRun(options.DryRun) {
		return ret.DeepCopyObject(), nil
	}
	svc := ret.(*api.Service)
	s.saveService(svc)

	return s.Services[svc.Name].DeepCopy(), nil
}

func (s *serviceStorage) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	ret, created, err := s.inner.Update(ctx, name, objInfo, createValidation, updateValidation, forceAllowCreate, options)
	if err != nil {
		return ret, created, err
	}
	if dryrun.IsDryRun(options.DryRun) {
		return ret.DeepCopyObject(), created, err
	}
	svc := ret.(*api.Service)
	s.saveService(svc)

	return s.Services[name].DeepCopy(), created, nil
}

func (s *serviceStorage) Delete(ctx context.Context, name string, deleteValidation rest.ValidateObjectFunc, options *metav1.DeleteOptions) (runtime.Object, bool, error) {
	ret, del, err := s.inner.Delete(ctx, name, deleteValidation, options)
	if err != nil {
		return ret, del, err
	}

	if dryrun.IsDryRun(options.DryRun) {
		return ret.DeepCopyObject(), del, nil
	}
	delete(s.Services, name)

	return ret.DeepCopyObject(), del, err
}

func (s *serviceStorage) DeleteCollection(ctx context.Context, deleteValidation rest.ValidateObjectFunc, options *metav1.DeleteOptions, listOptions *metainternalversion.ListOptions) (runtime.Object, error) {
	panic("not implemented")
}

func (s *serviceStorage) Watch(ctx context.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	panic("not implemented")
}

func (s *serviceStorage) ConvertToTable(ctx context.Context, object runtime.Object, tableOptions runtime.Object) (*metav1.Table, error) {
	panic("not implemented")
}

func (s *serviceStorage) StorageVersion() runtime.GroupVersioner {
	panic("not implemented")
}

// GetResetFields implements rest.ResetFieldsStrategy
func (s *serviceStorage) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	//FIXME: should panic?
	return nil
}

// ResourceLocation implements rest.Redirector
func (s *serviceStorage) ResourceLocation(ctx context.Context, id string) (remoteLocation *url.URL, transport http.RoundTripper, err error) {
	panic("not implemented")
}

func NewTestREST(t *testing.T, ipFamilies []api.IPFamily) (*REST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")

	podStorage, err := podstore.NewStorage(generic.RESTOptions{
		StorageConfig:           etcdStorage.ForResource(schema.GroupResource{Resource: "pods"}),
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 3,
		ResourcePrefix:          "pods",
	}, nil, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}

	endpointStorage, err := endpointstore.NewREST(generic.RESTOptions{
		StorageConfig:  etcdStorage.ForResource(schema.GroupResource{Resource: "endpoints"}),
		Decorator:      generic.UndecoratedStorage,
		ResourcePrefix: "endpoints",
	})
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}

	var rPrimary ipallocator.Interface
	var rSecondary ipallocator.Interface

	if len(ipFamilies) < 1 || len(ipFamilies) > 2 {
		t.Fatalf("unexpected ipfamilies passed: %v", ipFamilies)
	}
	for i, family := range ipFamilies {
		var r ipallocator.Interface
		switch family {
		case api.IPv4Protocol:
			r, err = ipallocator.NewInMemory(makeIPNet(t))
			if err != nil {
				t.Fatalf("cannot create CIDR Range %v", err)
			}
		case api.IPv6Protocol:
			r, err = ipallocator.NewInMemory(makeIPNet6(t))
			if err != nil {
				t.Fatalf("cannot create CIDR Range %v", err)
			}
		}
		switch i {
		case 0:
			rPrimary = r
		case 1:
			rSecondary = r
		}
	}

	portRange := utilnet.PortRange{Base: 30000, Size: 1000}
	portAllocator, err := portallocator.NewInMemory(portRange)
	if err != nil {
		t.Fatalf("cannot create port allocator %v", err)
	}

	ipAllocators := map[api.IPFamily]ipallocator.Interface{
		rPrimary.IPFamily(): rPrimary,
	}
	if rSecondary != nil {
		ipAllocators[rSecondary.IPFamily()] = rSecondary
	}

	inner := newInnerREST(t, etcdStorage, ipAllocators, portAllocator)
	rest, _ := NewREST(inner, endpointStorage, podStorage.Pod, rPrimary.IPFamily(), ipAllocators, portAllocator, nil)

	return rest, server
}

// This bridges to the "inner" REST implementation so tests continue to run
// during the delayering of service REST code.
func newInnerREST(t *testing.T, etcdStorage *storagebackend.ConfigForResource, ipAllocs map[api.IPFamily]ipallocator.Interface, portAlloc portallocator.Interface) *serviceStorage {
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage.ForResource(schema.GroupResource{Resource: "services"}),
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "services",
	}
	endpoints, err := endpointstore.NewREST(generic.RESTOptions{
		StorageConfig:  etcdStorage,
		Decorator:      generic.UndecoratedStorage,
		ResourcePrefix: "endpoints",
	})

	inner, _, _, err := NewGenericREST(restOptions, api.IPv4Protocol, ipAllocs, portAlloc, endpoints, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return &serviceStorage{inner: inner}
}

func makeIPNet(t *testing.T) *net.IPNet {
	_, net, err := netutils.ParseCIDRSloppy("1.2.3.0/24")
	if err != nil {
		t.Error(err)
	}
	return net
}
func makeIPNet6(t *testing.T) *net.IPNet {
	_, net, err := netutils.ParseCIDRSloppy("2000::/108")
	if err != nil {
		t.Error(err)
	}
	return net
}

func TestServiceRegistryUpdateUnspecifiedAllocations(t *testing.T) {
	type proof func(t *testing.T, s *api.Service)
	prove := func(proofs ...proof) []proof {
		return proofs
	}
	proveClusterIP := func(idx int, ip string) proof {
		return func(t *testing.T, s *api.Service) {
			if want, got := ip, s.Spec.ClusterIPs[idx]; want != got {
				t.Errorf("wrong ClusterIPs[%d]: want %q, got %q", idx, want, got)
			}
		}
	}
	proveNodePort := func(idx int, port int32) proof {
		return func(t *testing.T, s *api.Service) {
			got := s.Spec.Ports[idx].NodePort
			if port > 0 && got != port {
				t.Errorf("wrong Ports[%d].NodePort: want %d, got %d", idx, port, got)
			} else if port < 0 && got == -port {
				t.Errorf("wrong Ports[%d].NodePort: wanted anything but %d", idx, got)
			}
		}
	}
	proveHCNP := func(port int32) proof {
		return func(t *testing.T, s *api.Service) {
			got := s.Spec.HealthCheckNodePort
			if port > 0 && got != port {
				t.Errorf("wrong HealthCheckNodePort: want %d, got %d", port, got)
			} else if port < 0 && got == -port {
				t.Errorf("wrong HealthCheckNodePort: wanted anything but %d", got)
			}
		}
	}

	testCases := []struct {
		name        string
		create      *api.Service // Needs clusterIP, NodePort, and HealthCheckNodePort allocated
		update      *api.Service // Needs clusterIP, NodePort, and/or HealthCheckNodePort blank
		expectError bool
		prove       []proof
	}{{
		name: "single-ip_single-port",
		create: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal),
			svctest.SetClusterIPs("1.2.3.4"),
			svctest.SetNodePorts(30093),
			svctest.SetHealthCheckNodePort(30118)),
		update: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal)),
		prove: prove(
			proveClusterIP(0, "1.2.3.4"),
			proveNodePort(0, 30093),
			proveHCNP(30118)),
	}, {
		name: "multi-ip_multi-port",
		create: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal),
			svctest.SetClusterIPs("1.2.3.4", "2000::1"),
			svctest.SetPorts(
				svctest.MakeServicePort("p", 867, intstr.FromInt(867), api.ProtocolTCP),
				svctest.MakeServicePort("q", 5309, intstr.FromInt(5309), api.ProtocolTCP)),
			svctest.SetNodePorts(30093, 30076),
			svctest.SetHealthCheckNodePort(30118)),
		update: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal),
			svctest.SetPorts(
				svctest.MakeServicePort("p", 867, intstr.FromInt(867), api.ProtocolTCP),
				svctest.MakeServicePort("q", 5309, intstr.FromInt(5309), api.ProtocolTCP))),
		prove: prove(
			proveClusterIP(0, "1.2.3.4"),
			proveClusterIP(1, "2000::1"),
			proveNodePort(0, 30093),
			proveNodePort(1, 30076),
			proveHCNP(30118)),
	}, {
		name: "multi-ip_partial",
		create: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal),
			svctest.SetClusterIPs("1.2.3.4", "2000::1"),
			svctest.SetPorts(
				svctest.MakeServicePort("p", 867, intstr.FromInt(867), api.ProtocolTCP),
				svctest.MakeServicePort("q", 5309, intstr.FromInt(5309), api.ProtocolTCP)),
			svctest.SetNodePorts(30093, 30076),
			svctest.SetHealthCheckNodePort(30118)),
		update: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal),
			svctest.SetClusterIPs("1.2.3.4")),
		expectError: true,
	}, {
		name: "multi-port_partial",
		create: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal),
			svctest.SetPorts(
				svctest.MakeServicePort("p", 867, intstr.FromInt(867), api.ProtocolTCP),
				svctest.MakeServicePort("q", 5309, intstr.FromInt(5309), api.ProtocolTCP)),
			svctest.SetNodePorts(30093, 30076),
			svctest.SetHealthCheckNodePort(30118)),
		update: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal),
			svctest.SetPorts(
				svctest.MakeServicePort("p", 867, intstr.FromInt(867), api.ProtocolTCP),
				svctest.MakeServicePort("q", 5309, intstr.FromInt(5309), api.ProtocolTCP)),
			svctest.SetNodePorts(30093, 0)), // provide just 1 value
		prove: prove(
			proveNodePort(0, 30093),
			proveNodePort(1, 30076),
			proveHCNP(30118)),
	}, {
		name: "swap-ports",
		create: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal),
			svctest.SetPorts(
				svctest.MakeServicePort("p", 867, intstr.FromInt(867), api.ProtocolTCP),
				svctest.MakeServicePort("q", 5309, intstr.FromInt(5309), api.ProtocolTCP)),
			svctest.SetNodePorts(30093, 30076),
			svctest.SetHealthCheckNodePort(30118)),
		update: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal),
			svctest.SetPorts(
				// swapped from above
				svctest.MakeServicePort("q", 5309, intstr.FromInt(5309), api.ProtocolTCP),
				svctest.MakeServicePort("p", 867, intstr.FromInt(867), api.ProtocolTCP))),
		prove: prove(
			proveNodePort(0, 30076),
			proveNodePort(1, 30093),
			proveHCNP(30118)),
	}, {
		name: "partial-swap-ports",
		create: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal),
			svctest.SetPorts(
				svctest.MakeServicePort("p", 867, intstr.FromInt(867), api.ProtocolTCP),
				svctest.MakeServicePort("q", 5309, intstr.FromInt(5309), api.ProtocolTCP)),
			svctest.SetNodePorts(30093, 30076),
			svctest.SetHealthCheckNodePort(30118)),
		update: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal),
			svctest.SetPorts(
				svctest.MakeServicePort("p", 867, intstr.FromInt(867), api.ProtocolTCP),
				svctest.MakeServicePort("q", 5309, intstr.FromInt(5309), api.ProtocolTCP)),
			svctest.SetNodePorts(30076, 0), // set [0] to [1], omit [1]
			svctest.SetHealthCheckNodePort(30118)),
		prove: prove(
			proveNodePort(0, 30076),
			proveNodePort(1, -30076),
			proveHCNP(30118)),
	}, {
		name: "swap-port-with-hcnp",
		create: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal),
			svctest.SetPorts(
				svctest.MakeServicePort("p", 867, intstr.FromInt(867), api.ProtocolTCP),
				svctest.MakeServicePort("q", 5309, intstr.FromInt(5309), api.ProtocolTCP)),
			svctest.SetNodePorts(30093, 30076),
			svctest.SetHealthCheckNodePort(30118)),
		update: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal),
			svctest.SetPorts(
				svctest.MakeServicePort("p", 867, intstr.FromInt(867), api.ProtocolTCP),
				svctest.MakeServicePort("q", 5309, intstr.FromInt(5309), api.ProtocolTCP)),
			svctest.SetNodePorts(30076, 30118)), // set [0] to [1], set [1] to HCNP
		expectError: true,
	}, {
		name: "partial-swap-port-with-hcnp",
		create: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal),
			svctest.SetPorts(
				svctest.MakeServicePort("p", 867, intstr.FromInt(867), api.ProtocolTCP),
				svctest.MakeServicePort("q", 5309, intstr.FromInt(5309), api.ProtocolTCP)),
			svctest.SetNodePorts(30093, 30076),
			svctest.SetHealthCheckNodePort(30118)),
		update: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal),
			svctest.SetPorts(
				svctest.MakeServicePort("p", 867, intstr.FromInt(867), api.ProtocolTCP),
				svctest.MakeServicePort("q", 5309, intstr.FromInt(5309), api.ProtocolTCP)),
			svctest.SetNodePorts(30118, 0)), // set [0] to HCNP, omit [1]
		expectError: true,
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := genericapirequest.NewDefaultContext()
			storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol})
			defer server.Terminate(t)

			svc := tc.create.DeepCopy()
			obj, err := storage.Create(ctx, svc.DeepCopy(), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("unexpected error on create: %v", err)
			}
			createdSvc := obj.(*api.Service)
			if createdSvc.Spec.ClusterIP == "" {
				t.Fatalf("expected ClusterIP to be set")
			}
			if len(createdSvc.Spec.ClusterIPs) == 0 {
				t.Fatalf("expected ClusterIPs to be set")
			}
			for i := range createdSvc.Spec.Ports {
				if createdSvc.Spec.Ports[i].NodePort == 0 {
					t.Fatalf("expected NodePort[%d] to be set", i)
				}
			}
			if createdSvc.Spec.HealthCheckNodePort == 0 {
				t.Fatalf("expected HealthCheckNodePort to be set")
			}

			// Update - change the selector to be sure.
			svc = tc.update.DeepCopy()
			svc.Spec.Selector = map[string]string{"bar": "baz2"}
			svc.ResourceVersion = createdSvc.ResourceVersion

			obj, _, err = storage.Update(ctx, svc.Name, rest.DefaultUpdatedObjectInfo(svc.DeepCopy()),
				rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
			if tc.expectError {
				if err == nil {
					t.Fatalf("unexpected success on update")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error on update: %v", err)
			}
			updatedSvc := obj.(*api.Service)

			if want, got := createdSvc.Spec.ClusterIP, updatedSvc.Spec.ClusterIP; want != got {
				t.Errorf("expected ClusterIP to not change: wanted %v, got %v", want, got)
			}
			if want, got := createdSvc.Spec.ClusterIPs, updatedSvc.Spec.ClusterIPs; !reflect.DeepEqual(want, got) {
				t.Errorf("expected ClusterIPs to not change: wanted %v, got %v", want, got)
			}

			for _, proof := range tc.prove {
				proof(t, updatedSvc)
			}
		})
	}
}

func TestServiceStorageValidatesUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	_, err := storage.Create(ctx, svctest.MakeService("foo"), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	failureCases := map[string]*api.Service{
		"empty ID": svctest.MakeService(""),
		"invalid selector": svctest.MakeService("", func(svc *api.Service) {
			svc.Spec.Selector = map[string]string{"ThisSelectorFailsValidation": "ok"}
		}),
	}
	for _, failureCase := range failureCases {
		c, created, err := storage.Update(ctx, failureCase.Name, rest.DefaultUpdatedObjectInfo(failureCase), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
		if err == nil {
			t.Errorf("expected error")
		}
		if c != nil || created {
			t.Errorf("Expected nil object or created false")
		}
	}
}

func TestServiceRegistryUpdateLoadBalancerService(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)

	// Create non-loadbalancer.
	svc1 := svctest.MakeService("foo")
	obj, err := storage.Create(ctx, svc1, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Modify to be loadbalancer.
	svc2 := obj.(*api.Service).DeepCopy()
	svc2.Spec.Type = api.ServiceTypeLoadBalancer
	svc2.Spec.ExternalTrafficPolicy = api.ServiceExternalTrafficPolicyTypeCluster
	svc2.Spec.AllocateLoadBalancerNodePorts = utilpointer.BoolPtr(true)
	obj, _, err = storage.Update(ctx, svc2.Name, rest.DefaultUpdatedObjectInfo(svc2), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Change port.
	svc3 := obj.(*api.Service).DeepCopy()
	svc3.Spec.Ports[0].Port = 6504
	if _, _, err := storage.Update(ctx, svc3.Name, rest.DefaultUpdatedObjectInfo(svc3), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
}

func TestServiceRegistryUpdateMultiPortLoadBalancerService(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)

	// Create load balancer.
	svc1 := svctest.MakeService("foo",
		svctest.SetTypeLoadBalancer,
		svctest.SetPorts(
			svctest.MakeServicePort("p", 6502, intstr.FromInt(6502), api.ProtocolTCP),
			svctest.MakeServicePort("q", 8086, intstr.FromInt(8086), api.ProtocolTCP)))
	obj, err := storage.Create(ctx, svc1, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Modify ports
	svc2 := obj.(*api.Service).DeepCopy()
	svc2.Spec.Ports[1].Port = 8088
	if _, _, err := storage.Update(ctx, svc2.Name, rest.DefaultUpdatedObjectInfo(svc2), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
}

// this is local because it's not fully fleshed out enough for general use.
func makePod(name string, ips ...string) api.Pod {
	p := api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceDefault,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSDefault,
			Containers:    []api.Container{{Name: "ctr", Image: "img", ImagePullPolicy: api.PullIfNotPresent, TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
		Status: api.PodStatus{
			PodIPs: []api.PodIP{},
		},
	}

	for _, ip := range ips {
		p.Status.PodIPs = append(p.Status.PodIPs, api.PodIP{IP: ip})
	}

	return p
}

func TestServiceRegistryList(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	_, err := storage.Create(ctx, svctest.MakeService("foo"), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, err = storage.Create(ctx, svctest.MakeService("foo2"), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
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
}

func TestServiceRegistryIPUpdate(t *testing.T) {
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)

	svc := svctest.MakeService("foo")
	ctx := genericapirequest.NewDefaultContext()
	createdSvc, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	createdService := createdSvc.(*api.Service)
	if createdService.Spec.Ports[0].Port != svc.Spec.Ports[0].Port {
		t.Errorf("Expected port %d, but got %v", svc.Spec.Ports[0].Port, createdService.Spec.Ports[0].Port)
	}
	if !makeIPNet(t).Contains(netutils.ParseIPSloppy(createdService.Spec.ClusterIPs[0])) {
		t.Errorf("Unexpected ClusterIP: %s", createdService.Spec.ClusterIPs[0])
	}

	update := createdService.DeepCopy()
	update.Spec.Ports[0].Port = 6503

	updatedSvc, _, errUpdate := storage.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if errUpdate != nil {
		t.Fatalf("unexpected error during update %v", errUpdate)
	}
	updatedService := updatedSvc.(*api.Service)
	if updatedService.Spec.Ports[0].Port != 6503 {
		t.Errorf("Expected port 6503, but got %v", updatedService.Spec.Ports[0].Port)
	}

	testIPs := []string{"1.2.3.93", "1.2.3.94", "1.2.3.95", "1.2.3.96"}
	testIP := ""
	for _, ip := range testIPs {
		if !ipIsAllocated(t, storage.alloc.serviceIPAllocatorsByFamily[storage.alloc.defaultServiceIPFamily].(*ipallocator.Range), ip) {
			testIP = ip
			break
		}
	}

	update = updatedService.DeepCopy()
	update.Spec.Ports[0].Port = 6503
	update.Spec.ClusterIP = testIP
	update.Spec.ClusterIPs[0] = testIP

	_, _, err = storage.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err == nil || !errors.IsInvalid(err) {
		t.Errorf("Unexpected error type: %v", err)
	}
}

// Validate the internalTrafficPolicy field when set to "Cluster" then updated to "Local"
func TestServiceRegistryInternalTrafficPolicyClusterThenLocal(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	svc := svctest.MakeService("internal-traffic-policy-cluster",
		svctest.SetInternalTrafficPolicy(api.ServiceInternalTrafficPolicyCluster),
	)
	obj, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if obj == nil || err != nil {
		t.Errorf("Unexpected failure creating service %v", err)
	}

	createdSvc := obj.(*api.Service)
	if *createdSvc.Spec.InternalTrafficPolicy != api.ServiceInternalTrafficPolicyCluster {
		t.Errorf("Expecting internalTrafficPolicy field to have value Cluster, got: %s", *createdSvc.Spec.InternalTrafficPolicy)
	}

	update := createdSvc.DeepCopy()
	local := api.ServiceInternalTrafficPolicyLocal
	update.Spec.InternalTrafficPolicy = &local

	updatedSvc, _, errUpdate := storage.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if errUpdate != nil {
		t.Fatalf("unexpected error during update %v", errUpdate)
	}
	updatedService := updatedSvc.(*api.Service)
	if *updatedService.Spec.InternalTrafficPolicy != api.ServiceInternalTrafficPolicyLocal {
		t.Errorf("Expected internalTrafficPolicy to be Local, got: %s", *updatedService.Spec.InternalTrafficPolicy)
	}
}

// Validate the internalTrafficPolicy field when set to "Local" and then updated to "Cluster"
func TestServiceRegistryInternalTrafficPolicyLocalThenCluster(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	svc := svctest.MakeService("internal-traffic-policy-cluster",
		svctest.SetInternalTrafficPolicy(api.ServiceInternalTrafficPolicyLocal),
	)
	obj, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if obj == nil || err != nil {
		t.Errorf("Unexpected failure creating service %v", err)
	}

	createdSvc := obj.(*api.Service)
	if *createdSvc.Spec.InternalTrafficPolicy != api.ServiceInternalTrafficPolicyLocal {
		t.Errorf("Expecting internalTrafficPolicy field to have value Local, got: %s", *createdSvc.Spec.InternalTrafficPolicy)
	}

	update := createdSvc.DeepCopy()
	cluster := api.ServiceInternalTrafficPolicyCluster
	update.Spec.InternalTrafficPolicy = &cluster

	updatedSvc, _, errUpdate := storage.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if errUpdate != nil {
		t.Fatalf("unexpected error during update %v", errUpdate)
	}
	updatedService := updatedSvc.(*api.Service)
	if *updatedService.Spec.InternalTrafficPolicy != api.ServiceInternalTrafficPolicyCluster {
		t.Errorf("Expected internalTrafficPolicy to be Cluster, got: %s", *updatedService.Spec.InternalTrafficPolicy)
	}
}

func TestServiceUpgrade(t *testing.T) {
	requireDualStack := api.IPFamilyPolicyRequireDualStack

	ctx := genericapirequest.NewDefaultContext()
	testCases := []struct {
		name                     string
		updateFunc               func(svc *api.Service)
		enableDualStackAllocator bool
		enableDualStackGate      bool
		allocateIPsBeforeUpdate  map[api.IPFamily]string
		expectUpgradeError       bool
		svc                      *api.Service
	}{{
		name:                     "normal, no upgrade needed",
		enableDualStackAllocator: false,
		enableDualStackGate:      true,
		allocateIPsBeforeUpdate:  nil,
		expectUpgradeError:       false,

		updateFunc: func(s *api.Service) {
			s.Spec.Selector = map[string]string{"bar": "baz2"}
		},

		svc: svctest.MakeService("foo"),
	}, {
		name:                     "error, no upgrade (has single allocator)",
		enableDualStackAllocator: false,
		enableDualStackGate:      true,
		allocateIPsBeforeUpdate:  nil,
		expectUpgradeError:       true,

		updateFunc: func(s *api.Service) {
			s.Spec.IPFamilyPolicy = &requireDualStack
			s.Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol}
		},

		svc: svctest.MakeService("foo", func(s *api.Service) {
			s.Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol}
		}),
	}, {
		name:                     "upgrade to v4,6",
		enableDualStackAllocator: true,
		enableDualStackGate:      true,
		allocateIPsBeforeUpdate:  nil,
		expectUpgradeError:       false,

		updateFunc: func(s *api.Service) {
			s.Spec.IPFamilyPolicy = &requireDualStack
			s.Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol}
		},

		svc: svctest.MakeService("foo", func(s *api.Service) {
			s.Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol}
		}),
	}, {
		name:                     "upgrade to v4,6 (specific ip)",
		enableDualStackAllocator: true,
		enableDualStackGate:      true,
		allocateIPsBeforeUpdate:  nil,
		expectUpgradeError:       false,

		updateFunc: func(s *api.Service) {
			s.Spec.IPFamilyPolicy = &requireDualStack
			s.Spec.ClusterIPs = append(s.Spec.ClusterIPs, "2000:0:0:0:0:0:0:1")
			s.Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol}
		},

		svc: svctest.MakeService("foo", func(s *api.Service) {
			s.Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol}
		}),
	}, {
		name:                     "upgrade to v4,6 (specific ip) - fail, ip is not available",
		enableDualStackAllocator: true,
		enableDualStackGate:      true,
		allocateIPsBeforeUpdate:  map[api.IPFamily]string{api.IPv6Protocol: "2000:0:0:0:0:0:0:1"},
		expectUpgradeError:       true,

		updateFunc: func(s *api.Service) {
			s.Spec.IPFamilyPolicy = &requireDualStack
			s.Spec.ClusterIPs = append(s.Spec.ClusterIPs, "2000:0:0:0:0:0:0:1")
			s.Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol}
		},

		svc: svctest.MakeService("foo", func(s *api.Service) {
			s.Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol}
		}),
	}, {
		name:                     "upgrade to v6,4",
		enableDualStackAllocator: true,
		enableDualStackGate:      true,
		allocateIPsBeforeUpdate:  nil,
		expectUpgradeError:       false,

		updateFunc: func(s *api.Service) {
			s.Spec.IPFamilyPolicy = &requireDualStack
			s.Spec.IPFamilies = []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol}
		},

		svc: svctest.MakeService("foo", func(s *api.Service) {
			s.Spec.IPFamilies = []api.IPFamily{api.IPv6Protocol}
		}),
	}, {
		name:                     "upgrade to v6,4 (specific ip)",
		enableDualStackAllocator: true,
		enableDualStackGate:      true,
		allocateIPsBeforeUpdate:  nil,
		expectUpgradeError:       false,

		updateFunc: func(s *api.Service) {
			s.Spec.IPFamilyPolicy = &requireDualStack
			s.Spec.ClusterIPs = append(s.Spec.ClusterIPs, "1.2.3.4")
			s.Spec.IPFamilies = []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol}
		},

		svc: svctest.MakeService("foo", func(s *api.Service) {
			s.Spec.IPFamilies = []api.IPFamily{api.IPv6Protocol}
		}),
	}, {
		name:                     "upgrade to v6,4 (specific ip) - fail ip is already allocated",
		enableDualStackAllocator: true,
		enableDualStackGate:      true,
		allocateIPsBeforeUpdate:  map[api.IPFamily]string{api.IPv4Protocol: "1.2.3.4"},
		expectUpgradeError:       true,

		updateFunc: func(s *api.Service) {
			s.Spec.IPFamilyPolicy = &requireDualStack
			s.Spec.ClusterIPs = append(s.Spec.ClusterIPs, "1.2.3.4")
			s.Spec.IPFamilies = []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol}
		},

		svc: svctest.MakeService("foo", func(s *api.Service) {
			s.Spec.IPFamilies = []api.IPFamily{api.IPv6Protocol}
		}),
	}}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			families := []api.IPFamily{api.IPv4Protocol}
			if testCase.enableDualStackAllocator {
				families = append(families, api.IPv6Protocol)
			}
			storage, server := NewTestREST(t, families)
			defer server.Terminate(t)
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, testCase.enableDualStackGate)()

			obj, err := storage.Create(ctx, testCase.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("error is unexpected: %v", err)
			}

			createdSvc := obj.(*api.Service)
			// allocated IP
			for family, ip := range testCase.allocateIPsBeforeUpdate {
				alloc := storage.alloc.serviceIPAllocatorsByFamily[family]
				if err := alloc.Allocate(netutils.ParseIPSloppy(ip)); err != nil {
					t.Fatalf("test is incorrect, unable to preallocate ip:%v", ip)
				}
			}
			// run the modifier
			testCase.updateFunc(createdSvc)

			// run the update
			updated, _, err := storage.Update(ctx,
				createdSvc.Name,
				rest.DefaultUpdatedObjectInfo(createdSvc),
				rest.ValidateAllObjectFunc,
				rest.ValidateAllObjectUpdateFunc,
				false,
				&metav1.UpdateOptions{})

			if err != nil && !testCase.expectUpgradeError {
				t.Fatalf("an error was not expected during upgrade %v", err)
			}

			if err == nil && testCase.expectUpgradeError {
				t.Fatalf("error was expected during upgrade")
			}

			if err != nil {
				return
			}

			updatedSvc := updated.(*api.Service)
			isValidClusterIPFields(t, storage, updatedSvc, updatedSvc)

			shouldUpgrade := len(createdSvc.Spec.IPFamilies) == 2 && *(createdSvc.Spec.IPFamilyPolicy) != api.IPFamilyPolicySingleStack && len(storage.alloc.serviceIPAllocatorsByFamily) == 2
			if shouldUpgrade && len(updatedSvc.Spec.ClusterIPs) < 2 {
				t.Fatalf("Service should have been upgraded %+v", createdSvc)
			}

			if !shouldUpgrade && len(updatedSvc.Spec.ClusterIPs) > 1 {
				t.Fatalf("Service should not have been upgraded %+v", createdSvc)
			}

			// make sure that ips were allocated, correctly
			for i, family := range updatedSvc.Spec.IPFamilies {
				ip := updatedSvc.Spec.ClusterIPs[i]
				allocator := storage.alloc.serviceIPAllocatorsByFamily[family]
				if !ipIsAllocated(t, allocator, ip) {
					t.Fatalf("expected ip:%v to be allocated by %v allocator. it was not", ip, family)
				}
			}
		})
	}
}

func TestServiceDowngrade(t *testing.T) {
	requiredDualStack := api.IPFamilyPolicyRequireDualStack
	singleStack := api.IPFamilyPolicySingleStack
	ctx := genericapirequest.NewDefaultContext()
	testCases := []struct {
		name                     string
		updateFunc               func(svc *api.Service)
		enableDualStackAllocator bool
		enableDualStackGate      bool
		expectDowngradeError     bool
		svc                      *api.Service
	}{{
		name:                     "normal, no downgrade needed. single stack => single stack",
		enableDualStackAllocator: true,
		enableDualStackGate:      true,
		expectDowngradeError:     false,

		updateFunc: func(s *api.Service) { s.Spec.Selector = map[string]string{"bar": "baz2"} },

		svc: svctest.MakeService("foo", func(s *api.Service) {
			s.Spec.IPFamilyPolicy = &requiredDualStack
			s.Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol}
		}),
	}, {
		name:                     "normal, no downgrade needed. dual stack => dual stack",
		enableDualStackAllocator: true,
		enableDualStackGate:      true,
		expectDowngradeError:     false,

		updateFunc: func(s *api.Service) { s.Spec.Selector = map[string]string{"bar": "baz2"} },

		svc: svctest.MakeService("foo", func(s *api.Service) {
			s.Spec.IPFamilyPolicy = &requiredDualStack
			s.Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol}
		}),
	}, {
		name:                     "normal, downgrade v4,v6 => v4",
		enableDualStackAllocator: true,
		enableDualStackGate:      true,
		expectDowngradeError:     false,

		updateFunc: func(s *api.Service) {
			s.Spec.IPFamilyPolicy = &singleStack
			s.Spec.ClusterIPs = s.Spec.ClusterIPs[0:1]
			s.Spec.IPFamilies = s.Spec.IPFamilies[0:1]
		},

		svc: svctest.MakeService("foo", func(s *api.Service) {
			s.Spec.IPFamilyPolicy = &requiredDualStack
			s.Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol}
		}),
	}, {
		name:                     "normal, downgrade v6,v4 => v6",
		enableDualStackAllocator: true,
		enableDualStackGate:      true,
		expectDowngradeError:     false,

		updateFunc: func(s *api.Service) {
			s.Spec.IPFamilyPolicy = &singleStack
			s.Spec.ClusterIPs = s.Spec.ClusterIPs[0:1]
			s.Spec.IPFamilies = s.Spec.IPFamilies[0:1]
		},

		svc: svctest.MakeService("foo", func(s *api.Service) {
			s.Spec.IPFamilyPolicy = &requiredDualStack
			s.Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol}
		}),
	}}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol})
			defer server.Terminate(t)
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, testCase.enableDualStackGate)()

			obj, err := storage.Create(ctx, testCase.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("error is unexpected: %v", err)
			}

			createdSvc := obj.(*api.Service)
			copySvc := createdSvc.DeepCopy()

			// run the modifier
			testCase.updateFunc(createdSvc)

			// run the update
			updated, _, err := storage.Update(ctx,
				createdSvc.Name,
				rest.DefaultUpdatedObjectInfo(createdSvc),
				rest.ValidateAllObjectFunc,
				rest.ValidateAllObjectUpdateFunc,
				false,
				&metav1.UpdateOptions{})

			if err != nil && !testCase.expectDowngradeError {
				t.Fatalf("an error was not expected during upgrade %v", err)
			}

			if err == nil && testCase.expectDowngradeError {
				t.Fatalf("error was expected during upgrade")
			}

			if err != nil {
				return
			}

			updatedSvc := updated.(*api.Service)
			isValidClusterIPFields(t, storage, createdSvc, updatedSvc)

			shouldDowngrade := len(copySvc.Spec.ClusterIPs) == 2 && *(createdSvc.Spec.IPFamilyPolicy) == api.IPFamilyPolicySingleStack

			if shouldDowngrade && len(updatedSvc.Spec.ClusterIPs) > 1 {
				t.Fatalf("Service should have been downgraded %+v", createdSvc)
			}

			if !shouldDowngrade && len(updatedSvc.Spec.ClusterIPs) < 2 {
				t.Fatalf("Service should not have been downgraded %+v", createdSvc)
			}

			if shouldDowngrade {
				releasedIP := copySvc.Spec.ClusterIPs[1]
				releasedIPFamily := copySvc.Spec.IPFamilies[1]
				allocator := storage.alloc.serviceIPAllocatorsByFamily[releasedIPFamily]

				if ipIsAllocated(t, allocator, releasedIP) {
					t.Fatalf("expected ip:%v to be released by %v allocator. it was not", releasedIP, releasedIPFamily)
				}
			}
		})
	}
}

// validates that the service created, updated by REST
// has correct ClusterIPs related fields
func isValidClusterIPFields(t *testing.T, storage *REST, pre *api.Service, post *api.Service) {
	t.Helper()

	// valid for gate off/on scenarios
	// ClusterIP
	if len(post.Spec.ClusterIP) == 0 {
		t.Fatalf("service must have clusterIP : %+v", post)
	}
	// cluster IPs
	if len(post.Spec.ClusterIPs) == 0 {
		t.Fatalf("new service must have at least one IP: %+v", post)
	}

	if post.Spec.ClusterIP != post.Spec.ClusterIPs[0] {
		t.Fatalf("clusterIP does not match ClusterIPs[0]: %+v", post)
	}

	// if feature gate is not enabled then we need to ignore need fields
	if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) {
		if post.Spec.IPFamilyPolicy != nil {
			t.Fatalf("service must be set to  nil for IPFamilyPolicy: %+v", post)
		}

		if len(post.Spec.IPFamilies) != 0 {
			t.Fatalf("service must be set to nil for IPFamilies: %+v", post)
		}

		return
	}

	// for gate on scenarios
	// prefer dual stack field
	if post.Spec.IPFamilyPolicy == nil {
		t.Fatalf("service must not have nil for IPFamilyPolicy: %+v", post)
	}

	if pre.Spec.IPFamilyPolicy != nil && *(pre.Spec.IPFamilyPolicy) != *(post.Spec.IPFamilyPolicy) {
		t.Fatalf("new service must not change PreferDualStack if it was set by user pre: %v post: %v", *(pre.Spec.IPFamilyPolicy), *(post.Spec.IPFamilyPolicy))
	}

	if pre.Spec.IPFamilyPolicy == nil && *(post.Spec.IPFamilyPolicy) != api.IPFamilyPolicySingleStack {
		t.Fatalf("new services with prefer dual stack nil must be set to false (prefer dual stack) %+v", post)
	}

	// external name or headless services offer no more ClusterIPs field validation
	if post.Spec.ClusterIPs[0] == api.ClusterIPNone {
		return
	}

	// len of ClusteIPs can not be more than Families
	// and for providedIPs it needs to match

	// if families are provided then it shouldn't be changed
	// this applies on first entry on
	if len(pre.Spec.IPFamilies) > 0 {
		if len(post.Spec.IPFamilies) == 0 {
			t.Fatalf("allocator shouldn't remove ipfamilies[0] pre:%+v, post:%+v", pre.Spec.IPFamilies, post.Spec.IPFamilies)
		}

		if pre.Spec.IPFamilies[0] != post.Spec.IPFamilies[0] {
			t.Fatalf("allocator shouldn't change post.Spec.IPFamilies[0] pre:%+v post:%+v", pre.Spec.IPFamilies, post.Spec.IPFamilies)
		}
	}
	// if two families are assigned, then they must be dual stack
	if len(post.Spec.IPFamilies) == 2 {
		if post.Spec.IPFamilies[0] == post.Spec.IPFamilies[1] {
			t.Fatalf("allocator assigned two of the same family %+v", post)
		}
	}
	// ips must match families
	for i, ip := range post.Spec.ClusterIPs {
		isIPv6 := netutils.IsIPv6String(ip)
		if isIPv6 && post.Spec.IPFamilies[i] != api.IPv6Protocol {
			t.Fatalf("ips does not match assigned families %+v %+v", post.Spec.ClusterIPs, post.Spec.IPFamilies)
		}
	}
}
