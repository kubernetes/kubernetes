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
	"reflect"
	"sort"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/util/dryrun"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	epstest "k8s.io/kubernetes/pkg/api/endpoints/testing"
	"k8s.io/kubernetes/pkg/api/service"
	svctest "k8s.io/kubernetes/pkg/api/service/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	endpointstore "k8s.io/kubernetes/pkg/registry/core/endpoint/storage"
	podstore "k8s.io/kubernetes/pkg/registry/core/pod/storage"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	netutil "k8s.io/utils/net"
	utilpointer "k8s.io/utils/pointer"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// TODO(wojtek-t): Cleanup this file.
// It is now testing mostly the same things as other resources but
// in a completely different way. We should unify it.

type serviceStorage struct {
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
	if dryrun.IsDryRun(options.DryRun) {
		return obj, nil
	}
	svc := obj.(*api.Service)
	s.saveService(svc)
	s.Services[svc.Name].ResourceVersion = "1"

	return s.Services[svc.Name].DeepCopy(), nil
}

func (s *serviceStorage) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	obj, err := objInfo.UpdatedObject(ctx, nil)
	if err != nil {
		return nil, false, err
	}
	if !dryrun.IsDryRun(options.DryRun) {
		s.saveService(obj.(*api.Service))
	}
	return obj, false, nil
}

func (s *serviceStorage) Delete(ctx context.Context, name string, deleteValidation rest.ValidateObjectFunc, options *metav1.DeleteOptions) (runtime.Object, bool, error) {
	ret := s.Services[name]
	delete(s.Services, name)
	return ret, false, nil
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
	return nil
}

func NewTestREST(t *testing.T, ipFamilies []api.IPFamily) (*REST, *etcd3testing.EtcdTestServer) {
	return NewTestRESTWithPods(t, nil, nil, ipFamilies)
}

func NewTestRESTWithPods(t *testing.T, endpoints []*api.Endpoints, pods []api.Pod, ipFamilies []api.IPFamily) (*REST, *etcd3testing.EtcdTestServer) {
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
	ctx := genericapirequest.NewDefaultContext()
	for ix := range pods {
		key, _ := podStorage.Pod.KeyFunc(ctx, pods[ix].Name)
		if err := podStorage.Pod.Storage.Create(ctx, key, &pods[ix], nil, 0, false); err != nil {
			t.Fatalf("Couldn't create pod: %v", err)
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
	for ix := range endpoints {
		key, _ := endpointStorage.KeyFunc(ctx, endpoints[ix].Name)
		if err := endpointStorage.Store.Storage.Create(ctx, key, endpoints[ix], nil, 0, false); err != nil {
			t.Fatalf("Couldn't create endpoint: %v", err)
		}
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
			r, err = ipallocator.NewCIDRRange(makeIPNet(t))
			if err != nil {
				t.Fatalf("cannot create CIDR Range %v", err)
			}
		case api.IPv6Protocol:
			r, err = ipallocator.NewCIDRRange(makeIPNet6(t))
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
	portAllocator, err := portallocator.NewPortAllocator(portRange)
	if err != nil {
		t.Fatalf("cannot create port allocator %v", err)
	}

	rest, _ := NewREST(serviceStorage, endpointStorage, podStorage.Pod, rPrimary, rSecondary, portAllocator, nil)

	return rest, server
}

func makeIPNet(t *testing.T) *net.IPNet {
	_, net, err := net.ParseCIDR("1.2.3.0/24")
	if err != nil {
		t.Error(err)
	}
	return net
}
func makeIPNet6(t *testing.T) *net.IPNet {
	_, net, err := net.ParseCIDR("2000::/108")
	if err != nil {
		t.Error(err)
	}
	return net
}

func ipIsAllocated(t *testing.T, alloc ipallocator.Interface, ipstr string) bool {
	t.Helper()
	ip := net.ParseIP(ipstr)
	if ip == nil {
		t.Errorf("error parsing IP %q", ipstr)
		return false
	}
	return alloc.Has(ip)
}

func portIsAllocated(t *testing.T, alloc portallocator.Interface, port int32) bool {
	t.Helper()
	if port == 0 {
		t.Errorf("port is 0")
		return false
	}
	return alloc.Has(int(port))
}

func TestServiceRegistryCreate(t *testing.T) {
	testCases := []struct {
		svc             *api.Service
		name            string
		families        []api.IPFamily
		enableDualStack bool
	}{{
		name:            "Service IPFamily default cluster dualstack:off",
		enableDualStack: false,
		families:        []api.IPFamily{api.IPv4Protocol},
		svc:             svctest.MakeService("foo"),
	}, {
		name:            "Service IPFamily:v4 dualstack off",
		enableDualStack: false,
		families:        []api.IPFamily{api.IPv4Protocol},
		svc:             svctest.MakeService("foo", svctest.SetIPFamilies(api.IPv4Protocol)),
	}, {
		name:            "Service IPFamily:v4 dualstack on",
		enableDualStack: true,
		families:        []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		svc:             svctest.MakeService("foo", svctest.SetIPFamilies(api.IPv4Protocol)),
	}, {
		name:            "Service IPFamily:v6 dualstack on",
		enableDualStack: true,
		families:        []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		svc:             svctest.MakeService("foo", svctest.SetIPFamilies(api.IPv6Protocol)),
	}}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			storage, server := NewTestREST(t, tc.families)
			defer server.Terminate(t)

			ctx := genericapirequest.NewDefaultContext()
			createdSvc, err := storage.Create(ctx, tc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("error creating service %v", err)
			}
			createdService := createdSvc.(*api.Service)
			objMeta, err := meta.Accessor(createdService)
			if err != nil {
				t.Fatal(err)
			}
			if !metav1.HasObjectMetaSystemFieldValues(objMeta) {
				t.Errorf("storage did not populate object meta field values")
			}
			if createdService.Name != "foo" {
				t.Errorf("Expected foo, but got %v", createdService.Name)
			}
			if createdService.CreationTimestamp.IsZero() {
				t.Errorf("Expected timestamp to be set, got: %v", createdService.CreationTimestamp)
			}

			for i, family := range createdService.Spec.IPFamilies {
				allocator := storage.serviceIPAllocatorsByFamily[family]
				c := allocator.CIDR()
				cidr := &c
				if !cidr.Contains(net.ParseIP(createdService.Spec.ClusterIPs[i])) {
					t.Errorf("Unexpected ClusterIP: %s", createdService.Spec.ClusterIPs[i])
				}
			}
			srv, err := getService(storage, ctx, tc.svc.Name, &metav1.GetOptions{})
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if srv == nil {
				t.Errorf("Failed to find service: %s", tc.svc.Name)
			}
		})
	}
}

func TestServiceRegistryCreateDryRun(t *testing.T) {
	testCases := []struct {
		name            string
		svc             *api.Service
		enableDualStack bool
	}{{
		name:            "v4 service featuregate off",
		enableDualStack: false,
		svc:             svctest.MakeService("foo", svctest.SetClusterIPs("1.2.3.4")),
	}, {
		name:            "v6 service featuregate on but singlestack",
		enableDualStack: true,
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilies(api.IPv6Protocol),
			svctest.SetClusterIPs("2000::1")),
	}, {
		name:            "dualstack v4,v6 service",
		enableDualStack: true,
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
			svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
			svctest.SetClusterIPs("1.2.3.4", "2000::1")),
	}, {
		name:            "dualstack v6,v4 service",
		enableDualStack: true,
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
			svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
			svctest.SetClusterIPs("2000::1", "1.2.3.4")),
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, tc.enableDualStack)()

			families := []api.IPFamily{api.IPv4Protocol}
			if tc.enableDualStack {
				families = append(families, api.IPv6Protocol)
			}
			storage, server := NewTestREST(t, families)
			defer server.Terminate(t)

			ctx := genericapirequest.NewDefaultContext()
			_, err := storage.Create(ctx, tc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			for i, family := range tc.svc.Spec.IPFamilies {
				alloc := storage.serviceIPAllocatorsByFamily[family]
				if ipIsAllocated(t, alloc, tc.svc.Spec.ClusterIPs[i]) {
					t.Errorf("unexpected side effect: ip allocated %v", tc.svc.Spec.ClusterIPs[i])
				}
			}

			_, err = getService(storage, ctx, tc.svc.Name, &metav1.GetOptions{})
			if err == nil {
				t.Errorf("expected error")
			}
		})
	}
}

func TestDryRunNodePort(t *testing.T) {
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)

	// Test dry run create request with a node port
	svc := svctest.MakeService("foo", svctest.SetTypeNodePort)
	ctx := genericapirequest.NewDefaultContext()

	obj, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	createdSvc := obj.(*api.Service)
	if createdSvc.Spec.Ports[0].NodePort == 0 {
		t.Errorf("expected NodePort value assigned")
	}
	if portIsAllocated(t, storage.serviceNodePorts, createdSvc.Spec.Ports[0].NodePort) {
		t.Errorf("unexpected side effect: NodePort allocated")
	}
	_, err = getService(storage, ctx, svc.Name, &metav1.GetOptions{})
	if err == nil {
		// Should get a not-found.
		t.Errorf("expected error")
	}

	// Test dry run create request with multi node port
	svc = svctest.MakeService("foo",
		svctest.SetTypeNodePort,
		svctest.SetPorts(
			svctest.MakeServicePort("port-tcp", 53, intstr.FromInt(6503), api.ProtocolTCP),
			svctest.MakeServicePort("port-udp", 53, intstr.FromInt(6503), api.ProtocolUDP)),
		svctest.SetNodePorts(30053, 30053))
	expectNodePorts := collectServiceNodePorts(svc)
	obj, err = storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	createdSvc = obj.(*api.Service)
	actualNodePorts := collectServiceNodePorts(createdSvc)
	if !reflect.DeepEqual(actualNodePorts, expectNodePorts) {
		t.Errorf("Expected %v, but got %v", expectNodePorts, actualNodePorts)
	}
	for i := range svc.Spec.Ports {
		if portIsAllocated(t, storage.serviceNodePorts, svc.Spec.Ports[i].NodePort) {
			t.Errorf("unexpected side effect: NodePort allocated")
		}
	}
	_, err = getService(storage, ctx, svc.Name, &metav1.GetOptions{})
	if err == nil {
		// Should get a not-found.
		t.Errorf("expected error")
	}

	// Test dry run create request with multiple unspecified node ports,
	// so PortAllocationOperation.AllocateNext() will be called multiple times.
	svc = svctest.MakeService("foo",
		svctest.SetTypeNodePort,
		svctest.SetPorts(
			svctest.MakeServicePort("port-a", 53, intstr.FromInt(6503), api.ProtocolTCP),
			svctest.MakeServicePort("port-b", 54, intstr.FromInt(6504), api.ProtocolUDP)))
	obj, err = storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	createdSvc = obj.(*api.Service)
	actualNodePorts = collectServiceNodePorts(createdSvc)
	if len(actualNodePorts) != len(svc.Spec.Ports) {
		t.Fatalf("Expected service to have %d ports, but got %v", len(svc.Spec.Ports), actualNodePorts)
	}
	seen := map[int]bool{}
	for _, np := range actualNodePorts {
		if seen[np] {
			t.Errorf("Expected unique port numbers, but got %v", actualNodePorts)
		} else {
			seen[np] = true
		}
	}
}

func TestServiceRegistryCreateMultiNodePortsService(t *testing.T) {
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)

	testCases := []struct {
		svc             *api.Service
		name            string
		expectNodePorts []int
	}{{
		svc: svctest.MakeService("foo1",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("port-tcp", 53, intstr.FromInt(6503), api.ProtocolTCP),
				svctest.MakeServicePort("port-udp", 53, intstr.FromInt(6503), api.ProtocolUDP)),
			svctest.SetNodePorts(30053, 30053)),
		name:            "foo1",
		expectNodePorts: []int{30053, 30053},
	}, {
		svc: svctest.MakeService("foo2",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("port-tcp", 54, intstr.FromInt(6504), api.ProtocolTCP),
				svctest.MakeServicePort("port-udp", 54, intstr.FromInt(6504), api.ProtocolUDP)),
			svctest.SetNodePorts(30054, 30054)),
		name:            "foo2",
		expectNodePorts: []int{30054, 30054},
	}, {
		svc: svctest.MakeService("foo3",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("port-tcp", 55, intstr.FromInt(6505), api.ProtocolTCP),
				svctest.MakeServicePort("port-udp", 55, intstr.FromInt(6506), api.ProtocolUDP)),
			svctest.SetNodePorts(30055, 30056)),
		name:            "foo3",
		expectNodePorts: []int{30055, 30056},
	}}

	ctx := genericapirequest.NewDefaultContext()
	for _, test := range testCases {
		createdSvc, err := storage.Create(ctx, test.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		createdService := createdSvc.(*api.Service)
		objMeta, err := meta.Accessor(createdService)
		if err != nil {
			t.Fatal(err)
		}
		if !metav1.HasObjectMetaSystemFieldValues(objMeta) {
			t.Errorf("storage did not populate object meta field values")
		}
		if createdService.Name != test.name {
			t.Errorf("Expected %s, but got %s", test.name, createdService.Name)
		}
		serviceNodePorts := collectServiceNodePorts(createdService)
		if !reflect.DeepEqual(serviceNodePorts, test.expectNodePorts) {
			t.Errorf("Expected %v, but got %v", test.expectNodePorts, serviceNodePorts)
		}
		srv, err := getService(storage, ctx, test.name, &metav1.GetOptions{})
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
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	failureCases := map[string]*api.Service{
		"empty ID": svctest.MakeService(""),
		"empty port": svctest.MakeService("foo", svctest.SetPorts(
			svctest.MakeServicePort("p", 0, intstr.FromInt(80), api.ProtocolTCP))),
		"missing targetPort": svctest.MakeService("foo", svctest.SetPorts(
			svctest.MakeServicePort("p", 80, intstr.IntOrString{}, api.ProtocolTCP))),
	}
	ctx := genericapirequest.NewDefaultContext()
	for _, failureCase := range failureCases {
		c, err := storage.Create(ctx, failureCase, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
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
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)

	_, err := storage.Create(ctx, svctest.MakeService("foo"), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}

	obj, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error :%v", err)
	}
	svc := obj.(*api.Service)

	// update selector
	svc.Spec.Selector = map[string]string{"bar": "baz2"}

	updatedSvc, created, err := storage.Update(ctx, "foo", rest.DefaultUpdatedObjectInfo(svc), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if updatedSvc == nil {
		t.Errorf("Expected non-nil object")
	}
	if created {
		t.Errorf("expected not created")
	}
	updatedService := updatedSvc.(*api.Service)
	if updatedService.Name != "foo" {
		t.Errorf("Expected foo, but got %v", updatedService.Name)
	}
}

func TestServiceRegistryUpdateUnspecifiedAllocations(t *testing.T) {
	testCases := []struct {
		name  string
		svc   *api.Service // Need a clusterIP, NodePort, and HealthCheckNodePort allocated
		tweak func(*api.Service)
	}{{
		name: "single-port",
		svc: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal)),
		tweak: nil,
	}, {
		name: "multi-port",
		svc: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal),
			svctest.SetPorts(
				svctest.MakeServicePort("p", 80, intstr.FromInt(80), api.ProtocolTCP),
				svctest.MakeServicePort("q", 443, intstr.FromInt(443), api.ProtocolTCP))),
		tweak: nil,
	}, {
		name: "shuffle-ports",
		svc: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyTypeLocal),
			svctest.SetPorts(
				svctest.MakeServicePort("p", 80, intstr.FromInt(80), api.ProtocolTCP),
				svctest.MakeServicePort("q", 443, intstr.FromInt(443), api.ProtocolTCP))),
		tweak: func(s *api.Service) {
			s.Spec.Ports[0], s.Spec.Ports[1] = s.Spec.Ports[1], s.Spec.Ports[0]
		},
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := genericapirequest.NewDefaultContext()
			storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
			defer server.Terminate(t)

			svc := tc.svc.DeepCopy()
			obj, err := storage.Create(ctx, svc.DeepCopy(), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Expected no error: %v", err)
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

			// Update from the original object - just change the selector.
			svc.Spec.Selector = map[string]string{"bar": "baz2"}
			svc.ResourceVersion = createdSvc.ResourceVersion

			obj, _, err = storage.Update(ctx, svc.Name, rest.DefaultUpdatedObjectInfo(svc.DeepCopy()), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("Expected no error: %v", err)
			}
			updatedSvc := obj.(*api.Service)

			if want, got := createdSvc.Spec.ClusterIP, updatedSvc.Spec.ClusterIP; want != got {
				t.Errorf("expected ClusterIP to not change: wanted %v, got %v", want, got)
			}
			if want, got := createdSvc.Spec.ClusterIPs, updatedSvc.Spec.ClusterIPs; !reflect.DeepEqual(want, got) {
				t.Errorf("expected ClusterIPs to not change: wanted %v, got %v", want, got)
			}
			portmap := func(s *api.Service) map[string]int32 {
				ret := map[string]int32{}
				for _, p := range s.Spec.Ports {
					ret[p.Name] = p.NodePort
				}
				return ret
			}
			if want, got := portmap(createdSvc), portmap(updatedSvc); !reflect.DeepEqual(want, got) {
				t.Errorf("expected NodePort to not change: wanted %v, got %v", want, got)
			}
			if want, got := createdSvc.Spec.HealthCheckNodePort, updatedSvc.Spec.HealthCheckNodePort; want != got {
				t.Errorf("expected HealthCheckNodePort to not change: wanted %v, got %v", want, got)
			}
		})
	}
}

func TestServiceRegistryUpdateDryRun(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)

	obj, err := storage.Create(ctx, svctest.MakeService("foo", svctest.SetTypeExternalName), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	svc := obj.(*api.Service)

	// Test dry run update request external name to node port
	new1 := svc.DeepCopy()
	svctest.SetTypeNodePort(new1)
	svctest.SetNodePorts(30001)(new1) // DryRun does not set port values yet
	obj, created, err := storage.Update(ctx, new1.Name, rest.DefaultUpdatedObjectInfo(new1),
		rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if obj == nil {
		t.Errorf("Expected non-nil object")
	}
	if created {
		t.Errorf("expected not created")
	}
	if portIsAllocated(t, storage.serviceNodePorts, new1.Spec.Ports[0].NodePort) {
		t.Errorf("unexpected side effect: NodePort allocated")
	}

	// Test dry run update request external name to cluster ip
	new2 := svc.DeepCopy()
	svctest.SetTypeClusterIP(new2)
	svctest.SetClusterIPs("1.2.3.4")(new2) // DryRun does not set IP values yet
	_, _, err = storage.Update(ctx, svc.Name, rest.DefaultUpdatedObjectInfo(new2),
		rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if ipIsAllocated(t, storage.serviceIPAllocatorsByFamily[storage.defaultServiceIPFamily], new2.Spec.ClusterIP) {
		t.Errorf("unexpected side effect: ip allocated")
	}

	// Test dry run update request remove node port
	obj, err = storage.Create(ctx, svctest.MakeService("foo2", svctest.SetTypeNodePort, svctest.SetNodePorts(30001)), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	svc = obj.(*api.Service)
	if !ipIsAllocated(t, storage.serviceIPAllocatorsByFamily[storage.defaultServiceIPFamily], svc.Spec.ClusterIP) {
		t.Errorf("expected IP to be allocated")
	}
	if !portIsAllocated(t, storage.serviceNodePorts, svc.Spec.Ports[0].NodePort) {
		t.Errorf("expected NodePort to be allocated")
	}

	new3 := svc.DeepCopy()
	svctest.SetTypeExternalName(new3)
	_, _, err = storage.Update(ctx, svc.Name, rest.DefaultUpdatedObjectInfo(new3),
		rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if !portIsAllocated(t, storage.serviceNodePorts, svc.Spec.Ports[0].NodePort) {
		t.Errorf("unexpected side effect: NodePort unallocated")
	}

	// Test dry run update request remove cluster ip
	obj, err = storage.Create(ctx, svctest.MakeService("foo3", svctest.SetClusterIPs("1.2.3.4")), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("expected no error: %v", err)
	}
	svc = obj.(*api.Service)

	new4 := svc.DeepCopy()
	svctest.SetTypeExternalName(new4)
	_, _, err = storage.Update(ctx, svc.Name, rest.DefaultUpdatedObjectInfo(new4),
		rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("expected no error: %v", err)
	}
	if !ipIsAllocated(t, storage.serviceIPAllocatorsByFamily[storage.defaultServiceIPFamily], svc.Spec.ClusterIP) {
		t.Errorf("unexpected side effect: ip unallocated")
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

func TestServiceRegistryLoadBalancerService(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	svc := svctest.MakeService("foo", svctest.SetTypeLoadBalancer)
	_, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Errorf("Failed to create service: %#v", err)
	}
	srv, err := getService(storage, ctx, svc.Name, &metav1.GetOptions{})
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
}

func TestAllocateLoadBalancerNodePorts(t *testing.T) {
	testcases := []struct {
		name                 string
		svc                  *api.Service
		expectNodePorts      bool
		allocateNodePortGate bool
		expectError          bool
	}{{
		name: "allocate false, gate on, not specified",
		svc: svctest.MakeService("alloc-false",
			svctest.SetTypeLoadBalancer,
			svctest.SetAllocateLoadBalancerNodePorts(false)),
		expectNodePorts:      false,
		allocateNodePortGate: true,
	}, {
		name: "allocate true, gate on, not specified",
		svc: svctest.MakeService("alloc-true",
			svctest.SetTypeLoadBalancer,
			svctest.SetAllocateLoadBalancerNodePorts(true)),
		expectNodePorts:      true,
		allocateNodePortGate: true,
	}, {
		name: "allocate false, gate on, port specified",
		svc: svctest.MakeService("alloc-false-specific",
			svctest.SetTypeLoadBalancer,
			svctest.SetNodePorts(30000),
			svctest.SetAllocateLoadBalancerNodePorts(false)),
		expectNodePorts:      true,
		allocateNodePortGate: true,
	}, {
		name: "allocate true, gate on, port specified",
		svc: svctest.MakeService("alloc-true-specific",
			svctest.SetTypeLoadBalancer,
			svctest.SetNodePorts(30000),
			svctest.SetAllocateLoadBalancerNodePorts(true)),
		expectNodePorts:      true,
		allocateNodePortGate: true,
	}, {
		name: "allocate nil, gate off",
		svc: svctest.MakeService("alloc-nil",
			svctest.SetTypeLoadBalancer,
			func(s *api.Service) {
				s.Spec.AllocateLoadBalancerNodePorts = nil
			}),
		expectNodePorts:      true,
		allocateNodePortGate: false,
	}, {
		name: "allocate false, gate off",
		svc: svctest.MakeService("alloc-false",
			svctest.SetTypeLoadBalancer,
			svctest.SetAllocateLoadBalancerNodePorts(false)),
		expectNodePorts:      true,
		allocateNodePortGate: false,
	}, {
		name: "allocate true, gate off",
		svc: svctest.MakeService("alloc-true",
			svctest.SetTypeLoadBalancer,
			svctest.SetAllocateLoadBalancerNodePorts(true)),
		expectNodePorts:      true,
		allocateNodePortGate: false,
	}}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := genericapirequest.NewDefaultContext()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ServiceLBNodePortControl, tc.allocateNodePortGate)()

			storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
			defer server.Terminate(t)

			_, err := storage.Create(ctx, tc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				if tc.expectError {
					return
				}
				t.Errorf("%s; Failed to create service: %#v", tc.name, err)
			}
			srv, err := getService(storage, ctx, tc.svc.Name, &metav1.GetOptions{})
			if err != nil {
				t.Errorf("%s; Unexpected error: %v", tc.name, err)
			}
			if srv == nil {
				t.Fatalf("%s; Failed to find service: %s", tc.name, tc.svc.Name)
			}
			serviceNodePorts := collectServiceNodePorts(srv)
			if (len(serviceNodePorts) != 0) != tc.expectNodePorts {
				t.Errorf("%s; Allocated NodePorts not as expected", tc.name)
			}
		})
	}
}

func TestServiceRegistryDelete(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	svc := svctest.MakeService("foo")
	_, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, _, err = storage.Delete(ctx, svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestServiceRegistryDeleteDryRun(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)

	// Test dry run delete request with cluster ip
	svc := svctest.MakeService("foo")
	obj, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	createdSvc := obj.(*api.Service)
	if createdSvc.Spec.ClusterIP == "" {
		t.Fatalf("expected ClusterIP to be set")
	}
	if !ipIsAllocated(t, storage.serviceIPAllocatorsByFamily[storage.defaultServiceIPFamily], createdSvc.Spec.ClusterIP) {
		t.Errorf("expected ClusterIP to be allocated")
	}
	_, _, err = storage.Delete(ctx, svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if !ipIsAllocated(t, storage.serviceIPAllocatorsByFamily[storage.defaultServiceIPFamily], createdSvc.Spec.ClusterIP) {
		t.Errorf("unexpected side effect: ip unallocated")
	}

	// Test dry run delete request with node port
	svc = svctest.MakeService("foo2", svctest.SetTypeNodePort)
	obj, err = storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	createdSvc = obj.(*api.Service)
	if createdSvc.Spec.Ports[0].NodePort == 0 {
		t.Fatalf("expected NodePort to be set")
	}
	if !portIsAllocated(t, storage.serviceNodePorts, createdSvc.Spec.Ports[0].NodePort) {
		t.Errorf("expected NodePort to be allocated")
	}

	isValidClusterIPFields(t, storage, svc, svc)

	_, _, err = storage.Delete(ctx, svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if !portIsAllocated(t, storage.serviceNodePorts, createdSvc.Spec.Ports[0].NodePort) {
		t.Errorf("unexpected side effect: NodePort unallocated")
	}
}

func TestDualStackServiceRegistryDeleteDryRun(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	// dry run for non dualstack
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, true)()
	dualstack_storage, dualstack_server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol})
	defer dualstack_server.Terminate(t)
	// Test dry run delete request with cluster ip
	dualstack_svc := svctest.MakeService("foo",
		svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
		svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
		svctest.SetClusterIPs("2000:0:0:0:0:0:0:1", "1.2.3.4"))

	_, err := dualstack_storage.Create(ctx, dualstack_svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	isValidClusterIPFields(t, dualstack_storage, dualstack_svc, dualstack_svc)
	_, _, err = dualstack_storage.Delete(ctx, dualstack_svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	for i, family := range dualstack_svc.Spec.IPFamilies {
		if !ipIsAllocated(t, dualstack_storage.serviceIPAllocatorsByFamily[family], dualstack_svc.Spec.ClusterIPs[i]) {
			t.Errorf("unexpected side effect: ip unallocated %v", dualstack_svc.Spec.ClusterIPs[i])
		}
	}
}

func TestServiceRegistryDeleteExternalName(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	svc := svctest.MakeService("foo", svctest.SetTypeExternalName)
	_, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, _, err = storage.Delete(ctx, svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
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
	svc2.Spec.AllocateLoadBalancerNodePorts = utilpointer.BoolPtr(true)
	if _, _, err := storage.Update(ctx, svc2.Name, rest.DefaultUpdatedObjectInfo(svc2), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Change port.
	svc3 := svc2.DeepCopy()
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

func TestServiceRegistryGet(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	_, err := storage.Create(ctx, svctest.MakeService("foo"), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating service: %v", err)
	}
	obj, _ := storage.Get(ctx, "foo", &metav1.GetOptions{})
	svc := obj.(*api.Service)
	if e, a := "foo", svc.Name; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
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

func TestServiceRegistryResourceLocation(t *testing.T) {
	pods := []api.Pod{
		makePod("unnamed", "1.2.3.4", "1.2.3.5"),
		makePod("named", "1.2.3.6", "1.2.3.7"),
		makePod("no-endpoints", "9.9.9.9"), // to prove this does not get chosen
	}

	endpoints := []*api.Endpoints{
		epstest.MakeEndpoints("unnamed",
			[]api.EndpointAddress{
				epstest.MakeEndpointAddress("1.2.3.4", "unnamed"),
			},
			[]api.EndpointPort{
				epstest.MakeEndpointPort("", 80),
			}),
		epstest.MakeEndpoints("unnamed2",
			[]api.EndpointAddress{
				epstest.MakeEndpointAddress("1.2.3.5", "unnamed"),
			},
			[]api.EndpointPort{
				epstest.MakeEndpointPort("", 80),
			}),
		epstest.MakeEndpoints("named",
			[]api.EndpointAddress{
				epstest.MakeEndpointAddress("1.2.3.6", "named"),
			},
			[]api.EndpointPort{
				epstest.MakeEndpointPort("p", 80),
				epstest.MakeEndpointPort("q", 81),
			}),
		epstest.MakeEndpoints("no-endpoints", nil, nil), // to prove this does not get chosen
	}

	storage, server := NewTestRESTWithPods(t, endpoints, pods, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)

	ctx := genericapirequest.NewDefaultContext()
	for _, name := range []string{"unnamed", "unnamed2", "no-endpoints"} {
		_, err := storage.Create(ctx,
			svctest.MakeService(name, svctest.SetPorts(
				svctest.MakeServicePort("", 93, intstr.FromInt(80), api.ProtocolTCP))),
			rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("unexpected error creating service %q: %v", name, err)
		}

	}
	_, err := storage.Create(ctx,
		svctest.MakeService("named", svctest.SetPorts(
			svctest.MakeServicePort("p", 93, intstr.FromInt(80), api.ProtocolTCP),
			svctest.MakeServicePort("q", 76, intstr.FromInt(81), api.ProtocolTCP))),
		rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error creating service %q: %v", "named", err)
	}
	redirector := rest.Redirector(storage)

	cases := []struct {
		query  string
		err    bool
		expect string
	}{{
		query:  "unnamed",
		expect: "//1.2.3.4:80",
	}, {
		query:  "unnamed:",
		expect: "//1.2.3.4:80",
	}, {
		query:  "unnamed:93",
		expect: "//1.2.3.4:80",
	}, {
		query:  "http:unnamed:",
		expect: "http://1.2.3.4:80",
	}, {
		query:  "http:unnamed:93",
		expect: "http://1.2.3.4:80",
	}, {
		query: "unnamed:80",
		err:   true,
	}, {
		query:  "unnamed2",
		expect: "//1.2.3.5:80",
	}, {
		query:  "named:p",
		expect: "//1.2.3.6:80",
	}, {
		query:  "named:q",
		expect: "//1.2.3.6:81",
	}, {
		query:  "named:93",
		expect: "//1.2.3.6:80",
	}, {
		query:  "named:76",
		expect: "//1.2.3.6:81",
	}, {
		query:  "http:named:p",
		expect: "http://1.2.3.6:80",
	}, {
		query:  "http:named:q",
		expect: "http://1.2.3.6:81",
	}, {
		query: "named:bad",
		err:   true,
	}, {
		query: "no-endpoints",
		err:   true,
	}, {
		query: "non-existent",
		err:   true,
	}}
	for _, tc := range cases {
		t.Run(tc.query, func(t *testing.T) {
			location, _, err := redirector.ResourceLocation(ctx, tc.query)
			if tc.err == false && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.err == true && err == nil {
				t.Fatalf("unexpected success")
			}
			if !tc.err {
				if location == nil {
					t.Errorf("unexpected location: %v", location)
				}
				if e, a := tc.expect, location.String(); e != a {
					t.Errorf("expected %q, but got %q", e, a)
				}
			}
		})
	}
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

func TestServiceRegistryIPAllocation(t *testing.T) {
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)

	svc1 := svctest.MakeService("foo")
	ctx := genericapirequest.NewDefaultContext()
	obj, err := storage.Create(ctx, svc1, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating service: %v", err)
	}
	createdSvc1 := obj.(*api.Service)
	if createdSvc1.Name != "foo" {
		t.Errorf("Expected foo, but got %v", createdSvc1.Name)
	}
	if !makeIPNet(t).Contains(net.ParseIP(createdSvc1.Spec.ClusterIPs[0])) {
		t.Errorf("Unexpected ClusterIP: %s", createdSvc1.Spec.ClusterIPs[0])
	}

	svc2 := svctest.MakeService("bar")
	ctx = genericapirequest.NewDefaultContext()
	obj, err = storage.Create(ctx, svc2, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating service: %v", err)
	}
	createdSvc2 := obj.(*api.Service)
	if createdSvc2.Name != "bar" {
		t.Errorf("Expected bar, but got %v", createdSvc2.Name)
	}
	if !makeIPNet(t).Contains(net.ParseIP(createdSvc2.Spec.ClusterIPs[0])) {
		t.Errorf("Unexpected ClusterIP: %s", createdSvc2.Spec.ClusterIPs[0])
	}

	testIPs := []string{"1.2.3.93", "1.2.3.94", "1.2.3.95", "1.2.3.96"}
	testIP := "not-an-ip"
	for _, ip := range testIPs {
		if !ipIsAllocated(t, storage.serviceIPAllocatorsByFamily[storage.defaultServiceIPFamily].(*ipallocator.Range), ip) {
			testIP = ip
			break
		}
	}

	svc3 := svctest.MakeService("qux", svctest.SetClusterIPs(testIP))
	ctx = genericapirequest.NewDefaultContext()
	obj, err = storage.Create(ctx, svc3, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	createdSvc3 := obj.(*api.Service)
	if createdSvc3.Spec.ClusterIPs[0] != testIP { // specific IP
		t.Errorf("Unexpected ClusterIP: %s", createdSvc3.Spec.ClusterIPs[0])
	}
}

func TestServiceRegistryIPReallocation(t *testing.T) {
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)

	svc1 := svctest.MakeService("foo")
	ctx := genericapirequest.NewDefaultContext()
	obj, err := storage.Create(ctx, svc1, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating service: %v", err)
	}
	createdSvc1 := obj.(*api.Service)
	if createdSvc1.Name != "foo" {
		t.Errorf("Expected foo, but got %v", createdSvc1.Name)
	}
	if !makeIPNet(t).Contains(net.ParseIP(createdSvc1.Spec.ClusterIPs[0])) {
		t.Errorf("Unexpected ClusterIP: %s", createdSvc1.Spec.ClusterIPs[0])
	}

	_, _, err = storage.Delete(ctx, createdSvc1.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
	if err != nil {
		t.Errorf("Unexpected error deleting service: %v", err)
	}

	svc2 := svctest.MakeService("bar", svctest.SetClusterIPs(svc1.Spec.ClusterIP))
	ctx = genericapirequest.NewDefaultContext()
	obj, err = storage.Create(ctx, svc2, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating service: %v", err)
	}
	createdSvc2 := obj.(*api.Service)
	if createdSvc2.Name != "bar" {
		t.Errorf("Expected bar, but got %v", createdSvc2.Name)
	}
	if !makeIPNet(t).Contains(net.ParseIP(createdSvc2.Spec.ClusterIPs[0])) {
		t.Errorf("Unexpected ClusterIP: %s", createdSvc2.Spec.ClusterIPs[0])
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
	if !makeIPNet(t).Contains(net.ParseIP(createdService.Spec.ClusterIPs[0])) {
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
		if !ipIsAllocated(t, storage.serviceIPAllocatorsByFamily[storage.defaultServiceIPFamily].(*ipallocator.Range), ip) {
			testIP = ip
			break
		}
	}

	update = createdService.DeepCopy()
	update.Spec.Ports[0].Port = 6503
	update.Spec.ClusterIP = testIP
	update.Spec.ClusterIPs[0] = testIP

	_, _, err = storage.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err == nil || !errors.IsInvalid(err) {
		t.Errorf("Unexpected error type: %v", err)
	}
}

func TestServiceRegistryIPLoadBalancer(t *testing.T) {
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)

	svc := svctest.MakeService("foo", svctest.SetTypeLoadBalancer)
	ctx := genericapirequest.NewDefaultContext()
	createdSvc, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if createdSvc == nil || err != nil {
		t.Errorf("Unexpected failure creating service %v", err)
	}

	createdService := createdSvc.(*api.Service)
	if createdService.Spec.Ports[0].Port != svc.Spec.Ports[0].Port {
		t.Errorf("Expected port %d, but got %v", svc.Spec.Ports[0].Port, createdService.Spec.Ports[0].Port)
	}
	if !makeIPNet(t).Contains(net.ParseIP(createdService.Spec.ClusterIPs[0])) {
		t.Errorf("Unexpected ClusterIP: %s", createdService.Spec.ClusterIPs[0])
	}

	update := createdService.DeepCopy()

	_, _, err = storage.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
}

// Validate allocation of a nodePort when ExternalTrafficPolicy is set to Local
// and type is LoadBalancer.
func TestServiceRegistryExternalTrafficHealthCheckNodePortAllocation(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	svc := svctest.MakeService("external-lb-esipp",
		svctest.SetTypeLoadBalancer,
		func(s *api.Service) {
			s.Spec.ExternalTrafficPolicy = api.ServiceExternalTrafficPolicyTypeLocal
		},
	)
	obj, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if obj == nil || err != nil {
		t.Errorf("Unexpected failure creating service %v", err)
	}

	createdSvc := obj.(*api.Service)
	if !service.NeedsHealthCheck(createdSvc) {
		t.Errorf("Expecting health check needed, returned health check not needed instead")
	}
	port := createdSvc.Spec.HealthCheckNodePort
	if port == 0 {
		t.Errorf("Failed to allocate health check node port and set the HealthCheckNodePort")
	}
}

// Validate using the user specified nodePort when ExternalTrafficPolicy is set to Local
// and type is LoadBalancer.
func TestServiceRegistryExternalTrafficHealthCheckNodePortUserAllocation(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	svc := svctest.MakeService("external-lb-esipp",
		svctest.SetTypeLoadBalancer,
		func(s *api.Service) {
			// hard-code NodePort to make sure it doesn't conflict with the healthport.
			// TODO: remove this once http://issue.k8s.io/93922 fixes auto-allocation conflicting with user-specified health check ports
			s.Spec.Ports[0].NodePort = 30500
			s.Spec.ExternalTrafficPolicy = api.ServiceExternalTrafficPolicyTypeLocal
			s.Spec.HealthCheckNodePort = 30501
		},
	)
	obj, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if obj == nil || err != nil {
		t.Fatalf("Unexpected failure creating service :%v", err)
	}

	createdSvc := obj.(*api.Service)
	if !service.NeedsHealthCheck(createdSvc) {
		t.Errorf("Expecting health check needed, returned health check not needed instead")
	}
	port := createdSvc.Spec.HealthCheckNodePort
	if port == 0 {
		t.Errorf("Failed to allocate health check node port and set the HealthCheckNodePort")
	}
	if port != 30501 {
		t.Errorf("Failed to allocate requested nodePort expected %d, got %d", 30501, port)
	}
}

// Validate that the service creation fails when the requested port number is -1.
func TestServiceRegistryExternalTrafficHealthCheckNodePortNegative(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	svc := svctest.MakeService("external-lb-esipp", svctest.SetTypeLoadBalancer, func(s *api.Service) {
		s.Spec.ExternalTrafficPolicy = api.ServiceExternalTrafficPolicyTypeLocal
		s.Spec.HealthCheckNodePort = int32(-1)
	})
	obj, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if obj == nil || err != nil {
		return
	}
	t.Errorf("Unexpected creation of service with invalid HealthCheckNodePort specified")
}

// Validate that the health check nodePort is not allocated when ExternalTrafficPolicy is set to Global.
func TestServiceRegistryExternalTrafficGlobal(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	svc := svctest.MakeService("external-lb-esipp",
		svctest.SetTypeLoadBalancer,
		func(s *api.Service) {
			s.Spec.ExternalTrafficPolicy = api.ServiceExternalTrafficPolicyTypeCluster
		},
	)
	obj, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if obj == nil || err != nil {
		t.Errorf("Unexpected failure creating service %v", err)
	}

	createdSvc := obj.(*api.Service)
	if service.NeedsHealthCheck(createdSvc) {
		t.Errorf("Expecting health check not needed, returned health check needed instead")
	}
	// Make sure the service does not have the health check node port allocated
	port := createdSvc.Spec.HealthCheckNodePort
	if port != 0 {
		t.Errorf("Unexpected allocation of health check node port: %v", port)
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

func TestInitClusterIP(t *testing.T) {
	testCases := []struct {
		name string
		svc  *api.Service

		enableDualStackAllocator bool
		preAllocateClusterIPs    map[api.IPFamily]string
		expectError              bool
		expectedCountIPs         int
		expectedClusterIPs       []string
	}{{
		name:                     "Allocate single stack ClusterIP (v4)",
		svc:                      svctest.MakeService("foo"),
		enableDualStackAllocator: false,
		expectError:              false,
		preAllocateClusterIPs:    nil,
		expectedCountIPs:         1,
	}, {
		name: "Allocate single ClusterIP (v6)",
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilies(api.IPv6Protocol)),
		expectError:              false,
		enableDualStackAllocator: true,
		preAllocateClusterIPs:    nil,
		expectedCountIPs:         1,
	}, {
		name: "Allocate specified ClusterIP (v4)",
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilies(api.IPv4Protocol),
			svctest.SetClusterIPs("1.2.3.4")),
		expectError:              false,
		enableDualStackAllocator: true,
		preAllocateClusterIPs:    nil,
		expectedCountIPs:         1,
		expectedClusterIPs:       []string{"1.2.3.4"},
	}, {
		name: "Allocate specified ClusterIP-v6",
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilies(api.IPv6Protocol),
			svctest.SetClusterIPs("2000:0:0:0:0:0:0:1")),
		expectError:              false,
		enableDualStackAllocator: true,
		expectedCountIPs:         1,
		expectedClusterIPs:       []string{"2000:0:0:0:0:0:0:1"},
	}, {
		name: "Allocate dual stack - on a non dual stack ",
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilies(api.IPv4Protocol)),
		expectError:              false,
		enableDualStackAllocator: false,
		expectedCountIPs:         1,
	}, {
		name: "Allocate dual stack - upgrade - v4, v6",
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
			svctest.SetIPFamilies(api.IPv4Protocol)),
		expectError:              false,
		enableDualStackAllocator: true,
		expectedCountIPs:         2,
	}, {
		name: "Allocate dual stack - upgrade - v4, v6 - specific first IP",
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
			svctest.SetIPFamilies(api.IPv4Protocol),
			svctest.SetClusterIPs("1.2.3.4")),
		expectError:              false,
		enableDualStackAllocator: true,
		expectedCountIPs:         2,
		expectedClusterIPs:       []string{"1.2.3.4"},
	}, {
		name: "Allocate dual stack - upgrade - v6, v4",
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
			svctest.SetIPFamilies(api.IPv6Protocol)),
		expectError:              false,
		enableDualStackAllocator: true,
		expectedCountIPs:         2,
	}, {
		name: "Allocate dual stack - v4, v6 - specific ips",
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
			svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
			svctest.SetClusterIPs("1.2.3.4", "2000:0:0:0:0:0:0:1")),
		expectError:              false,
		enableDualStackAllocator: true,
		expectedCountIPs:         2,
		expectedClusterIPs:       []string{"1.2.3.4", "2000:0:0:0:0:0:0:1"},
	}, {
		name: "Allocate dual stack - upgrade - v6, v4 - specific ips",
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
			svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
			svctest.SetClusterIPs("2000:0:0:0:0:0:0:1", "1.2.3.4")),
		expectError:              false,
		enableDualStackAllocator: true,
		expectedCountIPs:         2,
		expectedClusterIPs:       []string{"2000:0:0:0:0:0:0:1", "1.2.3.4"},
	}, {
		name: "Shouldn't allocate ClusterIP",
		svc: svctest.MakeService("foo",
			svctest.SetClusterIPs("None")),
		expectError:              false,
		enableDualStackAllocator: false,
		expectedCountIPs:         0,
	}, {
		name: "single stack, ip is pre allocated (ipv4)",
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
			svctest.SetIPFamilies(api.IPv4Protocol),
			svctest.SetClusterIPs("1.2.3.4")),
		expectError:              true,
		enableDualStackAllocator: false,
		expectedCountIPs:         0,
		preAllocateClusterIPs:    map[api.IPFamily]string{api.IPv4Protocol: "1.2.3.4"},
	}, {
		name: "single stack, ip is pre allocated (ipv6)",
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
			svctest.SetIPFamilies(api.IPv6Protocol),
			svctest.SetClusterIPs("2000:0:0:0:0:0:0:1")),
		expectError:              true,
		enableDualStackAllocator: true, // ipv6 allocator is always the second one during test
		expectedCountIPs:         0,
		preAllocateClusterIPs:    map[api.IPFamily]string{api.IPv6Protocol: "2000:0:0:0:0:0:0:1"},
	}, {
		name: "Allocate dual stack - upgrade - v6, v4 - specific ips (first ip can't be allocated)",
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
			svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
			svctest.SetClusterIPs("2000:0:0:0:0:0:0:1", "1.2.3.4")),
		expectError:              true,
		enableDualStackAllocator: true,
		expectedCountIPs:         0,
		preAllocateClusterIPs:    map[api.IPFamily]string{api.IPv6Protocol: "2000:0:0:0:0:0:0:1"},
	}, {
		name: "Allocate dual stack - upgrade - v6, v4 - specific ips (second ip can't be allocated)",
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
			svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
			svctest.SetClusterIPs("2000:0:0:0:0:0:0:1", "1.2.3.4")),
		expectError:              true,
		enableDualStackAllocator: true,
		expectedCountIPs:         0,
		preAllocateClusterIPs:    map[api.IPFamily]string{api.IPv4Protocol: "1.2.3.4"},
	}}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, true)()

			// create the rest stack
			families := []api.IPFamily{api.IPv4Protocol}
			if test.enableDualStackAllocator {
				families = append(families, api.IPv6Protocol)
			}
			storage, server := NewTestREST(t, families)
			defer server.Terminate(t)

			copySvc := test.svc.DeepCopy()

			// pre allocate ips if any
			for family, ip := range test.preAllocateClusterIPs {
				allocator, ok := storage.serviceIPAllocatorsByFamily[family]
				if !ok {
					t.Fatalf("test is incorrect, allocator does not exist on rest")
				}
				if err := allocator.Allocate(net.ParseIP(ip)); err != nil {
					t.Fatalf("test is incorrect, allocator failed to pre allocate IP with error:%v", err)
				}
			}
			ctx := genericapirequest.NewDefaultContext()
			createdSvc, err := storage.Create(ctx, test.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if test.expectError && err == nil {
				t.Fatalf("error was expected, but no error was returned")
			}

			if !test.expectError && err != nil {
				t.Fatalf("error was not expected, but got error %v", err)
			}

			if err != nil {
				return // no more testing needed for this case
			}
			newSvc := createdSvc.(*api.Service)
			isValidClusterIPFields(t, storage, copySvc, newSvc)

			// if it has ips then let us check they have been correctly allocated
			if newSvc.Spec.ClusterIPs[0] != api.ClusterIPNone {
				for _, ip := range newSvc.Spec.ClusterIPs {
					family := api.IPv4Protocol
					if netutil.IsIPv6String(ip) {
						family = api.IPv6Protocol
					}
					allocator := storage.serviceIPAllocatorsByFamily[family]
					if !ipIsAllocated(t, allocator, ip) {
						t.Fatalf("expected ip:%v to be allocated by %v allocator. it was not", ip, family)
					}
				}
			}

			allocatedIPs := 0
			for _, ip := range newSvc.Spec.ClusterIPs {
				if ip != api.ClusterIPNone {
					allocatedIPs++
				}
			}

			if allocatedIPs != test.expectedCountIPs {
				t.Fatalf("incorrect allocated IP count expected %v got %v", test.expectedCountIPs, allocatedIPs)
			}

			for i, ip := range test.expectedClusterIPs {
				if i >= len(newSvc.Spec.ClusterIPs) {
					t.Fatalf("incorrect ips were assigne. expected to find %+v in %+v",
						ip, newSvc.Spec.ClusterIPs)
				}

				if ip != newSvc.Spec.ClusterIPs[i] {
					t.Fatalf("incorrect ips were assigne. expected to find %+v == %+v at position %v",
						ip, newSvc.Spec.ClusterIPs[i], i)
				}
			}

			// the following apply only on dual stack
			if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) {
				return
			}

			shouldUpgrade := len(newSvc.Spec.IPFamilies) == 2 && *(newSvc.Spec.IPFamilyPolicy) != api.IPFamilyPolicySingleStack && len(storage.serviceIPAllocatorsByFamily) == 2
			if shouldUpgrade && len(newSvc.Spec.ClusterIPs) < 2 {
				t.Fatalf("Service should have been upgraded %+v", newSvc)
			}

			if !shouldUpgrade && len(newSvc.Spec.ClusterIPs) > 1 {
				t.Fatalf("Service should not have been upgraded %+v", newSvc)
			}

		})
	}
}

func TestInitNodePorts(t *testing.T) {
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	nodePortOp := portallocator.StartOperation(storage.serviceNodePorts, false)

	testCases := []struct {
		name                     string
		service                  *api.Service
		expectSpecifiedNodePorts []int
	}{{
		name:                     "Service doesn't have specified NodePort",
		service:                  svctest.MakeService("foo", svctest.SetTypeNodePort),
		expectSpecifiedNodePorts: []int{},
	}, {
		name: "Service has one specified NodePort",
		service: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("port-tcp", 53, intstr.FromInt(6502), api.ProtocolTCP)),
			svctest.SetNodePorts(30053)),
		expectSpecifiedNodePorts: []int{30053},
	}, {
		name: "Service has two same ports with different protocols and specifies same NodePorts",
		service: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("port-tcp", 53, intstr.FromInt(6502), api.ProtocolTCP),
				svctest.MakeServicePort("port-udp", 53, intstr.FromInt(6502), api.ProtocolUDP)),
			svctest.SetNodePorts(30054, 30054)),
		expectSpecifiedNodePorts: []int{30054, 30054},
	}, {
		name: "Service has two same ports with different protocols and specifies different NodePorts",
		service: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("port-tcp", 53, intstr.FromInt(6502), api.ProtocolTCP),
				svctest.MakeServicePort("port-udp", 53, intstr.FromInt(6502), api.ProtocolUDP)),
			svctest.SetNodePorts(30055, 30056)),
		expectSpecifiedNodePorts: []int{30055, 30056},
	}, {
		name: "Service has two different ports with different protocols and specifies different NodePorts",
		service: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("port-tcp", 53, intstr.FromInt(6502), api.ProtocolTCP),
				svctest.MakeServicePort("port-udp", 54, intstr.FromInt(6502), api.ProtocolUDP)),
			svctest.SetNodePorts(30057, 30058)),
		expectSpecifiedNodePorts: []int{30057, 30058},
	}, {
		name: "Service has two same ports with different protocols but only specifies one NodePort",
		service: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("port-tcp", 53, intstr.FromInt(6502), api.ProtocolTCP),
				svctest.MakeServicePort("port-udp", 53, intstr.FromInt(6502), api.ProtocolUDP)),
			svctest.SetNodePorts(30059)),
		expectSpecifiedNodePorts: []int{30059, 30059},
	}}

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
	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	nodePortOp := portallocator.StartOperation(storage.serviceNodePorts, false)

	testCases := []struct {
		name                     string
		oldService               *api.Service
		newService               *api.Service
		expectSpecifiedNodePorts []int
	}{{
		name: "Old service and new service have the same NodePort",
		oldService: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("", 6502, intstr.FromInt(6502), api.ProtocolTCP)),
			svctest.SetNodePorts(30053)),
		newService: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("", 6502, intstr.FromInt(6502), api.ProtocolTCP)),
			svctest.SetNodePorts(30053)),
		expectSpecifiedNodePorts: []int{30053},
	}, {
		name: "Old service has more NodePorts than new service has",
		oldService: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("port-tcp", 53, intstr.FromInt(6502), api.ProtocolTCP),
				svctest.MakeServicePort("port-udp", 53, intstr.FromInt(6502), api.ProtocolUDP)),
			svctest.SetNodePorts(30053, 30053)),
		newService: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("port-tcp", 53, intstr.FromInt(6502), api.ProtocolTCP)),
			svctest.SetNodePorts(30053)),
		expectSpecifiedNodePorts: []int{30053},
	}, {
		name: "Change protocol of ServicePort without changing NodePort",
		oldService: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("port-tcp", 53, intstr.FromInt(6502), api.ProtocolTCP)),
			svctest.SetNodePorts(30053)),
		newService: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("port-udp", 53, intstr.FromInt(6502), api.ProtocolUDP)),
			svctest.SetNodePorts(30053)),
		expectSpecifiedNodePorts: []int{30053},
	}, {
		name: "Should allocate NodePort when changing service type to NodePort",
		oldService: svctest.MakeService("foo",
			svctest.SetTypeClusterIP,
			svctest.SetPorts(
				svctest.MakeServicePort("", 6502, intstr.FromInt(6502), api.ProtocolUDP))),
		newService: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("", 6502, intstr.FromInt(6502), api.ProtocolUDP))),
		expectSpecifiedNodePorts: []int{},
	}, {
		name: "Add new ServicePort with a different protocol without changing port numbers",
		oldService: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("port-tcp", 53, intstr.FromInt(6502), api.ProtocolTCP)),
			svctest.SetNodePorts(30053)),
		newService: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("port-tcp", 53, intstr.FromInt(6502), api.ProtocolTCP),
				svctest.MakeServicePort("port-udp", 53, intstr.FromInt(6502), api.ProtocolUDP)),
			svctest.SetNodePorts(30053, 30053)),
		expectSpecifiedNodePorts: []int{30053, 30053},
	}, {
		name: "Change service type from ClusterIP to NodePort with same NodePort number but different protocols",
		oldService: svctest.MakeService("foo",
			svctest.SetTypeClusterIP,
			svctest.SetPorts(
				svctest.MakeServicePort("", 53, intstr.FromInt(6502), api.ProtocolTCP))),
		newService: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("port-tcp", 53, intstr.FromInt(6502), api.ProtocolTCP),
				svctest.MakeServicePort("port-udp", 53, intstr.FromInt(6502), api.ProtocolUDP)),
			svctest.SetNodePorts(30053, 30053)),
		expectSpecifiedNodePorts: []int{30053, 30053},
	}}

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
				alloc := storage.serviceIPAllocatorsByFamily[family]
				if err := alloc.Allocate(net.ParseIP(ip)); err != nil {
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

			shouldUpgrade := len(createdSvc.Spec.IPFamilies) == 2 && *(createdSvc.Spec.IPFamilyPolicy) != api.IPFamilyPolicySingleStack && len(storage.serviceIPAllocatorsByFamily) == 2
			if shouldUpgrade && len(updatedSvc.Spec.ClusterIPs) < 2 {
				t.Fatalf("Service should have been upgraded %+v", createdSvc)
			}

			if !shouldUpgrade && len(updatedSvc.Spec.ClusterIPs) > 1 {
				t.Fatalf("Service should not have been upgraded %+v", createdSvc)
			}

			// make sure that ips were allocated, correctly
			for i, family := range updatedSvc.Spec.IPFamilies {
				ip := updatedSvc.Spec.ClusterIPs[i]
				allocator := storage.serviceIPAllocatorsByFamily[family]
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
				allocator := storage.serviceIPAllocatorsByFamily[releasedIPFamily]

				if ipIsAllocated(t, allocator, releasedIP) {
					t.Fatalf("expected ip:%v to be released by %v allocator. it was not", releasedIP, releasedIPFamily)
				}
			}
		})
	}
}

func TestDefaultingValidation(t *testing.T) {
	singleStack := api.IPFamilyPolicySingleStack
	preferDualStack := api.IPFamilyPolicyPreferDualStack
	requireDualStack := api.IPFamilyPolicyRequireDualStack

	// takes in REST and modify it for a specific config
	fnMakeSingleStackIPv4Allocator := func(rest *REST) {
		rest.defaultServiceIPFamily = api.IPv4Protocol
		rest.serviceIPAllocatorsByFamily = map[api.IPFamily]ipallocator.Interface{api.IPv4Protocol: rest.serviceIPAllocatorsByFamily[api.IPv4Protocol]}
	}

	fnMakeSingleStackIPv6Allocator := func(rest *REST) {
		rest.defaultServiceIPFamily = api.IPv6Protocol
		rest.serviceIPAllocatorsByFamily = map[api.IPFamily]ipallocator.Interface{api.IPv6Protocol: rest.serviceIPAllocatorsByFamily[api.IPv6Protocol]}
	}

	fnMakeDualStackStackIPv4IPv6Allocator := func(rest *REST) {
		rest.defaultServiceIPFamily = api.IPv4Protocol
		rest.serviceIPAllocatorsByFamily = map[api.IPFamily]ipallocator.Interface{
			api.IPv6Protocol: rest.serviceIPAllocatorsByFamily[api.IPv6Protocol],
			api.IPv4Protocol: rest.serviceIPAllocatorsByFamily[api.IPv4Protocol],
		}
	}

	fnMakeDualStackStackIPv6IPv4Allocator := func(rest *REST) {
		rest.defaultServiceIPFamily = api.IPv6Protocol
		rest.serviceIPAllocatorsByFamily = map[api.IPFamily]ipallocator.Interface{
			api.IPv6Protocol: rest.serviceIPAllocatorsByFamily[api.IPv6Protocol],
			api.IPv4Protocol: rest.serviceIPAllocatorsByFamily[api.IPv4Protocol],
		}
	}

	testCases := []struct {
		name       string
		modifyRest func(rest *REST)
		oldSvc     *api.Service
		svc        *api.Service

		expectedIPFamilyPolicy *api.IPFamilyPolicyType
		expectedIPFamilies     []api.IPFamily
		expectError            bool
	}{
		////////////////////////////
		// cluster configured as single stack v4
		////////////////////////////
		{
			name:                   "[singlestack:v4] set: externalname on a single stack - v4",
			modifyRest:             fnMakeSingleStackIPv4Allocator,
			svc:                    svctest.MakeService("foo", svctest.SetTypeExternalName),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            false,
		},
		{
			name:                   "[singlestack:v4] set: nothing",
			modifyRest:             fnMakeSingleStackIPv4Allocator,
			svc:                    svctest.MakeService("foo"),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},

		{
			name:       "[singlestack:v4] set: v4Cluster IPSet",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: v4IPFamilySet",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: v4IPFamilySet",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("10.0.0.4"),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: PreferDualStack",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: PreferDualStack + v4ClusterIPSet",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: PreferDualStack + v4ClusterIPSet + v4FamilySet",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: v6IPSet",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("2000::1")),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v4] set: v6IPFamily",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v4] set: RequireDualStack",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v4] set: RequireDualStack + family",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		// selector less
		{
			name:       "[singlestack:v4] set: selectorless, families are ignored",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("None"),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: selectorless, no families",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("None"),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: selectorless, user selected",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("None"),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: selectorless, user set to preferDualStack",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("None"),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		// tests incorrect setting for IPFamilyPolicy
		{
			name:       "[singlestack:v4] set: multifamily set to preferDualStack",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v4] set: multifamily set to singleStack",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v4] set: mult clusterips set to preferDualStack",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("1.1.1.1", "2001::1"),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v4] set: multi clusterips set to singleStack",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("1.1.1.1", "2001::1"),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},

		////////////////////////////
		// cluster configured as single stack v6
		////////////////////////////
		{
			name:                   "[singlestack:v6] set: externalname on a single stack - v4",
			modifyRest:             fnMakeSingleStackIPv6Allocator,
			svc:                    svctest.MakeService("foo", svctest.SetTypeExternalName),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            false,
		},
		{
			name:                   "[singlestack:v6] set: nothing",
			modifyRest:             fnMakeSingleStackIPv6Allocator,
			svc:                    svctest.MakeService("foo"),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: v6Cluster IPSet",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("2000::1")),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: v4IPFamilySet",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: v6IPFamilySet",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("2000::1"),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: PreferDualStack",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: PreferDualStack + v6ClusterIPSet",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetClusterIPs("2000::1")),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: PreferDualStack + v6ClusterIPSet + v6FamilySet",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1")),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: v4IPSet",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("10.0.0.10")),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v6] set: v4IPFamily",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v6] set: RequireDualStack (on single stack ipv6 cluster)",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v6] set: RequireDualStack + family (on single stack ipv6 cluster)",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},

		// selector less
		{
			name:       "[singlestack:v6] set: selectorless, families are ignored",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("None"),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: selectorless, no families",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("None"),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: selectorless, user selected",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("None"),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: selectorless, user set to preferDualStack",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("None"),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		// tests incorrect setting for IPFamilyPolicy
		{
			name:       "[singlestack:v6] set: multifamily set to preferDualStack",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v6] set: multifamily set to singleStack",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v6] set: mult clusterips set to preferDualStack",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("1.1.1.1", "2001::1"),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v6] set: multi clusterips set to singleStack",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("1.1.1.1", "2001::1"),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},

		////////////////////////////
		// cluster configured as dual stack v4,6
		////////////////////////////
		{
			name:                   "[dualstack:v4,v6] set: externalname on a dual stack - v4,v6",
			modifyRest:             fnMakeDualStackStackIPv4IPv6Allocator,
			svc:                    svctest.MakeService("foo", svctest.SetTypeExternalName),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            false,
		},
		{
			name:                   "[dualstack:v4,v6] set: nothing",
			modifyRest:             fnMakeDualStackStackIPv4IPv6Allocator,
			svc:                    svctest.MakeService("foo"),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},

		{
			name:       "[dualstack:v4,v6] set: v4ClusterIPSet",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: v4IPFamilySet",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: v4IPFamilySet",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("10.0.0.4"),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: v6ClusterIPSet",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("2000::1")),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: v6IPFamilySet",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: v6IPFamilySet",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("2000::1"),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		// prefer dual stack
		{
			name:       "[dualstack:v4,v6] set: PreferDualStack.",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: PreferDualStack + v4ClusterIPSet",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: PreferDualStack + v4ClusterIPSet + v4FamilySet",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		// require dual stack
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + family v4",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + family v6",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},

		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + family +ip v4",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.10"),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			//
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + family +ip v6",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1"),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + ip v6",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1")),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + ip v4",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.10")),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + ips",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.10", "2000::1")),
			//
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + ips",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.10")),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + ips + families v6,v4",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.10"),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + ips + families v4,v6",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.10", "2000::1"),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: selectorless, no families",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("None"),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,6] set: selectorless, user selected",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("None"),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,6] set: selectorless, user set to prefer",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("None"),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		// tests incorrect setting for IPFamilyPolicy
		{
			name:       "[duakstack:v4,6] set: multifamily set to preferDualStack",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,6] set: multifamily set to singleStack",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[dualstack:v4,6] set: mult clusterips set to preferDualStack",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("1.1.1.1", "2001::1"),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,6] set: multi clusterips set to singleStack",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("1.1.1.1", "2001::1"),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},

		////////////////////////////
		// cluster configured as dual stack v6,4
		////////////////////////////
		{
			name:                   "[dualstack:v6,v4] set: externalname on a dual stack - v6,v4",
			modifyRest:             fnMakeDualStackStackIPv6IPv4Allocator,
			svc:                    svctest.MakeService("foo", svctest.SetTypeExternalName),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            false,
		},
		{
			name:                   "[dualstack:v6,v4] set: nothing",
			modifyRest:             fnMakeDualStackStackIPv6IPv4Allocator,
			svc:                    svctest.MakeService("foo"),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: v4ClusterIPSet",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: v4IPFamilySet",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv4Protocol)),
			//
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: v4IPFamilySet",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("10.0.0.4"),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: v6ClusterIPSet",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("2000::1")),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: v6IPFamilySet",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: v6IPFamilySet",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("2000::1"),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		// prefer dual stack
		{
			name:       "[dualstack:v6,v4] set: PreferDualStack.",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: PreferDualStack + v4ClusterIPSet",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: PreferDualStack + v4ClusterIPSet + v4FamilySet",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		// require dual stack
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + family v4",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + family v6",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + family +ip v4",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.10"),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + family +ip v6",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1"),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + ip v6",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1")),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + ip v4",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.10")),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + ip v4",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.10")),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + ips",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.10", "2000::1")),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + ips",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.10")),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + ips + families v6,v4",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.10"),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + ips + families v4,v6",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.10", "2000::1"),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: selectorless, no families",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("None"),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: selectorless, user selected",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("None"),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),

			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: selectorless, user set to prefer",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("None"),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),

			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		// tests incorrect setting for IPFamilyPolicy
		{
			name:       "[duakstack:v6,5] set: multifamily set to preferDualStack",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,6] set: multifamily set to singleStack",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[dualstack:v6,4] set: mult clusterips set to preferDualStack",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("1.1.1.1", "2001::1"),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,4] set: multi clusterips set to singleStack",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("1.1.1.1", "2001::1"),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},

		// preferDualStack services should not be updated
		// to match cluster config if the user didn't change any
		// ClusterIPs related fields
		{
			name:       "unchanged preferDualStack-1-ClusterUpgraded",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			oldSvc: svctest.MakeService("foo",
				svctest.SetClusterIPs("1.1.1.1"),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),

			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("1.1.1.1"),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},

		{
			name:       "unchanged preferDualStack-2-ClusterDowngraded",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			oldSvc: svctest.MakeService("foo",
				svctest.SetClusterIPs("1.1.1.1", "2001::1"),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),

			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("1.1.1.1", "2001::1"),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},

		{
			name:       "changed preferDualStack-1 (cluster upgraded)",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			oldSvc: svctest.MakeService("foo",
				svctest.SetClusterIPs("1.1.1.1"),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),

			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},

		{
			name:       "changed preferDualStack-2-ClusterDowngraded",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			oldSvc: svctest.MakeService("foo",
				svctest.SetClusterIPs("1.1.1.1", "2001::1"),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),

			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("1.1.1.1"),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
	}

	// This func only runs when feature gate is on
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, true)()

	storage, server := NewTestREST(t, []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol})
	defer server.Terminate(t)

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {

			// reset to defaults
			fnMakeDualStackStackIPv4IPv6Allocator(storage)
			// optionally apply test-specific changes
			if testCase.modifyRest != nil {
				testCase.modifyRest(storage)
			}

			err := storage.tryDefaultValidateServiceClusterIPFields(testCase.oldSvc, testCase.svc)
			if err != nil && !testCase.expectError {
				t.Fatalf("error %v was not expected", err)
			}

			if err == nil && testCase.expectError {
				t.Fatalf("error was expected, but no error returned")
			}

			if err != nil {
				t.Logf("test concluded successfully with terminal error %v", err)
				return
			}

			// IPFamily Policy
			if (testCase.expectedIPFamilyPolicy == nil && testCase.svc.Spec.IPFamilyPolicy != nil) ||
				(testCase.expectedIPFamilyPolicy != nil && testCase.svc.Spec.IPFamilyPolicy == nil) {
				t.Fatalf("ipFamilyPolicy expected:%v got %v", testCase.expectedIPFamilyPolicy, testCase.svc.Spec.IPFamilyPolicy)
			}

			if testCase.expectedIPFamilyPolicy != nil {
				if *testCase.expectedIPFamilyPolicy != *testCase.svc.Spec.IPFamilyPolicy {
					t.Fatalf("ipFamilyPolicy expected:%s got %s", *testCase.expectedIPFamilyPolicy, *testCase.svc.Spec.IPFamilyPolicy)
				}
			}

			if len(testCase.expectedIPFamilies) != len(testCase.svc.Spec.IPFamilies) {
				t.Fatalf("expected len of IPFamilies %v got %v", len(testCase.expectedIPFamilies), len(testCase.svc.Spec.IPFamilies))
			}

			// match families
			for i, family := range testCase.expectedIPFamilies {
				if testCase.svc.Spec.IPFamilies[i] != family {
					t.Fatalf("expected ip family %v at %v got %v", family, i, testCase.svc.Spec.IPFamilies)
				}
			}
		})
	}
}

// validates that the service created, updated by REST
// has correct ClusterIPs related fields
func isValidClusterIPFields(t *testing.T, storage *REST, pre *api.Service, post *api.Service) {
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
		isIPv6 := netutil.IsIPv6String(ip)
		if isIPv6 && post.Spec.IPFamilies[i] != api.IPv6Protocol {
			t.Fatalf("ips does not match assigned families %+v %+v", post.Spec.ClusterIPs, post.Spec.IPFamilies)
		}
	}
}
