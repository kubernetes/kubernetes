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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/util/dryrun"

	"k8s.io/kubernetes/pkg/api/service"
	api "k8s.io/kubernetes/pkg/apis/core"
	endpointstore "k8s.io/kubernetes/pkg/registry/core/endpoint/storage"
	podstore "k8s.io/kubernetes/pkg/registry/core/pod/storage"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"

	netutil "k8s.io/utils/net"
	utilpointer "k8s.io/utils/pointer"
)

var (
	singleStackIPv4 = []api.IPFamily{api.IPv4Protocol}
)

// TODO(wojtek-t): Cleanup this file.
// It is now testing mostly the same things as other resources but
// in a completely different way. We should unify it.

type serviceStorage struct {
	GottenID    string
	UpdatedID   string
	CreatedID   string
	DeletedID   string
	Service     *api.Service
	ServiceList *api.ServiceList
}

func (s *serviceStorage) NamespaceScoped() bool {
	return true
}

func (s *serviceStorage) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	s.GottenID = name
	return s.Service, nil
}

func (s *serviceStorage) GetService(ctx context.Context, name string, options *metav1.GetOptions) (*api.Service, error) {
	return s.Service, nil
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
	s.CreatedID = obj.(metav1.Object).GetName()
	s.Service = svc.DeepCopy()
	s.Service.ResourceVersion = "1"

	if s.ServiceList == nil {
		s.ServiceList = &api.ServiceList{}
	}

	s.ServiceList.Items = append(s.ServiceList.Items, *svc)
	return s.Service.DeepCopy(), nil
}

func (s *serviceStorage) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	obj, err := objInfo.UpdatedObject(ctx, nil)
	if err != nil {
		return nil, false, err
	}
	if !dryrun.IsDryRun(options.DryRun) {
		s.UpdatedID = name
		s.Service = obj.(*api.Service)
	}
	return obj, false, nil
}

func (s *serviceStorage) Delete(ctx context.Context, name string, deleteValidation rest.ValidateObjectFunc, options *metav1.DeleteOptions) (runtime.Object, bool, error) {
	if !dryrun.IsDryRun(options.DryRun) {
		s.DeletedID = name
	}
	return s.Service, false, nil
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

func NewTestREST(t *testing.T, endpoints *api.EndpointsList, ipFamilies []api.IPFamily) (*REST, *serviceStorage, *etcd3testing.EtcdTestServer) {
	return NewTestRESTWithPods(t, endpoints, nil, ipFamilies)
}

func NewTestRESTWithPods(t *testing.T, endpoints *api.EndpointsList, pods *api.PodList, ipFamilies []api.IPFamily) (*REST, *serviceStorage, *etcd3testing.EtcdTestServer) {
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

	return rest, serviceStorage, server
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

// For clarity.
type serviceTweak func(*api.Service)

// This is used to clarify test sites.
func makeService(name string, tweaks ...serviceTweak) *api.Service {
	// NOTE: Any field that would be populated by defaulting needs to be
	// present and valid here.
	svc := &api.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceDefault,
		},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			SessionAffinity: api.ServiceAffinityNone,
			Ports: []api.ServicePort{{
				Port:       93,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(76),
			}},
		},
	}
	// Default to ClusterIP
	tweakTypeClusterIP(svc)

	for _, tweak := range tweaks {
		tweak(svc)
	}

	return svc
}

func tweakTypeClusterIP(svc *api.Service) {
	svc.Spec.Type = api.ServiceTypeClusterIP
}

func tweakTypeNodePort(svc *api.Service) {
	svc.Spec.Type = api.ServiceTypeNodePort
}

func tweakTypeLoadBalancer(svc *api.Service) {
	svc.Spec.Type = api.ServiceTypeLoadBalancer
}

func tweakTypeExternalName(svc *api.Service) {
	svc.Spec.Type = api.ServiceTypeExternalName
	svc.Spec.ExternalName = "example.com"
	svc.Spec.ExternalTrafficPolicy = ""
	svc.Spec.ClusterIP = ""
	svc.Spec.ClusterIPs = nil
}

func tweakPorts(ports ...api.ServicePort) serviceTweak {
	return func(svc *api.Service) {
		svc.Spec.Ports = ports
	}
}

func makeServicePort(name string, port int, tgtPort intstr.IntOrString) api.ServicePort {
	return api.ServicePort{
		Name:       name,
		Port:       int32(port),
		TargetPort: tgtPort,
		Protocol:   api.ProtocolTCP,
	}
}

func tweakClusterIPs(ips ...string) serviceTweak {
	return func(svc *api.Service) {
		svc.Spec.ClusterIP = ips[0]
		svc.Spec.ClusterIPs = ips
	}
}

func tweakIPFamilies(families ...api.IPFamily) serviceTweak {
	return func(svc *api.Service) {
		svc.Spec.IPFamilies = families
	}
}

func tweakIPFamilyPolicy(policy api.IPFamilyPolicyType) serviceTweak {
	return func(svc *api.Service) {
		svc.Spec.IPFamilyPolicy = &policy
	}
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
	testCases := []struct {
		svc             *api.Service
		name            string
		families        []api.IPFamily
		enableDualStack bool
	}{{
		name:            "Service IPFamily default cluster dualstack:off",
		enableDualStack: false,
		families:        []api.IPFamily{api.IPv4Protocol},
		svc:             makeService("foo"),
	}, {
		name:            "Service IPFamily:v4 dualstack off",
		enableDualStack: false,
		families:        []api.IPFamily{api.IPv4Protocol},
		svc:             makeService("foo", tweakIPFamilies(api.IPv4Protocol)),
	}, {
		name:            "Service IPFamily:v4 dualstack on",
		enableDualStack: true,
		families:        []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		svc:             makeService("foo", tweakIPFamilies(api.IPv4Protocol)),
	}, {
		name:            "Service IPFamily:v6 dualstack on",
		enableDualStack: true,
		families:        []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		svc:             makeService("foo", tweakIPFamilies(api.IPv6Protocol)),
	}}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			storage, registry, server := NewTestREST(t, nil, tc.families)
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
			srv, err := registry.GetService(ctx, tc.svc.Name, &metav1.GetOptions{})
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
		svc:             makeService("foo", tweakClusterIPs("1.2.3.4")),
	}, {
		name:            "v6 service featuregate on but singlestack",
		enableDualStack: true,
		svc: makeService("foo",
			tweakIPFamilies(api.IPv6Protocol),
			tweakClusterIPs("2000::1")),
	}, {
		name:            "dualstack v4,v6 service",
		enableDualStack: true,
		svc: makeService("foo",
			tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
			tweakIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
			tweakClusterIPs("1.2.3.4", "2000::1")),
	}, {
		name:            "dualstack v6,v4 service",
		enableDualStack: true,
		svc: makeService("foo",
			tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
			tweakIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
			tweakClusterIPs("2000::1", "1.2.3.4")),
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, tc.enableDualStack)()

			families := []api.IPFamily{api.IPv4Protocol}
			if tc.enableDualStack {
				families = append(families, api.IPv6Protocol)
			}
			storage, registry, server := NewTestREST(t, nil, families)
			defer server.Terminate(t)

			ctx := genericapirequest.NewDefaultContext()
			_, err := storage.Create(ctx, tc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			for i, family := range tc.svc.Spec.IPFamilies {
				alloc := storage.serviceIPAllocatorsByFamily[family]
				if alloc.Has(net.ParseIP(tc.svc.Spec.ClusterIPs[i])) {
					t.Errorf("unexpected side effect: ip allocated %v", tc.svc.Spec.ClusterIPs[i])
				}
			}

			srv, err := registry.GetService(ctx, tc.svc.Name, &metav1.GetOptions{})
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if srv != nil {
				t.Errorf("unexpected service found: %v", srv)
			}
		})
	}
}

func TestDryRunNodePort(t *testing.T) {
	storage, registry, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)

	// Test dry run create request with a node port
	svc := makeService("foo", tweakTypeNodePort)
	ctx := genericapirequest.NewDefaultContext()

	obj, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	createdSvc := obj.(*api.Service)
	if createdSvc.Spec.Ports[0].NodePort == 0 {
		t.Errorf("expected NodePort value assigned")
	}
	if storage.serviceNodePorts.Has(int(createdSvc.Spec.Ports[0].NodePort)) {
		t.Errorf("unexpected side effect: NodePort allocated")
	}
	srv, err := registry.GetService(ctx, svc.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if srv != nil {
		t.Errorf("unexpected service found: %v", srv)
	}

	// Test dry run create request with multi node port
	svc = makeService("foo",
		tweakTypeNodePort,
		func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Name:       "port-tcp",
				Port:       53,
				NodePort:   30053,
				TargetPort: intstr.FromInt(6503),
				Protocol:   api.ProtocolTCP,
			}, {
				Name:       "port-udp",
				Port:       53,
				NodePort:   30053,
				TargetPort: intstr.FromInt(6503),
				Protocol:   api.ProtocolUDP,
			}}
		})
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
		if storage.serviceNodePorts.Has(int(svc.Spec.Ports[i].NodePort)) {
			t.Errorf("unexpected side effect: NodePort allocated")
		}
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
	svc = makeService("foo",
		tweakTypeNodePort,
		func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Name:       "port-a",
				Port:       53,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6503),
			}, {
				Name:       "port-b",
				Port:       54,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6504),
			}}
		})
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
	storage, registry, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)

	testCases := []struct {
		svc             *api.Service
		name            string
		expectNodePorts []int
	}{{
		svc: makeService("foo1",
			tweakTypeNodePort,
			func(s *api.Service) {
				s.Spec.Ports = []api.ServicePort{{
					Name:       "port-tcp",
					Port:       53,
					NodePort:   30053,
					TargetPort: intstr.FromInt(6503),
					Protocol:   api.ProtocolTCP,
				}, {
					Name:       "port-udp",
					Port:       53,
					NodePort:   30053,
					TargetPort: intstr.FromInt(6503),
					Protocol:   api.ProtocolUDP,
				}}
			}),
		name:            "foo1",
		expectNodePorts: []int{30053, 30053},
	}, {
		svc: makeService("foo2",
			tweakTypeNodePort,
			func(s *api.Service) {
				s.Spec.Ports = []api.ServicePort{{
					Name:       "port-tcp",
					Port:       54,
					TargetPort: intstr.FromInt(6504),
					Protocol:   api.ProtocolTCP,
				}, {
					Name:       "port-udp",
					Port:       54,
					NodePort:   30054,
					TargetPort: intstr.FromInt(6504),
					Protocol:   api.ProtocolUDP,
				}}
			}),
		name:            "foo2",
		expectNodePorts: []int{30054, 30054},
	}, {
		svc: makeService("foo3",
			tweakTypeNodePort,
			func(s *api.Service) {
				s.Spec.Ports = []api.ServicePort{{
					Name:       "port-tcp",
					Port:       55,
					NodePort:   30055,
					TargetPort: intstr.FromInt(6505),
					Protocol:   api.ProtocolTCP,
				}, {
					Name:       "port-udp",
					Port:       55,
					NodePort:   30056,
					TargetPort: intstr.FromInt(6506),
					Protocol:   api.ProtocolUDP,
				}}
			}),
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
	storage, _, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)
	failureCases := map[string]*api.Service{
		"empty ID": makeService(""),
		"empty port": makeService("foo", func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Protocol: api.ProtocolTCP,
			}}
		}),
		"missing targetPort": makeService("foo", func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Port:     6502,
				Protocol: api.ProtocolTCP,
			}}
		}),
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
	storage, registry, server := NewTestREST(t, nil, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)

	_, err := registry.Create(ctx, makeService("foo"), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
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
	if e, a := "foo", registry.UpdatedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceRegistryUpdateDryRun(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)

	obj, err := registry.Create(ctx, makeService("foo", tweakTypeExternalName), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	svc := obj.(*api.Service)

	// Test dry run update request external name to node port
	new1 := svc.DeepCopy()
	tweakTypeNodePort(new1)
	updatedSvc, created, err := storage.Update(ctx, svc.Name, rest.DefaultUpdatedObjectInfo(new1),
		rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if updatedSvc == nil {
		t.Errorf("Expected non-nil object")
	}
	if created {
		t.Errorf("expected not created")
	}
	if storage.serviceNodePorts.Has(int(svc.Spec.Ports[0].NodePort)) {
		t.Errorf("unexpected side effect: NodePort allocated")
	}
	if e, a := "", registry.UpdatedID; e != a {
		t.Errorf("Expected %q, but got %q", e, a)
	}

	// Test dry run update request external name to cluster ip
	new2 := svc.DeepCopy()
	tweakTypeClusterIP(new2)
	_, _, err = storage.Update(ctx, svc.Name, rest.DefaultUpdatedObjectInfo(new2),
		rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if storage.serviceIPAllocatorsByFamily[storage.defaultServiceIPFamily].Has(net.ParseIP(svc.Spec.ClusterIP)) {
		t.Errorf("unexpected side effect: ip allocated")
	}

	// Test dry run update request remove node port
	obj, err = storage.Create(ctx, makeService("foo2", tweakTypeNodePort), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	svc = obj.(*api.Service)
	if !storage.serviceIPAllocatorsByFamily[storage.defaultServiceIPFamily].Has(net.ParseIP(svc.Spec.ClusterIP)) {
		t.Errorf("expected IP to be allocated")
	}
	if !storage.serviceNodePorts.Has(int(svc.Spec.Ports[0].NodePort)) {
		t.Errorf("expected NodePort to be allocated")
	}

	new3 := svc.DeepCopy()
	tweakTypeExternalName(new3)
	_, _, err = storage.Update(ctx, svc.Name, rest.DefaultUpdatedObjectInfo(new3),
		rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if !storage.serviceNodePorts.Has(int(svc.Spec.Ports[0].NodePort)) {
		t.Errorf("unexpected side effect: NodePort unallocated")
	}

	// Test dry run update request remove cluster ip
	obj, err = storage.Create(ctx, makeService("foo3"), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("expected no error: %v", err)
	}
	svc = obj.(*api.Service)

	new4 := svc.DeepCopy()
	tweakTypeExternalName(new4)
	_, _, err = storage.Update(ctx, svc.Name, rest.DefaultUpdatedObjectInfo(new4),
		rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("expected no error: %v", err)
	}
	if !storage.serviceIPAllocatorsByFamily[storage.defaultServiceIPFamily].Has(net.ParseIP(svc.Spec.ClusterIP)) {
		t.Errorf("unexpected side effect: ip unallocated")
	}
}

func TestServiceStorageValidatesUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)
	_, err := registry.Create(ctx, makeService("foo"), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	failureCases := map[string]*api.Service{
		"empty ID": makeService(""),
		"invalid selector": makeService("", func(svc *api.Service) {
			svc.Spec.Selector = map[string]string{"ThisSelectorFailsValidation": "ok"}
		}),
	}
	for _, failureCase := range failureCases {
		c, created, err := storage.Update(ctx, failureCase.Name, rest.DefaultUpdatedObjectInfo(failureCase), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
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
	storage, registry, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)
	svc := makeService("foo", tweakTypeLoadBalancer)
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

func TestAllocateLoadBalancerNodePorts(t *testing.T) {
	setAlloc := func(val bool) serviceTweak {
		return func(s *api.Service) {
			s.Spec.AllocateLoadBalancerNodePorts = utilpointer.BoolPtr(val)
		}
	}

	testcases := []struct {
		name                 string
		svc                  *api.Service
		expectNodePorts      bool
		allocateNodePortGate bool
		expectError          bool
	}{{
		name:                 "allocate false, gate on",
		svc:                  makeService("alloc-false", tweakTypeLoadBalancer, setAlloc(false)),
		expectNodePorts:      false,
		allocateNodePortGate: true,
	}, {
		name:                 "allocate true, gate on",
		svc:                  makeService("alloc-true", tweakTypeLoadBalancer, setAlloc(true)),
		expectNodePorts:      true,
		allocateNodePortGate: true,
	}, {
		name:                 "allocate nil, gate off",
		svc:                  makeService("alloc-nil", tweakTypeLoadBalancer),
		expectNodePorts:      true,
		allocateNodePortGate: false,
	}, {
		name:                 "allocate false, gate off",
		svc:                  makeService("alloc-false", tweakTypeLoadBalancer, setAlloc(false)),
		expectNodePorts:      true,
		allocateNodePortGate: false,
	}, {
		name:                 "allocate true, gate off",
		svc:                  makeService("alloc-true", tweakTypeLoadBalancer, setAlloc(true)),
		expectNodePorts:      true,
		allocateNodePortGate: false,
	}}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := genericapirequest.NewDefaultContext()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ServiceLBNodePortControl, tc.allocateNodePortGate)()

			storage, registry, server := NewTestREST(t, nil, singleStackIPv4)
			defer server.Terminate(t)

			_, err := storage.Create(ctx, tc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				if tc.expectError {
					return
				}
				t.Errorf("%s; Failed to create service: %#v", tc.name, err)
			}
			srv, err := registry.GetService(ctx, tc.svc.Name, &metav1.GetOptions{})
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

			for i := range serviceNodePorts {
				nodePort := serviceNodePorts[i]
				// Release the node port at the end of the test case.
				storage.serviceNodePorts.Release(nodePort)
			}
		})
	}
}

func TestServiceRegistryDelete(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)
	svc := makeService("foo")
	_, err := registry.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	storage.Delete(ctx, svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
	if e, a := "foo", registry.DeletedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceRegistryDeleteDryRun(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)

	// Test dry run delete request with cluster ip
	svc := makeService("foo")
	obj, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	createdSvc := obj.(*api.Service)
	if createdSvc.Spec.ClusterIP == "" {
		t.Fatalf("expected ClusterIP to be set")
	}
	if !storage.serviceIPAllocatorsByFamily[storage.defaultServiceIPFamily].Has(net.ParseIP(createdSvc.Spec.ClusterIP)) {
		t.Errorf("expected ClusterIP to be allocated")
	}
	_, _, err = storage.Delete(ctx, svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if e, a := "", registry.DeletedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
	if !storage.serviceIPAllocatorsByFamily[storage.defaultServiceIPFamily].Has(net.ParseIP(createdSvc.Spec.ClusterIP)) {
		t.Errorf("unexpected side effect: ip unallocated")
	}

	// Test dry run delete request with node port
	svc = makeService("foo2", tweakTypeNodePort)
	obj, err = storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	createdSvc = obj.(*api.Service)
	if createdSvc.Spec.Ports[0].NodePort == 0 {
		t.Fatalf("expected NodePort to be set")
	}
	if !storage.serviceNodePorts.Has(int(createdSvc.Spec.Ports[0].NodePort)) {
		t.Errorf("expected NodePort to be allocated")
	}

	isValidClusterIPFields(t, storage, svc, svc)

	_, _, err = storage.Delete(ctx, svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if e, a := "", registry.DeletedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
	if !storage.serviceNodePorts.Has(int(createdSvc.Spec.Ports[0].NodePort)) {
		t.Errorf("unexpected side effect: NodePort unallocated")
	}
}

func TestDualStackServiceRegistryDeleteDryRun(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	// dry run for non dualstack
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, true)()
	dualstack_storage, dualstack_registry, dualstack_server := NewTestREST(t, nil, []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol})
	defer dualstack_server.Terminate(t)
	// Test dry run delete request with cluster ip
	dualstack_svc := makeService("foo",
		tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
		tweakIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
		tweakClusterIPs("2000:0:0:0:0:0:0:1", "1.2.3.4"))

	_, err := dualstack_storage.Create(ctx, dualstack_svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	isValidClusterIPFields(t, dualstack_storage, dualstack_svc, dualstack_svc)
	_, _, err = dualstack_storage.Delete(ctx, dualstack_svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if e, a := "", dualstack_registry.DeletedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
	for i, family := range dualstack_svc.Spec.IPFamilies {
		if !dualstack_storage.serviceIPAllocatorsByFamily[family].Has(net.ParseIP(dualstack_svc.Spec.ClusterIPs[i])) {
			t.Errorf("unexpected side effect: ip unallocated %v", dualstack_svc.Spec.ClusterIPs[i])
		}
	}
}

func TestServiceRegistryDeleteExternal(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)
	svc := makeService("foo", tweakTypeExternalName)
	_, err := registry.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	storage.Delete(ctx, svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
	if e, a := "foo", registry.DeletedID; e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}
}

func TestServiceRegistryUpdateExternalService(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)

	// Create non-external load balancer.
	svc1 := makeService("foo")
	obj, err := storage.Create(ctx, svc1, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Modify load balancer to be external.
	svc2 := obj.(*api.Service).DeepCopy()
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
	storage, registry, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)

	// Create external load balancer.
	svc1 := makeService("foo", func(s *api.Service) {
		s.Spec.Type = api.ServiceTypeLoadBalancer
		s.Spec.Ports = []api.ServicePort{{
			Name:       "p",
			Port:       6502,
			Protocol:   api.ProtocolTCP,
			TargetPort: intstr.FromInt(6502),
		}, {
			Name:       "q",
			Port:       8086,
			Protocol:   api.ProtocolTCP,
			TargetPort: intstr.FromInt(8086),
		}}
	})
	obj, err := storage.Create(ctx, svc1, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer releaseServiceNodePorts(t, ctx, svc1.Name, storage, registry)

	// Modify ports
	svc2 := obj.(*api.Service).DeepCopy()
	svc2.Spec.Ports[1].Port = 8088
	if _, _, err := storage.Update(ctx, svc2.Name, rest.DefaultUpdatedObjectInfo(svc2), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
}

func TestServiceRegistryGet(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)
	_, err := registry.Create(ctx, makeService("foo"), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating service: %v", err)
	}
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
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo-second-ip",
					Namespace: metav1.NamespaceDefault,
				},
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "2001:db7::", TargetRef: &api.ObjectReference{Name: "foo", Namespace: metav1.NamespaceDefault}}},
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
	storage, registry, server := NewTestRESTWithPods(t, endpoints, pods, singleStackIPv4)
	defer server.Terminate(t)
	for _, name := range []string{"foo", "bad"} {
		_, err := registry.Create(ctx, makeService(name, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{
				// Service port 9393 should route to endpoint port "p", which is port 93
				{Name: "p", Port: 9393, TargetPort: intstr.FromString("p")},

				// Service port 93 should route to unnamed endpoint port, which is port 80
				// This is to test that the service port definition is used when determining resource location
				{Name: "", Port: 93, TargetPort: intstr.FromInt(80)},
			}
		}), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("error creating service: %v", err)
		}
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

	// Test a simple id (using second ip).
	location, _, err = redirector.ResourceLocation(ctx, "foo-second-ip")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if location == nil {
		t.Errorf("Unexpected nil: %v", location)
	}
	if e, a := "//[2001:db7::]:80", location.String(); e != a {
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

	// Test a name + port (using second ip).
	location, _, err = redirector.ResourceLocation(ctx, "foo-second-ip:p")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if location == nil {
		t.Errorf("Unexpected nil: %v", location)
	}
	if e, a := "//[2001:db7::]:93", location.String(); e != a {
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

	// Test a name + port number (service port 93 -> target port 80, using second ip)
	location, _, err = redirector.ResourceLocation(ctx, "foo-second-ip:93")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if location == nil {
		t.Errorf("Unexpected nil: %v", location)
	}
	if e, a := "//[2001:db7::]:80", location.String(); e != a {
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

	// Test a name + port number (service port 9393 -> target port "p" -> endpoint port 93, using second ip)
	location, _, err = redirector.ResourceLocation(ctx, "foo-second-ip:9393")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if location == nil {
		t.Errorf("Unexpected nil: %v", location)
	}
	if e, a := "//[2001:db7::]:93", location.String(); e != a {
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

	// Test a scheme + name + port (using second ip).
	location, _, err = redirector.ResourceLocation(ctx, "https:foo-second-ip:p")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if location == nil {
		t.Errorf("Unexpected nil: %v", location)
	}
	if e, a := "https://[2001:db7::]:93", location.String(); e != a {
		t.Errorf("Expected %v, but got %v", e, a)
	}

	// Test a non-existent name + port.
	_, _, err = redirector.ResourceLocation(ctx, "foo:q")
	if err == nil {
		t.Errorf("Unexpected nil error")
	}

	// Test a non-existent name + port (using second ip).
	_, _, err = redirector.ResourceLocation(ctx, "foo-second-ip:q")
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
	storage, registry, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)
	_, err := registry.Create(ctx, makeService("foo"), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, err = registry.Create(ctx, makeService("foo2"), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
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
	storage, _, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)

	svc1 := makeService("foo")
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

	svc2 := makeService("bar")
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
		if !storage.serviceIPAllocatorsByFamily[storage.defaultServiceIPFamily].(*ipallocator.Range).Has(net.ParseIP(ip)) {
			testIP = ip
			break
		}
	}

	svc3 := makeService("qux", tweakClusterIPs(testIP))
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
	storage, _, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)

	svc1 := makeService("foo")
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

	svc2 := makeService("bar", tweakClusterIPs(svc1.Spec.ClusterIP))
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
	storage, _, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)

	svc := makeService("foo")
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
		if !storage.serviceIPAllocatorsByFamily[storage.defaultServiceIPFamily].(*ipallocator.Range).Has(net.ParseIP(ip)) {
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
	storage, registry, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)

	svc := makeService("foo", tweakTypeLoadBalancer)
	ctx := genericapirequest.NewDefaultContext()
	createdSvc, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if createdSvc == nil || err != nil {
		t.Errorf("Unexpected failure creating service %v", err)
	}
	defer releaseServiceNodePorts(t, ctx, svc.Name, storage, registry)

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

func TestUpdateServiceWithConflictingNamespace(t *testing.T) {
	storage, _, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)
	service := makeService("test", func(s *api.Service) { s.Namespace = "not-default" })

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
	storage, registry, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)
	svc := makeService("external-lb-esipp", tweakTypeLoadBalancer, func(s *api.Service) {
		s.Spec.ExternalTrafficPolicy = api.ServiceExternalTrafficPolicyTypeLocal
	})
	obj, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if obj == nil || err != nil {
		t.Errorf("Unexpected failure creating service %v", err)
	}
	defer releaseServiceNodePorts(t, ctx, svc.Name, storage, registry)

	createdSvc := obj.(*api.Service)
	if !service.NeedsHealthCheck(createdSvc) {
		t.Errorf("Expecting health check needed, returned health check not needed instead")
	}
	port := createdSvc.Spec.HealthCheckNodePort
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
	ctx := genericapirequest.NewDefaultContext()
	storage, registry, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)
	svc := makeService("external-lb-esipp", tweakTypeLoadBalancer, func(s *api.Service) {
		// hard-code NodePort to make sure it doesn't conflict with the healthport.
		// TODO: remove this once http://issue.k8s.io/93922 fixes auto-allocation conflicting with user-specified health check ports
		s.Spec.Ports[0].NodePort = 30500
		s.Spec.ExternalTrafficPolicy = api.ServiceExternalTrafficPolicyTypeLocal
		s.Spec.HealthCheckNodePort = 30501
	})
	obj, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if obj == nil || err != nil {
		t.Fatalf("Unexpected failure creating service :%v", err)
	}
	defer releaseServiceNodePorts(t, ctx, svc.Name, storage, registry)

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
	if port != 0 {
		// Release the health check node port at the end of the test case.
		storage.serviceNodePorts.Release(int(port))
	}
}

// Validate that the service creation fails when the requested port number is -1.
func TestServiceRegistryExternalTrafficHealthCheckNodePortNegative(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	storage, _, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)
	svc := makeService("external-lb-esipp", tweakTypeLoadBalancer, func(s *api.Service) {
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
	storage, registry, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)
	svc := makeService("external-lb-esipp", tweakTypeLoadBalancer, func(s *api.Service) {
		s.Spec.ExternalTrafficPolicy = api.ServiceExternalTrafficPolicyTypeCluster
	})
	obj, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if obj == nil || err != nil {
		t.Errorf("Unexpected failure creating service %v", err)
	}
	defer releaseServiceNodePorts(t, ctx, svc.Name, storage, registry)

	createdSvc := obj.(*api.Service)
	if service.NeedsHealthCheck(createdSvc) {
		t.Errorf("Expecting health check not needed, returned health check needed instead")
	}
	// Make sure the service does not have the health check node port allocated
	port := createdSvc.Spec.HealthCheckNodePort
	if port != 0 {
		// Release the health check node port at the end of the test case.
		storage.serviceNodePorts.Release(int(port))
		t.Errorf("Unexpected allocation of health check node port: %v", port)
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
		svc:                      makeService("foo"),
		enableDualStackAllocator: false,
		expectError:              false,
		preAllocateClusterIPs:    nil,
		expectedCountIPs:         1,
	}, {
		name: "Allocate single ClusterIP (v6)",
		svc: makeService("foo",
			tweakIPFamilies(api.IPv6Protocol)),
		expectError:              false,
		enableDualStackAllocator: true,
		preAllocateClusterIPs:    nil,
		expectedCountIPs:         1,
	}, {
		name: "Allocate specified ClusterIP (v4)",
		svc: makeService("foo",
			tweakIPFamilies(api.IPv4Protocol),
			tweakClusterIPs("1.2.3.4")),
		expectError:              false,
		enableDualStackAllocator: true,
		preAllocateClusterIPs:    nil,
		expectedCountIPs:         1,
		expectedClusterIPs:       []string{"1.2.3.4"},
	}, {
		name: "Allocate specified ClusterIP-v6",
		svc: makeService("foo",
			tweakIPFamilies(api.IPv6Protocol),
			tweakClusterIPs("2000:0:0:0:0:0:0:1")),
		expectError:              false,
		enableDualStackAllocator: true,
		expectedCountIPs:         1,
		expectedClusterIPs:       []string{"2000:0:0:0:0:0:0:1"},
	}, {
		name: "Allocate dual stack - on a non dual stack ",
		svc: makeService("foo",
			tweakIPFamilies(api.IPv4Protocol)),
		expectError:              false,
		enableDualStackAllocator: false,
		expectedCountIPs:         1,
	}, {
		name: "Allocate dual stack - upgrade - v4, v6",
		svc: makeService("foo",
			tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
			tweakIPFamilies(api.IPv4Protocol)),
		expectError:              false,
		enableDualStackAllocator: true,
		expectedCountIPs:         2,
	}, {
		name: "Allocate dual stack - upgrade - v4, v6 - specific first IP",
		svc: makeService("foo",
			tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
			tweakIPFamilies(api.IPv4Protocol),
			tweakClusterIPs("1.2.3.4")),
		expectError:              false,
		enableDualStackAllocator: true,
		expectedCountIPs:         2,
		expectedClusterIPs:       []string{"1.2.3.4"},
	}, {
		name: "Allocate dual stack - upgrade - v6, v4",
		svc: makeService("foo",
			tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
			tweakIPFamilies(api.IPv6Protocol)),
		expectError:              false,
		enableDualStackAllocator: true,
		expectedCountIPs:         2,
	}, {
		name: "Allocate dual stack - v4, v6 - specific ips",
		svc: makeService("foo",
			tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
			tweakIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
			tweakClusterIPs("1.2.3.4", "2000:0:0:0:0:0:0:1")),
		expectError:              false,
		enableDualStackAllocator: true,
		expectedCountIPs:         2,
		expectedClusterIPs:       []string{"1.2.3.4", "2000:0:0:0:0:0:0:1"},
	}, {
		name: "Allocate dual stack - upgrade - v6, v4 - specific ips",
		svc: makeService("foo",
			tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
			tweakIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
			tweakClusterIPs("2000:0:0:0:0:0:0:1", "1.2.3.4")),
		expectError:              false,
		enableDualStackAllocator: true,
		expectedCountIPs:         2,
		expectedClusterIPs:       []string{"2000:0:0:0:0:0:0:1", "1.2.3.4"},
	}, {
		name: "Shouldn't allocate ClusterIP",
		svc: makeService("foo",
			tweakClusterIPs("None")),
		expectError:              false,
		enableDualStackAllocator: false,
		expectedCountIPs:         0,
	}, {
		name: "single stack, ip is pre allocated (ipv4)",
		svc: makeService("foo",
			tweakIPFamilyPolicy(api.IPFamilyPolicySingleStack),
			tweakIPFamilies(api.IPv4Protocol),
			tweakClusterIPs("1.2.3.4")),
		expectError:              true,
		enableDualStackAllocator: false,
		expectedCountIPs:         0,
		preAllocateClusterIPs:    map[api.IPFamily]string{api.IPv4Protocol: "1.2.3.4"},
	}, {
		name: "single stack, ip is pre allocated (ipv6)",
		svc: makeService("foo",
			tweakIPFamilyPolicy(api.IPFamilyPolicySingleStack),
			tweakIPFamilies(api.IPv6Protocol),
			tweakClusterIPs("2000:0:0:0:0:0:0:1")),
		expectError:              true,
		enableDualStackAllocator: true, // ipv6 allocator is always the second one during test
		expectedCountIPs:         0,
		preAllocateClusterIPs:    map[api.IPFamily]string{api.IPv6Protocol: "2000:0:0:0:0:0:0:1"},
	}, {
		name: "Allocate dual stack - upgrade - v6, v4 - specific ips (first ip can't be allocated)",
		svc: makeService("foo",
			tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
			tweakIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
			tweakClusterIPs("2000:0:0:0:0:0:0:1", "1.2.3.4")),
		expectError:              true,
		enableDualStackAllocator: true,
		expectedCountIPs:         0,
		preAllocateClusterIPs:    map[api.IPFamily]string{api.IPv6Protocol: "2000:0:0:0:0:0:0:1"},
	}, {
		name: "Allocate dual stack - upgrade - v6, v4 - specific ips (second ip can't be allocated)",
		svc: makeService("foo",
			tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
			tweakIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
			tweakClusterIPs("2000:0:0:0:0:0:0:1", "1.2.3.4")),
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
			storage, _, server := NewTestREST(t, nil, families)
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
					// has retruns true if it was allocated *sigh*..
					if !allocator.Has(net.ParseIP(ip)) {
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
	storage, _, server := NewTestREST(t, nil, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	nodePortOp := portallocator.StartOperation(storage.serviceNodePorts, false)
	defer nodePortOp.Finish()

	testCases := []struct {
		name                     string
		service                  *api.Service
		expectSpecifiedNodePorts []int
	}{{
		name:                     "Service doesn't have specified NodePort",
		service:                  makeService("foo", tweakTypeNodePort),
		expectSpecifiedNodePorts: []int{},
	}, {
		name: "Service has one specified NodePort",
		service: makeService("foo", tweakTypeNodePort, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Name:       "port-tcp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolTCP,
				NodePort:   30053,
			}}
		}),
		expectSpecifiedNodePorts: []int{30053},
	}, {
		name: "Service has two same ports with different protocols and specifies same NodePorts",
		service: makeService("foo", tweakTypeNodePort, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Name:       "port-tcp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolTCP,
				NodePort:   30054,
			}, {
				Name:       "port-udp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolUDP,
				NodePort:   30054,
			}}
		}),
		expectSpecifiedNodePorts: []int{30054, 30054},
	}, {
		name: "Service has two same ports with different protocols and specifies different NodePorts",
		service: makeService("foo", tweakTypeNodePort, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Name:       "port-tcp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolTCP,
				NodePort:   30055,
			}, {
				Name:       "port-udp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolUDP,
				NodePort:   30056,
			}}
		}),
		expectSpecifiedNodePorts: []int{30055, 30056},
	}, {
		name: "Service has two different ports with different protocols and specifies different NodePorts",
		service: makeService("foo", tweakTypeNodePort, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Name:       "port-tcp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolTCP,
				NodePort:   30057,
			}, {
				Name:       "port-udp",
				Port:       54,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolUDP,
				NodePort:   30058,
			}}
		}),
		expectSpecifiedNodePorts: []int{30057, 30058},
	}, {
		name: "Service has two same ports with different protocols but only specifies one NodePort",
		service: makeService("foo", tweakTypeNodePort, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Name:       "port-tcp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolTCP,
				NodePort:   30059,
			}, {
				Name:       "port-udp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolUDP,
			}}
		}),
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
	storage, _, server := NewTestREST(t, nil, singleStackIPv4)
	defer server.Terminate(t)
	nodePortOp := portallocator.StartOperation(storage.serviceNodePorts, false)
	defer nodePortOp.Finish()

	testCases := []struct {
		name                     string
		oldService               *api.Service
		newService               *api.Service
		expectSpecifiedNodePorts []int
	}{{
		name: "Old service and new service have the same NodePort",
		oldService: makeService("foo", tweakTypeNodePort, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
				NodePort:   30053,
			}}
		}),
		newService: makeService("foo", tweakTypeNodePort, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
				NodePort:   30053,
			}}
		}),
		expectSpecifiedNodePorts: []int{30053},
	}, {
		name: "Old service has more NodePorts than new service has",
		oldService: makeService("foo", tweakTypeNodePort, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Name:       "port-tcp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolTCP,
				NodePort:   30053,
			}, {
				Name:       "port-udp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolUDP,
				NodePort:   30053,
			}}
		}),
		newService: makeService("foo", tweakTypeNodePort, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Name:       "port-tcp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolTCP,
				NodePort:   30053,
			}}
		}),
		expectSpecifiedNodePorts: []int{30053},
	}, {
		name: "Change protocol of ServicePort without changing NodePort",
		oldService: makeService("foo", tweakTypeNodePort, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Name:       "port-tcp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolTCP,
				NodePort:   30053,
			}}
		}),
		newService: makeService("foo", tweakTypeNodePort, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Name:       "port-udp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolUDP,
				NodePort:   30053,
			}}
		}),
		expectSpecifiedNodePorts: []int{30053},
	}, {
		name: "Should allocate NodePort when changing service type to NodePort",
		oldService: makeService("foo", tweakTypeNodePort, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}}
		}),
		newService: makeService("foo", tweakTypeNodePort, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}}
		}),
		expectSpecifiedNodePorts: []int{},
	}, {
		name: "Add new ServicePort with a different protocol without changing port numbers",
		oldService: makeService("foo", tweakTypeNodePort, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Name:       "port-tcp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolTCP,
				NodePort:   30053,
			}}
		}),
		newService: makeService("foo", tweakTypeNodePort, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Name:       "port-tcp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolTCP,
				NodePort:   30053,
			}, {
				Name:       "port-udp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolUDP,
				NodePort:   30053,
			}}
		}),
		expectSpecifiedNodePorts: []int{30053, 30053},
	}, {
		name: "Change service type from ClusterIP to NodePort with same NodePort number but different protocols",
		oldService: makeService("foo", tweakTypeNodePort, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Port:       53,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}}
		}),
		newService: makeService("foo", tweakTypeNodePort, func(s *api.Service) {
			s.Spec.Ports = []api.ServicePort{{
				Name:       "port-tcp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolTCP,
				NodePort:   30053,
			}, {
				Name:       "port-udp",
				Port:       53,
				TargetPort: intstr.FromInt(6502),
				Protocol:   api.ProtocolUDP,
				NodePort:   30053,
			}}
		}),
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

		svc: makeService("foo"),
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

		svc: makeService("foo", func(s *api.Service) {
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

		svc: makeService("foo", func(s *api.Service) {
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

		svc: makeService("foo", func(s *api.Service) {
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

		svc: makeService("foo", func(s *api.Service) {
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

		svc: makeService("foo", func(s *api.Service) {
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

		svc: makeService("foo", func(s *api.Service) {
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

		svc: makeService("foo", func(s *api.Service) {
			s.Spec.IPFamilies = []api.IPFamily{api.IPv6Protocol}
		}),
	}}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			families := []api.IPFamily{api.IPv4Protocol}
			if testCase.enableDualStackAllocator {
				families = append(families, api.IPv6Protocol)
			}
			storage, _, server := NewTestREST(t, nil, families)
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
				// has retruns true if it was allocated *sigh*..
				if !allocator.Has(net.ParseIP(ip)) {
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

		svc: makeService("foo", func(s *api.Service) {
			s.Spec.IPFamilyPolicy = &requiredDualStack
			s.Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol}
		}),
	}, {
		name:                     "normal, no downgrade needed. dual stack => dual stack",
		enableDualStackAllocator: true,
		enableDualStackGate:      true,
		expectDowngradeError:     false,

		updateFunc: func(s *api.Service) { s.Spec.Selector = map[string]string{"bar": "baz2"} },

		svc: makeService("foo", func(s *api.Service) {
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

		svc: makeService("foo", func(s *api.Service) {
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

		svc: makeService("foo", func(s *api.Service) {
			s.Spec.IPFamilyPolicy = &requiredDualStack
			s.Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol}
		}),
	}}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			storage, _, server := NewTestREST(t, nil, []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol})
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

				if allocator.Has(net.ParseIP(releasedIP)) {
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
			svc:                    makeService("foo", tweakTypeExternalName),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            false,
		},
		{
			name:                   "[singlestack:v4] set: nothing",
			modifyRest:             fnMakeSingleStackIPv4Allocator,
			svc:                    makeService("foo"),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},

		{
			name:       "[singlestack:v4] set: v4Cluster IPSet",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: v4IPFamilySet",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: v4IPFamilySet",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("10.0.0.4"),
				tweakIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: PreferDualStack",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: PreferDualStack + v4ClusterIPSet",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				tweakClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: PreferDualStack + v4ClusterIPSet + v4FamilySet",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				tweakIPFamilies(api.IPv4Protocol),
				tweakClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: v6IPSet",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("2000::1")),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v4] set: v6IPFamily",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v4] set: RequireDualStack",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v4] set: RequireDualStack + family",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		// selector less
		{
			name:       "[singlestack:v4] set: selectorless, families are ignored",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("None"),
				tweakIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: selectorless, no families",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("None"),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: selectorless, user selected",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("None"),
				tweakIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v4] set: selectorless, user set to preferDualStack",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("None"),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		// tests incorrect setting for IPFamilyPolicy
		{
			name:       "[singlestack:v4] set: multifamily set to preferDualStack",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v4] set: multifamily set to singleStack",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				tweakIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v4] set: mult clusterips set to preferDualStack",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("1.1.1.1", "2001::1"),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v4] set: multi clusterips set to singleStack",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("1.1.1.1", "2001::1"),
				tweakIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
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
			svc:                    makeService("foo", tweakTypeExternalName),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            false,
		},
		{
			name:                   "[singlestack:v6] set: nothing",
			modifyRest:             fnMakeSingleStackIPv6Allocator,
			svc:                    makeService("foo"),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: v6Cluster IPSet",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("2000::1")),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: v4IPFamilySet",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: v6IPFamilySet",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("2000::1"),
				tweakIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: PreferDualStack",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: PreferDualStack + v6ClusterIPSet",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				tweakClusterIPs("2000::1")),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: PreferDualStack + v6ClusterIPSet + v6FamilySet",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				tweakIPFamilies(api.IPv6Protocol),
				tweakClusterIPs("2000::1")),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: v4IPSet",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("10.0.0.10")),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v6] set: v4IPFamily",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v6] set: RequireDualStack (on single stack ipv6 cluster)",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v6] set: RequireDualStack + family (on single stack ipv6 cluster)",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},

		// selector less
		{
			name:       "[singlestack:v6] set: selectorless, families are ignored",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("None"),
				tweakIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: selectorless, no families",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("None"),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: selectorless, user selected",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("None"),
				tweakIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[singlestack:v6] set: selectorless, user set to preferDualStack",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("None"),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		// tests incorrect setting for IPFamilyPolicy
		{
			name:       "[singlestack:v6] set: multifamily set to preferDualStack",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v6] set: multifamily set to singleStack",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				tweakIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v6] set: mult clusterips set to preferDualStack",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("1.1.1.1", "2001::1"),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[singlestack:v6] set: multi clusterips set to singleStack",
			modifyRest: fnMakeSingleStackIPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("1.1.1.1", "2001::1"),
				tweakIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
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
			svc:                    makeService("foo", tweakTypeExternalName),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            false,
		},
		{
			name:                   "[dualstack:v4,v6] set: nothing",
			modifyRest:             fnMakeDualStackStackIPv4IPv6Allocator,
			svc:                    makeService("foo"),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},

		{
			name:       "[dualstack:v4,v6] set: v4ClusterIPSet",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: v4IPFamilySet",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: v4IPFamilySet",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("10.0.0.4"),
				tweakIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: v6ClusterIPSet",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("2000::1")),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: v6IPFamilySet",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: v6IPFamilySet",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("2000::1"),
				tweakIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		// prefer dual stack
		{
			name:       "[dualstack:v4,v6] set: PreferDualStack.",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: PreferDualStack + v4ClusterIPSet",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				tweakClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: PreferDualStack + v4ClusterIPSet + v4FamilySet",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				tweakIPFamilies(api.IPv4Protocol),
				tweakClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		// require dual stack
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + family v4",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + family v6",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},

		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + family +ip v4",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakClusterIPs("10.0.0.10"),
				tweakIPFamilies(api.IPv4Protocol)),
			//
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + family +ip v6",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakClusterIPs("2000::1"),
				tweakIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + ip v6",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakClusterIPs("2000::1")),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + ip v4",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakClusterIPs("10.0.0.10")),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + ips",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakClusterIPs("10.0.0.10", "2000::1")),
			//
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + ips",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakClusterIPs("2000::1", "10.0.0.10")),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: RequireDualStack + ips + families v6,v4",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakClusterIPs("2000::1", "10.0.0.10"),
				tweakIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + ips + families v4,v6",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakClusterIPs("10.0.0.10", "2000::1"),
				tweakIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,v6] set: selectorless, no families",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("None"),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,6] set: selectorless, user selected",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("None"),
				tweakIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,6] set: selectorless, user set to prefer",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("None"),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		// tests incorrect setting for IPFamilyPolicy
		{
			name:       "[duakstack:v4,6] set: multifamily set to preferDualStack",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,6] set: multifamily set to singleStack",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				tweakIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[dualstack:v4,6] set: mult clusterips set to preferDualStack",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("1.1.1.1", "2001::1"),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,6] set: multi clusterips set to singleStack",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			svc: makeService("foo",
				tweakClusterIPs("1.1.1.1", "2001::1"),
				tweakIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
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
			svc:                    makeService("foo", tweakTypeExternalName),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            false,
		},
		{
			name:                   "[dualstack:v6,v4] set: nothing",
			modifyRest:             fnMakeDualStackStackIPv6IPv4Allocator,
			svc:                    makeService("foo"),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: v4ClusterIPSet",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: v4IPFamilySet",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilies(api.IPv4Protocol)),
			//
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: v4IPFamilySet",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("10.0.0.4"),
				tweakIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: v6ClusterIPSet",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("2000::1")),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: v6IPFamilySet",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: v6IPFamilySet",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("2000::1"),
				tweakIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		// prefer dual stack
		{
			name:       "[dualstack:v6,v4] set: PreferDualStack.",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: PreferDualStack + v4ClusterIPSet",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				tweakClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: PreferDualStack + v4ClusterIPSet + v4FamilySet",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				tweakIPFamilies(api.IPv4Protocol),
				tweakClusterIPs("10.0.0.4")),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		// require dual stack
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + family v4",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + family v6",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + family +ip v4",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakClusterIPs("10.0.0.10"),
				tweakIPFamilies(api.IPv4Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + family +ip v6",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakClusterIPs("2000::1"),
				tweakIPFamilies(api.IPv6Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + ip v6",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakClusterIPs("2000::1")),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + ip v4",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakClusterIPs("10.0.0.10")),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + ip v4",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakClusterIPs("10.0.0.10")),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + ips",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakClusterIPs("10.0.0.10", "2000::1")),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + ips",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakClusterIPs("2000::1", "10.0.0.10")),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + ips + families v6,v4",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakClusterIPs("2000::1", "10.0.0.10"),
				tweakIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: RequireDualStack + ips + families v4,v6",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				tweakClusterIPs("10.0.0.10", "2000::1"),
				tweakIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: selectorless, no families",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("None"),
				func(s *api.Service) { s.Spec.Selector = nil }),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: selectorless, user selected",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("None"),
				tweakIPFamilyPolicy(api.IPFamilyPolicySingleStack)),

			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,v4] set: selectorless, user set to prefer",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("None"),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),

			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			expectError:            false,
		},
		// tests incorrect setting for IPFamilyPolicy
		{
			name:       "[duakstack:v6,5] set: multifamily set to preferDualStack",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v4,6] set: multifamily set to singleStack",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				tweakIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectedIPFamilyPolicy: nil,
			expectedIPFamilies:     nil,
			expectError:            true,
		},
		{
			name:       "[dualstack:v6,4] set: mult clusterips set to preferDualStack",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("1.1.1.1", "2001::1"),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},
		{
			name:       "[dualstack:v6,4] set: multi clusterips set to singleStack",
			modifyRest: fnMakeDualStackStackIPv6IPv4Allocator,
			svc: makeService("foo",
				tweakClusterIPs("1.1.1.1", "2001::1"),
				tweakIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
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
			oldSvc: makeService("foo",
				tweakClusterIPs("1.1.1.1"),
				tweakIPFamilies(api.IPv4Protocol),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),

			svc: makeService("foo",
				tweakClusterIPs("1.1.1.1"),
				tweakIPFamilies(api.IPv4Protocol),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},

		{
			name:       "unchanged preferDualStack-2-ClusterDowngraded",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			oldSvc: makeService("foo",
				tweakClusterIPs("1.1.1.1", "2001::1"),
				tweakIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),

			svc: makeService("foo",
				tweakClusterIPs("1.1.1.1", "2001::1"),
				tweakIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectedIPFamilyPolicy: &preferDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},

		{
			name:       "changed preferDualStack-1 (cluster upgraded)",
			modifyRest: fnMakeDualStackStackIPv4IPv6Allocator,
			oldSvc: makeService("foo",
				tweakClusterIPs("1.1.1.1"),
				tweakIPFamilies(api.IPv4Protocol),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),

			svc: makeService("foo",
				tweakIPFamilies(api.IPv4Protocol),
				tweakIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
			expectedIPFamilyPolicy: &requireDualStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			expectError:            false,
		},

		{
			name:       "changed preferDualStack-2-ClusterDowngraded",
			modifyRest: fnMakeSingleStackIPv4Allocator,
			oldSvc: makeService("foo",
				tweakClusterIPs("1.1.1.1", "2001::1"),
				tweakIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				tweakIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),

			svc: makeService("foo",
				tweakClusterIPs("1.1.1.1"),
				tweakIPFamilies(api.IPv4Protocol),
				tweakIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectedIPFamilyPolicy: &singleStack,
			expectedIPFamilies:     []api.IPFamily{api.IPv4Protocol},
			expectError:            false,
		},
	}

	// This func only runs when feature gate is on
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, true)()

	storage, _, server := NewTestREST(t, nil, []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol})
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
