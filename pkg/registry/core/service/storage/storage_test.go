/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"net"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	machineryutilnet "k8s.io/apimachinery/pkg/util/net"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	svctest "k8s.io/kubernetes/pkg/api/service/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	utilnet "k8s.io/utils/net"
)

func makeIPAllocator(cidr *net.IPNet) ipallocator.Interface {
	al, err := ipallocator.NewInMemory(cidr)
	if err != nil {
		panic(fmt.Sprintf("error creating IP allocator: %v", err))
	}
	return al
}

func makePortAllocator(ports machineryutilnet.PortRange) portallocator.Interface {
	al, err := portallocator.NewInMemory(ports)
	if err != nil {
		panic(fmt.Sprintf("error creating port allocator: %v", err))
	}
	return al
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

func newStorage(t *testing.T, ipFamilies []api.IPFamily) (*GenericREST, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "services",
	}

	ipAllocs := map[api.IPFamily]ipallocator.Interface{}
	for _, fam := range ipFamilies {
		switch fam {
		case api.IPv4Protocol:
			_, cidr, _ := net.ParseCIDR("10.0.0.0/16")
			ipAllocs[fam] = makeIPAllocator(cidr)
		case api.IPv6Protocol:
			_, cidr, _ := net.ParseCIDR("2000::/108")
			ipAllocs[fam] = makeIPAllocator(cidr)
		default:
			t.Fatalf("Unknown IPFamily: %v", fam)
		}
	}

	portAlloc := makePortAllocator(*(machineryutilnet.ParsePortRangeOrDie("30000-32767")))

	serviceStorage, statusStorage, err := NewGenericREST(restOptions, ipFamilies[0], ipAllocs, portAlloc)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return serviceStorage, statusStorage, server
}

// This is used in generic registry tests.
func validService() *api.Service {
	return svctest.MakeService("foo",
		svctest.SetClusterIPs(api.ClusterIPNone),
		svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
		svctest.SetIPFamilies(api.IPv4Protocol))
}

func TestGenericCreate(t *testing.T) {
	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	svc := validService()
	svc.ObjectMeta = metav1.ObjectMeta{} // because genericregistrytest
	test.TestCreate(
		// valid
		svc,
		// invalid
		&api.Service{
			Spec: api.ServiceSpec{},
		},
	)
}

func TestGenericUpdate(t *testing.T) {
	clusterInternalTrafficPolicy := api.ServiceInternalTrafficPolicyCluster

	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).AllowCreateOnUpdate()
	test.TestUpdate(
		// valid
		validService(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*api.Service)
			object.Spec = api.ServiceSpec{
				Selector:        map[string]string{"bar": "baz2"},
				ClusterIP:       api.ClusterIPNone,
				ClusterIPs:      []string{api.ClusterIPNone},
				SessionAffinity: api.ServiceAffinityNone,
				Type:            api.ServiceTypeClusterIP,
				Ports: []api.ServicePort{{
					Port:       6502,
					Protocol:   api.ProtocolTCP,
					TargetPort: intstr.FromInt(6502),
				}},
				InternalTrafficPolicy: &clusterInternalTrafficPolicy,
			}
			return object
		},
	)
}

func TestGenericDelete(t *testing.T) {
	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).AllowCreateOnUpdate().ReturnDeletedObject()
	test.TestDelete(validService())
}

func TestGenericGet(t *testing.T) {
	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).AllowCreateOnUpdate()
	test.TestGet(validService())
}

func TestGenericList(t *testing.T) {
	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).AllowCreateOnUpdate()
	test.TestList(validService())
}

func TestGenericWatch(t *testing.T) {
	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestWatch(
		validService(),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": "foo"},
		},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "bar"},
		},
	)
}

func TestGenericShortNames(t *testing.T) {
	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"svc"}
	registrytest.AssertShortNames(t, storage, expected)
}

func TestGenericCategories(t *testing.T) {
	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"all"}
	registrytest.AssertCategories(t, storage, expected)
}

func makeServiceList() (undefaulted, defaulted *api.ServiceList) {
	undefaulted = &api.ServiceList{Items: []api.Service{}}
	defaulted = &api.ServiceList{Items: []api.Service{}}

	singleStack := api.IPFamilyPolicySingleStack
	requireDualStack := api.IPFamilyPolicyRequireDualStack

	var undefaultedSvc *api.Service
	var defaultedSvc *api.Service

	// (for headless) tests must set fields  manually according to how the cluster configured
	// headless w selector (subject to how the cluster is configured)
	undefaultedSvc = &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "headless_with_selector", ResourceVersion: "1", Namespace: metav1.NamespaceDefault},
		Spec: api.ServiceSpec{
			Type:       api.ServiceTypeClusterIP,
			ClusterIPs: []string{api.ClusterIPNone},
			Selector:   map[string]string{"foo": "bar"},
		},
	}
	defaultedSvc = undefaultedSvc.DeepCopy()
	defaultedSvc.Spec.IPFamilyPolicy = nil // forcing tests to set them
	defaultedSvc.Spec.IPFamilies = nil     // forcing tests to them

	undefaulted.Items = append(undefaulted.Items, *(undefaultedSvc))
	defaulted.Items = append(defaulted.Items, *(defaultedSvc))

	// headless w/o selector (always set to require and families according to cluster)
	undefaultedSvc = &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "headless_no_selector", ResourceVersion: "1", Namespace: metav1.NamespaceDefault},
		Spec: api.ServiceSpec{
			Type:       api.ServiceTypeClusterIP,
			ClusterIPs: []string{api.ClusterIPNone},
			Selector:   nil,
		},
	}
	defaultedSvc = undefaultedSvc.DeepCopy()
	defaultedSvc.Spec.IPFamilyPolicy = nil // forcing tests to set them
	defaultedSvc.Spec.IPFamilies = nil     // forcing tests to them

	undefaulted.Items = append(undefaulted.Items, *(undefaultedSvc))
	defaulted.Items = append(defaulted.Items, *(defaultedSvc))

	// single stack IPv4
	undefaultedSvc = &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "ipv4", ResourceVersion: "1", Namespace: metav1.NamespaceDefault},
		Spec: api.ServiceSpec{
			Type:      api.ServiceTypeClusterIP,
			ClusterIP: "10.0.0.4",
		},
	}
	defaultedSvc = undefaultedSvc.DeepCopy()
	defaultedSvc.Spec.IPFamilyPolicy = &singleStack
	defaultedSvc.Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol}

	undefaulted.Items = append(undefaulted.Items, *(undefaultedSvc))
	defaulted.Items = append(defaulted.Items, *(defaultedSvc))

	// single stack IPv6
	undefaultedSvc = &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "ipv6", ResourceVersion: "1", Namespace: metav1.NamespaceDefault},
		Spec: api.ServiceSpec{
			Type:      api.ServiceTypeClusterIP,
			ClusterIP: "2000::1",
		},
	}
	defaultedSvc = undefaultedSvc.DeepCopy()
	defaultedSvc.Spec.IPFamilyPolicy = &singleStack
	defaultedSvc.Spec.IPFamilies = []api.IPFamily{api.IPv6Protocol}

	undefaulted.Items = append(undefaulted.Items, *(undefaultedSvc))
	defaulted.Items = append(defaulted.Items, *(defaultedSvc))

	// dualstack IPv4 IPv6
	undefaultedSvc = &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "ipv4_ipv6", ResourceVersion: "1", Namespace: metav1.NamespaceDefault},
		Spec: api.ServiceSpec{
			Type:       api.ServiceTypeClusterIP,
			ClusterIP:  "10.0.0.4",
			ClusterIPs: []string{"10.0.0.4", "2000::1"},
		},
	}
	defaultedSvc = undefaultedSvc.DeepCopy()
	defaultedSvc.Spec.IPFamilyPolicy = &requireDualStack
	defaultedSvc.Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol}

	undefaulted.Items = append(undefaulted.Items, *(undefaultedSvc))
	defaulted.Items = append(defaulted.Items, *(defaultedSvc))

	// dualstack IPv6 IPv4
	undefaultedSvc = &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "ipv6_ipv4", ResourceVersion: "1", Namespace: metav1.NamespaceDefault},
		Spec: api.ServiceSpec{
			Type:       api.ServiceTypeClusterIP,
			ClusterIP:  "2000::1",
			ClusterIPs: []string{"2000::1", "10.0.0.4"},
		},
	}
	defaultedSvc = undefaultedSvc.DeepCopy()
	defaultedSvc.Spec.IPFamilyPolicy = &requireDualStack
	defaultedSvc.Spec.IPFamilies = []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol}

	undefaulted.Items = append(undefaulted.Items, *(undefaultedSvc))
	defaulted.Items = append(defaulted.Items, *(defaultedSvc))

	// external name
	undefaultedSvc = &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "external_name", ResourceVersion: "1", Namespace: metav1.NamespaceDefault},
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeExternalName,
		},
	}

	defaultedSvc = undefaultedSvc.DeepCopy()
	defaultedSvc.Spec.IPFamilyPolicy = nil
	defaultedSvc.Spec.IPFamilies = nil

	undefaulted.Items = append(undefaulted.Items, *(undefaultedSvc))
	defaulted.Items = append(defaulted.Items, *(defaultedSvc))

	return undefaulted, defaulted
}

func TestServiceDefaultOnRead(t *testing.T) {
	// Helper makes a mostly-valid Service.  Test-cases can tweak it as needed.
	makeService := func(tweak func(*api.Service)) *api.Service {
		svc := &api.Service{
			ObjectMeta: metav1.ObjectMeta{Name: "svc", Namespace: "ns"},
			Spec: api.ServiceSpec{
				Type:       api.ServiceTypeClusterIP,
				ClusterIP:  "1.2.3.4",
				ClusterIPs: []string{"1.2.3.4"},
			},
		}
		if tweak != nil {
			tweak(svc)
		}
		return svc
	}
	// Helper makes a mostly-valid ServiceList.  Test-cases can tweak it as needed.
	makeServiceList := func(tweak func(*api.ServiceList)) *api.ServiceList {
		list := &api.ServiceList{
			Items: []api.Service{{
				ObjectMeta: metav1.ObjectMeta{Name: "svc", Namespace: "ns"},
				Spec: api.ServiceSpec{
					Type:       api.ServiceTypeClusterIP,
					ClusterIP:  "1.2.3.4",
					ClusterIPs: []string{"1.2.3.4"},
				},
			}},
		}
		if tweak != nil {
			tweak(list)
		}
		return list
	}

	testCases := []struct {
		name   string
		input  runtime.Object
		expect runtime.Object
	}{{
		name:   "no change v4",
		input:  makeService(nil),
		expect: makeService(nil),
	}, {
		name: "missing clusterIPs v4",
		input: makeService(func(svc *api.Service) {
			svc.Spec.ClusterIPs = nil
		}),
		expect: makeService(nil),
	}, {
		name: "no change v6",
		input: makeService(func(svc *api.Service) {
			svc.Spec.ClusterIP = "2000::"
			svc.Spec.ClusterIPs = []string{"2000::"}
		}),
		expect: makeService(func(svc *api.Service) {
			svc.Spec.ClusterIP = "2000::"
			svc.Spec.ClusterIPs = []string{"2000::"}
		}),
	}, {
		name: "missing clusterIPs v6",
		input: makeService(func(svc *api.Service) {
			svc.Spec.ClusterIP = "2000::"
			svc.Spec.ClusterIPs = nil
		}),
		expect: makeService(func(svc *api.Service) {
			svc.Spec.ClusterIP = "2000::"
			svc.Spec.ClusterIPs = []string{"2000::"}
		}),
	}, {
		name:   "list, no change v4",
		input:  makeServiceList(nil),
		expect: makeServiceList(nil),
	}, {
		name: "list, missing clusterIPs v4",
		input: makeServiceList(func(list *api.ServiceList) {
			list.Items[0].Spec.ClusterIPs = nil
		}),
		expect: makeService(nil),
	}, {
		name:  "not Service or ServiceList",
		input: &api.Pod{},
	}}

	for _, tc := range testCases {
		makeStorage := func(t *testing.T) (*GenericREST, *etcd3testing.EtcdTestServer) {
			etcdStorage, server := registrytest.NewEtcdStorage(t, "")
			restOptions := generic.RESTOptions{
				StorageConfig:           etcdStorage,
				Decorator:               generic.UndecoratedStorage,
				DeleteCollectionWorkers: 1,
				ResourcePrefix:          "services",
			}

			_, cidr, err := net.ParseCIDR("10.0.0.0/24")
			if err != nil {
				t.Fatalf("failed to parse CIDR")
			}

			ipAllocs := map[api.IPFamily]ipallocator.Interface{
				api.IPv4Protocol: makeIPAllocator(cidr),
			}
			serviceStorage, _, err := NewGenericREST(restOptions, api.IPv4Protocol, ipAllocs, nil)
			if err != nil {
				t.Fatalf("unexpected error from REST storage: %v", err)
			}
			return serviceStorage, server
		}
		t.Run(tc.name, func(t *testing.T) {
			storage, server := makeStorage(t)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			tmp := tc.input.DeepCopyObject()
			storage.defaultOnRead(tmp)

			svc, ok := tmp.(*api.Service)
			if !ok {
				list, ok := tmp.(*api.ServiceList)
				if !ok {
					return
				}
				svc = &list.Items[0]
			}

			exp, ok := tc.expect.(*api.Service)
			if !ok {
				list, ok := tc.expect.(*api.ServiceList)
				if !ok {
					return
				}
				exp = &list.Items[0]
			}

			// Verify fields we know are affected
			if svc.Spec.ClusterIP != exp.Spec.ClusterIP {
				t.Errorf("clusterIP: expected %v, got %v", exp.Spec.ClusterIP, svc.Spec.ClusterIP)
			}
			if !reflect.DeepEqual(svc.Spec.ClusterIPs, exp.Spec.ClusterIPs) {
				t.Errorf("clusterIPs: expected %v, got %v", exp.Spec.ClusterIPs, svc.Spec.ClusterIPs)
			}
		})
	}
}

func TestServiceDefaulting(t *testing.T) {
	makeStorage := func(t *testing.T, ipFamilies []api.IPFamily) (*GenericREST, *StatusREST, *etcd3testing.EtcdTestServer) {
		etcdStorage, server := registrytest.NewEtcdStorage(t, "")
		restOptions := generic.RESTOptions{
			StorageConfig:           etcdStorage,
			Decorator:               generic.UndecoratedStorage,
			DeleteCollectionWorkers: 1,
			ResourcePrefix:          "services",
		}

		ipAllocs := map[api.IPFamily]ipallocator.Interface{}
		for _, fam := range ipFamilies {
			switch fam {
			case api.IPv4Protocol:
				_, cidr, _ := net.ParseCIDR("10.0.0.0/16")
				ipAllocs[fam] = makeIPAllocator(cidr)
			case api.IPv6Protocol:
				_, cidr, _ := net.ParseCIDR("2000::/108")
				ipAllocs[fam] = makeIPAllocator(cidr)
			}
		}

		serviceStorage, statusStorage, err := NewGenericREST(restOptions, ipFamilies[0], ipAllocs, nil)
		if err != nil {
			t.Fatalf("unexpected error from REST storage: %v", err)
		}
		return serviceStorage, statusStorage, server
	}

	testCases := []struct {
		name       string
		ipFamilies []api.IPFamily
	}{
		{
			name:       "IPv4 single stack cluster",
			ipFamilies: []api.IPFamily{api.IPv4Protocol},
		},
		{
			name:       "IPv6 single stack cluster",
			ipFamilies: []api.IPFamily{api.IPv6Protocol},
		},

		{
			name:       "IPv4, IPv6 dual stack cluster",
			ipFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		},
		{
			name:       "IPv6, IPv4 dual stack cluster",
			ipFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
		},
	}

	singleStack := api.IPFamilyPolicySingleStack
	preferDualStack := api.IPFamilyPolicyPreferDualStack

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			// this func only works with dual stack feature gate on.
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, true)()

			storage, _, server := makeStorage(t, testCase.ipFamilies)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			undefaultedServiceList, defaultedServiceList := makeServiceList()
			// set the two special ones (0: w/ selector, 1: w/o selector)
			// review default*OnRead(...)
			// Single stack cluster:
			// headless w/selector => singlestack
			// headless w/o selector => preferDualStack
			// dual stack cluster:
			// headless w/selector => preferDualStack
			// headless w/o selector => preferDualStack

			// assume single stack
			defaultedServiceList.Items[0].Spec.IPFamilyPolicy = &singleStack

			// primary family
			if testCase.ipFamilies[0] == api.IPv6Protocol {
				// no selector, gets both families
				defaultedServiceList.Items[1].Spec.IPFamilyPolicy = &preferDualStack
				defaultedServiceList.Items[1].Spec.IPFamilies = []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol}

				//assume single stack for w/selector
				defaultedServiceList.Items[0].Spec.IPFamilies = []api.IPFamily{api.IPv6Protocol}
				// make dualstacked. if needed
				if len(testCase.ipFamilies) > 1 {
					defaultedServiceList.Items[0].Spec.IPFamilyPolicy = &preferDualStack
					defaultedServiceList.Items[0].Spec.IPFamilies = append(defaultedServiceList.Items[0].Spec.IPFamilies, api.IPv4Protocol)
				}
			} else {
				// no selector gets both families
				defaultedServiceList.Items[1].Spec.IPFamilyPolicy = &preferDualStack
				defaultedServiceList.Items[1].Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol}

				// assume single stack for w/selector
				defaultedServiceList.Items[0].Spec.IPFamilies = []api.IPFamily{api.IPv4Protocol}
				// make dualstacked. if needed
				if len(testCase.ipFamilies) > 1 {
					defaultedServiceList.Items[0].Spec.IPFamilyPolicy = &preferDualStack
					defaultedServiceList.Items[0].Spec.IPFamilies = append(defaultedServiceList.Items[0].Spec.IPFamilies, api.IPv6Protocol)
				}
			}

			// data is now ready for testing over various cluster configuration
			compareSvc := func(out api.Service, expected api.Service) {
				if expected.Spec.IPFamilyPolicy == nil && out.Spec.IPFamilyPolicy != nil {
					t.Fatalf("service %+v expected IPFamilyPolicy to be nil", out)
				}
				if expected.Spec.IPFamilyPolicy != nil && out.Spec.IPFamilyPolicy == nil {
					t.Fatalf("service %+v expected IPFamilyPolicy not to be nil", out)
				}

				if expected.Spec.IPFamilyPolicy != nil {
					if *out.Spec.IPFamilyPolicy != *expected.Spec.IPFamilyPolicy {
						t.Fatalf("service %+v expected IPFamilyPolicy %v got %v", out, *expected.Spec.IPFamilyPolicy, *out.Spec.IPFamilyPolicy)
					}
				}

				if len(out.Spec.IPFamilies) != len(expected.Spec.IPFamilies) {
					t.Fatalf("service %+v expected len(IPFamilies) == %v", out, len(expected.Spec.IPFamilies))
				}
				for i, ipfamily := range out.Spec.IPFamilies {
					if expected.Spec.IPFamilies[i] != ipfamily {
						t.Fatalf("service %+v expected ip families %+v", out, expected.Spec.IPFamilies)
					}
				}
			}

			copyUndefaultedList := undefaultedServiceList.DeepCopy()
			// run for each Service
			for i, svc := range copyUndefaultedList.Items {
				storage.defaultOnRead(&svc)
				compareSvc(svc, defaultedServiceList.Items[i])
			}

			copyUndefaultedList = undefaultedServiceList.DeepCopy()
			// run as a ServiceList
			storage.defaultOnRead(copyUndefaultedList)
			for i, svc := range copyUndefaultedList.Items {
				compareSvc(svc, defaultedServiceList.Items[i])
			}

			// if there are more tests needed then the last call need to work
			// with copy of undefaulted list since
		})
	}
}

func fmtIPFamilyPolicy(pol *api.IPFamilyPolicyType) string {
	if pol == nil {
		return "<nil>"
	}
	return string(*pol)
}

func fmtIPFamilies(fams []api.IPFamily) string {
	if fams == nil {
		return "[]"
	}
	return fmt.Sprintf("%v", fams)
}

func familyOf(ip string) api.IPFamily {
	if utilnet.IsIPv4String(ip) {
		return api.IPv4Protocol
	}
	if utilnet.IsIPv6String(ip) {
		return api.IPv6Protocol
	}
	return api.IPFamily("unknown")
}

// Prove that create ignores IPFamily stuff when type is ExternalName.
func TestCreateIgnoresIPFamilyForExternalName(t *testing.T) {
	type testCase struct {
		name           string
		svc            *api.Service
		expectError    bool
		expectPolicy   *api.IPFamilyPolicyType
		expectFamilies []api.IPFamily
	}
	// These cases were chosen from the full gamut to ensure all "interesting"
	// cases are covered.
	testCases := []struct {
		name            string
		clusterFamilies []api.IPFamily
		enableDualStack bool
		cases           []testCase
	}{{
		name:            "singlestack:v4_gate:off",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol},
		enableDualStack: false,
		cases: []testCase{{
			name:           "Policy:unset_Families:unset",
			svc:            svctest.MakeService("foo"),
			expectPolicy:   nil,
			expectFamilies: nil,
		}, {
			name: "Policy:SingleStack_Families:v4",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectPolicy:   nil,
			expectFamilies: nil,
		}, {
			name: "Policy:PreferDualStack_Families:v4v6",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectPolicy:   nil,
			expectFamilies: nil,
		}, {
			name: "Policy:RequireDualStack_Families:v6v4",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectPolicy:   nil,
			expectFamilies: nil,
		}},
	}, {
		name:            "singlestack:v6_gate:on",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol},
		enableDualStack: true,
		cases: []testCase{{
			name:           "Policy:unset_Families:unset",
			svc:            svctest.MakeService("foo"),
			expectPolicy:   nil,
			expectFamilies: nil,
		}, {
			name: "Policy:SingleStack_Families:v6",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectError: true,
		}, {
			name: "Policy:PreferDualStack_Families:v4v6",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectError: true,
		}, {
			name: "Policy:RequireDualStack_Families:v6v4",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectError: true,
		}},
	}, {
		name:            "dualstack:v4v6_gate:off",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		enableDualStack: false,
		cases: []testCase{{
			name:           "Policy:unset_Families:unset",
			svc:            svctest.MakeService("foo"),
			expectPolicy:   nil,
			expectFamilies: nil,
		}, {
			name: "Policy:SingleStack_Families:v4",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectPolicy:   nil,
			expectFamilies: nil,
		}, {
			name: "Policy:PreferDualStack_Families:v4v6",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectPolicy:   nil,
			expectFamilies: nil,
		}, {
			name: "Policy:RequireDualStack_Families:v6v4",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectPolicy:   nil,
			expectFamilies: nil,
		}},
	}, {
		name:            "dualstack:v6v4_gate:on",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
		enableDualStack: true,
		cases: []testCase{{
			name:           "Policy:unset_Families:unset",
			svc:            svctest.MakeService("foo"),
			expectPolicy:   nil,
			expectFamilies: nil,
		}, {
			name: "Policy:SingleStack_Families:v6",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectError: true,
		}, {
			name: "Policy:PreferDualStack_Families:v6v4",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectError: true,
		}, {
			name: "Policy:RequireDualStack_Families:v4v6",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectError: true,
		}},
	}}

	for _, otc := range testCases {
		t.Run(otc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, otc.enableDualStack)()

			storage, _, server := newStorage(t, otc.clusterFamilies)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			for _, itc := range otc.cases {
				t.Run(itc.name, func(t *testing.T) {
					// This test is ONLY ExternalName services.
					itc.svc.Spec.Type = api.ServiceTypeExternalName
					itc.svc.Spec.ExternalName = "example.com"

					ctx := genericapirequest.NewDefaultContext()
					createdObj, err := storage.Create(ctx, itc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
					if itc.expectError && err != nil {
						return
					}
					if err != nil {
						t.Fatalf("unexpected error creating service: %v", err)
					}
					defer storage.Delete(ctx, itc.svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
					if itc.expectError && err == nil {
						t.Fatalf("unexpected success creating service")
					}
					createdSvc := createdObj.(*api.Service)

					if want, got := fmtIPFamilyPolicy(itc.expectPolicy), fmtIPFamilyPolicy(createdSvc.Spec.IPFamilyPolicy); want != got {
						t.Errorf("wrong IPFamilyPolicy: want %s, got %s", want, got)
					}
					if want, got := fmtIPFamilies(itc.expectFamilies), fmtIPFamilies(createdSvc.Spec.IPFamilies); want != got {
						t.Errorf("wrong IPFamilies: want %s, got %s", want, got)
					}
				})
			}
		})
	}
}

// Prove that create ignores IPFamily stuff when dual-stack is disabled.
func TestCreateIgnoresIPFamilyWithoutDualStack(t *testing.T) {
	// These cases were chosen from the full gamut to ensure all "interesting"
	// cases are covered.
	testCases := []struct {
		name string
		svc  *api.Service
	}{
		//----------------------------------------
		// ClusterIP:unset
		//----------------------------------------
		{
			name: "ClusterIP:unset_Policy:unset_Families:unset",
			svc:  svctest.MakeService("foo"),
		}, {
			name: "ClusterIP:unset_Policy:unset_Families:v4",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv4Protocol)),
		}, {
			name: "ClusterIP:unset_Policy:unset_Families:v6v4",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
		}, {
			name: "ClusterIP:unset_Policy:SingleStack_Families:unset",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
		}, {
			name: "ClusterIP:unset_Policy:SingleStack_Families:v4",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv4Protocol)),
		}, {
			name: "ClusterIP:unset_Policy:SingleStack_Families:v6v4",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
		}, {
			name: "ClusterIP:unset_Policy:PreferDualStack_Families:unset",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
		}, {
			name: "ClusterIP:unset_Policy:PreferDualStack_Families:v4",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol)),
		}, {
			name: "ClusterIP:unset_Policy:PreferDualStack_Families:v6v4",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
		}, {
			name: "ClusterIP:unset_Policy:RequireDualStack_Families:unset",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
		}, {
			name: "ClusterIP:unset_Policy:RequireDualStack_Families:v4",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol)),
		}, {
			name: "ClusterIP:unset_Policy:RequireDualStack_Families:v6v4",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
		},
		//----------------------------------------
		// ClusterIPs:v4v6
		//----------------------------------------
		{
			name: "ClusterIPs:v4v6_Policy:unset_Families:unset",
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
		}, {
			name: "ClusterIPs:v4v6_Policy:unset_Families:v4",
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("10.0.0.1", "2000::1"),
				svctest.SetIPFamilies(api.IPv4Protocol)),
		}, {
			name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:unset",
			svc: svctest.MakeService("foo",
				svctest.SetClusterIPs("10.0.0.1", "2000::1"),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
		},
		//----------------------------------------
		// Headless
		//----------------------------------------
		{
			name: "Headless_Policy:unset_Families:unset",
			svc: svctest.MakeService("foo",
				svctest.SetHeadless),
		}, {
			name: "Headless_Policy:unset_Families:v4",
			svc: svctest.MakeService("foo",
				svctest.SetHeadless,
				svctest.SetIPFamilies(api.IPv4Protocol)),
		}, {
			name: "Headless_Policy:RequireDualStack_Families:unset",
			svc: svctest.MakeService("foo",
				svctest.SetHeadless,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
		},
		//----------------------------------------
		// HeadlessSelectorless
		//----------------------------------------
		{
			name: "HeadlessSelectorless_Policy:unset_Families:unset",
			svc: svctest.MakeService("foo",
				svctest.SetHeadless,
				svctest.SetSelector(nil)),
		}, {
			name: "HeadlessSelectorless_Policy:unset_Families:v4",
			svc: svctest.MakeService("foo",
				svctest.SetHeadless,
				svctest.SetSelector(nil),
				svctest.SetIPFamilies(api.IPv4Protocol)),
		}, {
			name: "HeadlessSelectorless_Policy:RequireDualStack_Families:unset",
			svc: svctest.MakeService("foo",
				svctest.SetHeadless,
				svctest.SetSelector(nil),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
		},
	}

	// This test is ONLY with the gate off.
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, false)()

	// Do this in the outer scope for performance.
	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := genericapirequest.NewDefaultContext()
			createdObj, err := storage.Create(ctx, tc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("unexpected error creating service: %v", err)
			}
			defer storage.Delete(ctx, tc.svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
			createdSvc := createdObj.(*api.Service)
			//FIXME: HACK!!  Delete above calls "inner" which doesn't
			// yet call the allocators - no release = alloc errors!
			defer func() {
				for _, al := range storage.alloc.serviceIPAllocatorsByFamily {
					for _, ip := range createdSvc.Spec.ClusterIPs {
						al.Release(net.ParseIP(ip))
					}
				}
			}()

			// The gate is off - these should always be empty.
			if want, got := fmtIPFamilyPolicy(nil), fmtIPFamilyPolicy(createdSvc.Spec.IPFamilyPolicy); want != got {
				t.Errorf("wrong IPFamilyPolicy: want %s, got %s", want, got)
			}
			if want, got := fmtIPFamilies(nil), fmtIPFamilies(createdSvc.Spec.IPFamilies); want != got {
				t.Errorf("wrong IPFamilies: want %s, got %s", want, got)
			}
		})
	}
}

// Prove that create initializes clusterIPs from clusterIP.  This simplifies
// later tests to not need to re-prove this.
func TestCreateInitClusterIPsFromClusterIP(t *testing.T) {
	testCases := []struct {
		name            string
		clusterFamilies []api.IPFamily
		svc             *api.Service
	}{{
		name:            "singlestack:v4_clusterip:unset",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol},
		svc:             svctest.MakeService("foo"),
	}, {
		name:            "singlestack:v4_clusterip:set",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetClusterIP("10.0.0.1")),
	}, {
		name:            "singlestack:v6_clusterip:unset",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol},
		svc:             svctest.MakeService("foo"),
	}, {
		name:            "singlestack:v6_clusterip:set",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetClusterIP("2000::1")),
	}, {
		name:            "dualstack:v4v6_clusterip:unset",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		svc:             svctest.MakeService("foo"),
	}, {
		name:            "dualstack:v4v6_clusterip:set",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetClusterIP("10.0.0.1")),
	}, {
		name:            "dualstack:v6v4_clusterip:unset",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
		svc:             svctest.MakeService("foo"),
	}, {
		name:            "dualstack:v6v4_clusterip:set",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetClusterIP("2000::1")),
	}}

	// This test is ONLY with the gate enabled.
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, true)()

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			storage, _, server := newStorage(t, tc.clusterFamilies)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			ctx := genericapirequest.NewDefaultContext()
			createdObj, err := storage.Create(ctx, tc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("unexpected error creating service: %v", err)
			}
			createdSvc := createdObj.(*api.Service)

			if createdSvc.Spec.ClusterIP == "" {
				t.Errorf("expected ClusterIP to be set")

			}
			if tc.svc.Spec.ClusterIP != "" {
				if want, got := tc.svc.Spec.ClusterIP, createdSvc.Spec.ClusterIP; want != got {
					t.Errorf("wrong ClusterIP: want %s, got %s", want, got)
				}
			}
			if len(createdSvc.Spec.ClusterIPs) == 0 {
				t.Errorf("expected ClusterIPs to be set")
			}
			if want, got := createdSvc.Spec.ClusterIP, createdSvc.Spec.ClusterIPs[0]; want != got {
				t.Errorf("wrong ClusterIPs[0]: want %s, got %s", want, got)
			}
		})
	}
}

// Prove that create initializes IPFamily fields correctly.
func TestCreateInitIPFields(t *testing.T) {
	type testCase struct {
		name           string
		svc            *api.Service
		expectError    bool
		expectPolicy   api.IPFamilyPolicyType
		expectFamilies []api.IPFamily
		expectHeadless bool
	}
	// These cases were chosen from the full gamut to ensure all "interesting"
	// cases are covered.
	testCases := []struct {
		name            string
		clusterFamilies []api.IPFamily
		cases           []testCase
	}{
		{
			name:            "singlestack:v4",
			clusterFamilies: []api.IPFamily{api.IPv4Protocol},
			cases: []testCase{
				//----------------------------------------
				// singlestack:v4 ClusterIPs:unset
				//----------------------------------------
				{
					name:           "ClusterIPs:unset_Policy:unset_Families:unset",
					svc:            svctest.MakeService("foo"),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v4 ClusterIPs:v4
				//----------------------------------------
				{
					name: "ClusterIPs:v4_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1")),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v4 ClusterIPs:v4v6
				//----------------------------------------
				{
					name: "ClusterIPs:v4v6_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1")),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v4 ClusterIPs:v6v4
				//----------------------------------------
				{
					name: "ClusterIPs:v6v4_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1")),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v4 Headless
				//----------------------------------------
				{
					name: "Headless_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v4 HeadlessSelectorless
				//----------------------------------------
				{
					name: "HeadlessSelectorless_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				},
			},
		}, {
			name:            "singlestack:v6",
			clusterFamilies: []api.IPFamily{api.IPv6Protocol},
			cases: []testCase{
				//----------------------------------------
				// singlestack:v6 ClusterIPs:unset
				//----------------------------------------
				{
					name:           "ClusterIPs:unset_Policy:unset_Families:unset",
					svc:            svctest.MakeService("foo"),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v6 ClusterIPs:v6
				//----------------------------------------
				{
					name: "ClusterIPs:v6_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1")),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v6 ClusterIPs:v4v6
				//----------------------------------------
				{
					name: "ClusterIPs:v4v6_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1")),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v6 ClusterIPs:v6v4
				//----------------------------------------
				{
					name: "ClusterIPs:v6v4_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1")),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v6 Headless
				//----------------------------------------
				{
					name: "Headless_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v6 HeadlessSelectorless
				//----------------------------------------
				{
					name: "HeadlessSelectorless_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				},
			},
		}, {
			name:            "dualstack:v4v6",
			clusterFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			cases: []testCase{
				//----------------------------------------
				// dualstack:v4v6 ClusterIPs:unset
				//----------------------------------------
				{
					name:           "ClusterIPs:unset_Policy:unset_Families:unset",
					svc:            svctest.MakeService("foo"),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				},
				//----------------------------------------
				// dualstack:v4v6 ClusterIPs:v4
				//----------------------------------------
				{
					name: "ClusterIPs:v4_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1")),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// dualstack:v4v6 ClusterIPs:v6
				//----------------------------------------
				{
					name: "ClusterIPs:v6_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1")),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				},
				//----------------------------------------
				// dualstack:v4v6 ClusterIPs:v4v6
				//----------------------------------------
				{
					name: "ClusterIPs:v4v6_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1")),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// dualstack:v4v6 ClusterIPs:v6v4
				//----------------------------------------
				{
					name: "ClusterIPs:v6v4_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1")),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				},
				//----------------------------------------
				// dualstack:v4v6 Headless
				//----------------------------------------
				{
					name: "Headless_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				},
				//----------------------------------------
				// dualstack:v4v6 HeadlessSelectorless
				//----------------------------------------
				{
					name: "HeadlessSelectorless_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				},
			},
		}, {
			name:            "dualstack:v6v4",
			clusterFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			cases: []testCase{
				//----------------------------------------
				// dualstack:v6v4 ClusterIPs:unset
				//----------------------------------------
				{
					name:           "ClusterIPs:unset_Policy:unset_Families:unset",
					svc:            svctest.MakeService("foo"),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				},
				//----------------------------------------
				// dualstack:v6v4 ClusterIPs:v4
				//----------------------------------------
				{
					name: "ClusterIPs:v4_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1")),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// dualstack:v6v4 ClusterIPs:v6
				//----------------------------------------
				{
					name: "ClusterIPs:v6_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1")),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				},
				//----------------------------------------
				// dualstack:v6v4 ClusterIPs:v4v6
				//----------------------------------------
				{
					name: "ClusterIPs:v4v6_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1")),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// dualstack:v6v4 ClusterIPs:v6v4
				//----------------------------------------
				{
					name: "ClusterIPs:v6v4_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1")),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				},
				//----------------------------------------
				// dualstack:v6v4 Headless
				//----------------------------------------
				{
					name: "Headless_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				},
				//----------------------------------------
				// dualstack:v6v4 HeadlessSelectorless
				//----------------------------------------
				{
					name: "HeadlessSelectorless_Policy:unset_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:unset",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v4v6",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v6v4",
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				},
			},
		},
	}

	// This test is ONLY with the gate enabled.
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, true)()

	for _, otc := range testCases {
		t.Run(otc.name, func(t *testing.T) {

			// Do this in the outer loop for performance.
			storage, _, server := newStorage(t, otc.clusterFamilies)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			for _, itc := range otc.cases {
				t.Run(itc.name, func(t *testing.T) {
					ctx := genericapirequest.NewDefaultContext()
					createdObj, err := storage.Create(ctx, itc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
					if itc.expectError && err != nil {
						return
					}
					if err != nil {
						t.Fatalf("unexpected error creating service: %v", err)
					}
					defer storage.Delete(ctx, itc.svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
					if itc.expectError && err == nil {
						t.Fatalf("unexpected success creating service")
					}
					createdSvc := createdObj.(*api.Service)
					//FIXME: HACK!!  Delete above calls "inner" which doesn't
					// yet call the allocators - no release = alloc errors!
					defer func() {
						for _, al := range storage.alloc.serviceIPAllocatorsByFamily {
							for _, ip := range createdSvc.Spec.ClusterIPs {
								al.Release(net.ParseIP(ip))
							}
						}
					}()

					if want, got := fmtIPFamilyPolicy(&itc.expectPolicy), fmtIPFamilyPolicy(createdSvc.Spec.IPFamilyPolicy); want != got {
						t.Errorf("wrong IPFamilyPolicy: want %s, got %s", want, got)
					}
					if want, got := fmtIPFamilies(itc.expectFamilies), fmtIPFamilies(createdSvc.Spec.IPFamilies); want != got {
						t.Errorf("wrong IPFamilies: want %s, got %s", want, got)
					}
					if itc.expectHeadless {
						if !reflect.DeepEqual(createdSvc.Spec.ClusterIPs, []string{"None"}) {
							t.Errorf("wrong clusterIPs: want [\"None\"], got %v", createdSvc.Spec.ClusterIPs)
						}
					} else {
						if c, f := len(createdSvc.Spec.ClusterIPs), len(createdSvc.Spec.IPFamilies); c != f {
							t.Errorf("clusterIPs and ipFamilies are not the same length: %d vs %d", c, f)
						}
						for i, clip := range createdSvc.Spec.ClusterIPs {
							if cf, ef := familyOf(clip), createdSvc.Spec.IPFamilies[i]; cf != ef {
								t.Errorf("clusterIP is the wrong IP family: want %s, got %s", ef, cf)
							}
						}
					}
				})
			}
		})
	}
}

// Prove that a dry-run create doesn't actually allocate IPs or ports.
func TestCreateDryRun(t *testing.T) {
	testCases := []struct {
		name            string
		clusterFamilies []api.IPFamily
		enableDualStack bool
		svc             *api.Service
	}{{
		name:            "singlestack:v4_gate:off_clusterip:unset",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol},
		enableDualStack: false,
		svc:             svctest.MakeService("foo"),
	}, {
		name:            "singlestack:v4_gate:off_clusterip:set",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol},
		enableDualStack: false,
		svc:             svctest.MakeService("foo", svctest.SetClusterIPs("10.0.0.1")),
	}, {
		name:            "singlestack:v6_gate:on_clusterip:unset",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol},
		enableDualStack: true,
		svc:             svctest.MakeService("foo"),
	}, {
		name:            "singlestack:v6_gate:on_clusterip:set",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol},
		enableDualStack: true,
		svc:             svctest.MakeService("foo", svctest.SetClusterIPs("2000::1")),
	}, {
		name:            "dualstack:v4v6_gate:on_clusterip:unset",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		enableDualStack: true,
		svc:             svctest.MakeService("foo", svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
	}, {
		name:            "dualstack:v4v6_gate:on_clusterip:set",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		enableDualStack: true,
		svc:             svctest.MakeService("foo", svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack), svctest.SetClusterIPs("10.0.0.1", "2000::1")),
	}, {
		name:            "singlestack:v4_gate:off_type:NodePort_nodeport:unset",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol},
		enableDualStack: false,
		svc:             svctest.MakeService("foo", svctest.SetTypeNodePort),
	}, {
		name:            "singlestack:v4_gate:on_type:LoadBalancer_nodePort:set",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol},
		enableDualStack: true,
		svc:             svctest.MakeService("foo", svctest.SetTypeLoadBalancer, svctest.SetUniqueNodePorts),
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, tc.enableDualStack)()

			storage, _, server := newStorage(t, tc.clusterFamilies)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			ctx := genericapirequest.NewDefaultContext()
			createdObj, err := storage.Create(ctx, tc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			if err != nil {
				t.Fatalf("unexpected error creating service: %v", err)
			}
			createdSvc := createdObj.(*api.Service)

			// Ensure IPs were allocated
			if net.ParseIP(createdSvc.Spec.ClusterIP) == nil {
				t.Errorf("expected valid clusterIP: %v", createdSvc.Spec.ClusterIP)
			}

			// Ensure the IP allocators are clean.
			if !tc.enableDualStack {
				if ipIsAllocated(t, storage.alloc.serviceIPAllocatorsByFamily[api.IPv4Protocol], createdSvc.Spec.ClusterIP) {
					t.Errorf("expected IP to not be allocated: %v", createdSvc.Spec.ClusterIP)
				}
			} else {
				for _, ip := range createdSvc.Spec.ClusterIPs {
					if net.ParseIP(ip) == nil {
						t.Errorf("expected valid clusterIP: %v", createdSvc.Spec.ClusterIP)
					}
				}
				for i, fam := range createdSvc.Spec.IPFamilies {
					if ipIsAllocated(t, storage.alloc.serviceIPAllocatorsByFamily[fam], createdSvc.Spec.ClusterIPs[i]) {
						t.Errorf("expected IP to not be allocated: %v", createdSvc.Spec.ClusterIPs[i])
					}
				}
			}

			if tc.svc.Spec.Type != api.ServiceTypeClusterIP {
				for _, p := range createdSvc.Spec.Ports {
					if portIsAllocated(t, storage.alloc.serviceNodePorts, p.NodePort) {
						t.Errorf("expected port to not be allocated: %v", p.NodePort)
					}
				}
			}
		})
	}
}
