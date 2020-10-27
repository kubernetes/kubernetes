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
	"net"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/registrytest"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func newStorage(t *testing.T) (*GenericREST, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "services",
	}
	serviceStorage, statusStorage, err := NewGenericREST(restOptions, *makeIPNet(t), false)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return serviceStorage, statusStorage, server
}

func validService() *api.Service {
	singleStack := api.IPFamilyPolicySingleStack

	return &api.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			ClusterIP:       api.ClusterIPNone,
			ClusterIPs:      []string{api.ClusterIPNone},
			IPFamilyPolicy:  &singleStack,
			IPFamilies:      []api.IPFamily{api.IPv4Protocol},
			SessionAffinity: "None",
			Type:            api.ServiceTypeClusterIP,
			Ports: []api.ServicePort{{
				Port:       6502,
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(6502),
			}},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	validService := validService()
	validService.ObjectMeta = metav1.ObjectMeta{}
	test.TestCreate(
		// valid
		validService,
		// invalid
		&api.Service{
			Spec: api.ServiceSpec{},
		},
		// invalid
		&api.Service{
			Spec: api.ServiceSpec{
				Selector:        map[string]string{"bar": "baz"},
				ClusterIPs:      []string{"invalid"},
				SessionAffinity: "None",
				Type:            api.ServiceTypeClusterIP,
				Ports: []api.ServicePort{{
					Port:       6502,
					Protocol:   api.ProtocolTCP,
					TargetPort: intstr.FromInt(6502),
				}},
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, _, server := newStorage(t)
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
			}
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).AllowCreateOnUpdate().ReturnDeletedObject()
	test.TestDelete(validService())
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).AllowCreateOnUpdate()
	test.TestGet(validService())
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).AllowCreateOnUpdate()
	test.TestList(validService())
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
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

func TestShortNames(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"svc"}
	registrytest.AssertShortNames(t, storage, expected)
}

func TestCategories(t *testing.T) {
	storage, _, server := newStorage(t)
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
		name      string
		input     runtime.Object
		expectErr bool
		expect    runtime.Object
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
		name:      "not Service or ServiceList",
		input:     &api.Pod{},
		expectErr: false,
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

			serviceStorage, _, err := NewGenericREST(restOptions, *cidr, false)
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
			err := storage.defaultOnRead(tmp)
			if err != nil && !tc.expectErr {
				t.Errorf("unexpected error: %v", err)
			}
			if err == nil && tc.expectErr {
				t.Errorf("unexpected success")
			}

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
	makeStorage := func(t *testing.T, primaryCIDR string, isDualStack bool) (*GenericREST, *StatusREST, *etcd3testing.EtcdTestServer) {
		etcdStorage, server := registrytest.NewEtcdStorage(t, "")
		restOptions := generic.RESTOptions{
			StorageConfig:           etcdStorage,
			Decorator:               generic.UndecoratedStorage,
			DeleteCollectionWorkers: 1,
			ResourcePrefix:          "services",
		}

		_, cidr, err := net.ParseCIDR(primaryCIDR)
		if err != nil {
			t.Fatalf("failed to parse CIDR %s", primaryCIDR)
		}

		serviceStorage, statusStorage, err := NewGenericREST(restOptions, *(cidr), isDualStack)
		if err != nil {
			t.Fatalf("unexpected error from REST storage: %v", err)
		}
		return serviceStorage, statusStorage, server
	}

	testCases := []struct {
		name        string
		primaryCIDR string
		PrimaryIPv6 bool
		isDualStack bool
	}{
		{
			name:        "IPv4 single stack cluster",
			primaryCIDR: "10.0.0.0/16",
			PrimaryIPv6: false,
			isDualStack: false,
		},
		{
			name:        "IPv6 single stack cluster",
			primaryCIDR: "2000::/108",
			PrimaryIPv6: true,
			isDualStack: false,
		},

		{
			name:        "IPv4, IPv6 dual stack cluster",
			primaryCIDR: "10.0.0.0/16",
			PrimaryIPv6: false,
			isDualStack: true,
		},
		{
			name:        "IPv6, IPv4 dual stack cluster",
			primaryCIDR: "2000::/108",
			PrimaryIPv6: true,
			isDualStack: true,
		},
	}

	singleStack := api.IPFamilyPolicySingleStack
	preferDualStack := api.IPFamilyPolicyPreferDualStack

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			// this func only works with dual stack feature gate on.
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, true)()

			storage, _, server := makeStorage(t, testCase.primaryCIDR, testCase.isDualStack)
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
			if testCase.PrimaryIPv6 {
				// no selector, gets both families
				defaultedServiceList.Items[1].Spec.IPFamilyPolicy = &preferDualStack
				defaultedServiceList.Items[1].Spec.IPFamilies = []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol}

				//assume single stack for w/selector
				defaultedServiceList.Items[0].Spec.IPFamilies = []api.IPFamily{api.IPv6Protocol}
				// make dualstacked. if needed
				if testCase.isDualStack {
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
				if testCase.isDualStack {
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
