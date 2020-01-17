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
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/intstr"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*GenericREST, *StatusREST, *etcd3testing.EtcdTestServer) {
	return newStorageWithOptions(t, *makeIPNet(t), false)
}

func newStorageWithOptions(t *testing.T, cidr net.IPNet, hasSecondary bool) (*GenericREST, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "services",
	}
	serviceStorage, statusStorage, err := NewGenericREST(restOptions, cidr, hasSecondary)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return serviceStorage, statusStorage, server
}

func validService() *api.Service {
	return &api.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"bar": "baz"},
			ClusterIP:       "None",
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
				ClusterIP:       "invalid",
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
				ClusterIP:       "None",
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

func TestGetDecorator(t *testing.T) {
	ipv4 := api.IPv4Protocol
	ipv6 := api.IPv6Protocol
	testCases := []struct {
		name            string
		enableDualStack bool
		networkCIDR     net.IPNet
		hasSecondary    bool
		svc             *api.Service
		storageIPFamily *api.IPFamily
		expect          *api.Service
	}{
		{
			name:         "unset",
			networkCIDR:  *makeIPNet(t),
			hasSecondary: false,

			svc: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeExternalName,
					ExternalName:    "somewhere.test.com",
				},
			},
			storageIPFamily: nil,

			expect: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeExternalName,
					ExternalName:    "somewhere.test.com",
					IPFamily:        nil,
				},
			},
		},
		{
			name:         "storage has ipv4 but the gate is disabled",
			networkCIDR:  *makeIPNet(t),
			hasSecondary: false,

			svc: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeExternalName,
					ExternalName:    "somewhere.test.com",
				},
			},
			storageIPFamily: &ipv4,

			expect: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeExternalName,
					ExternalName:    "somewhere.test.com",
					IPFamily:        &ipv4,
				},
			},
		},
		{
			name:            "default IPFamily to ipv6",
			enableDualStack: true,
			networkCIDR:     *makeIPNet6(t),
			hasSecondary:    false,

			svc: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeExternalName,
					ExternalName:    "somewhere.test.com",
				},
			},
			storageIPFamily: nil,

			expect: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeExternalName,
					ExternalName:    "somewhere.test.com",
					IPFamily:        &ipv6,
				},
			},
		},
		{
			name:            "default IPFamily to ipv4 based on cluster IP when stored version is nil",
			enableDualStack: true,
			networkCIDR:     *makeIPNet6(t),
			hasSecondary:    true,

			svc: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeClusterIP,
					IPFamily:        &ipv4,
					ClusterIP:       "172.0.0.1",
					Ports: []api.ServicePort{{
						Port:       6502,
						Protocol:   api.ProtocolTCP,
						TargetPort: intstr.FromInt(6502),
					}},
				},
			},
			storageIPFamily: nil,

			expect: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector:        map[string]string{"bar": "baz"},
					SessionAffinity: api.ServiceAffinityNone,
					Type:            api.ServiceTypeClusterIP,
					IPFamily:        &ipv4,
					ClusterIP:       "172.0.0.1",
					Ports: []api.ServicePort{{
						Port:       6502,
						Protocol:   api.ProtocolTCP,
						TargetPort: intstr.FromInt(6502),
					}},
				},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, tc.enableDualStack)()
			storage, _, server := newStorageWithOptions(t, tc.networkCIDR, tc.hasSecondary)
			// abuse TTLFunc to allow us to mutate the object prior to creation
			// TODO: possibly this should be a real hook for testing
			storage.TTLFunc = func(obj runtime.Object, existing uint64, update bool) (uint64, error) {
				obj.(*api.Service).Spec.IPFamily = tc.storageIPFamily
				return 0, nil
			}
			defer server.Terminate(t)

			ctx := genericapirequest.NewDefaultContext()
			_, err := storage.Create(ctx, tc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			obj, err := storage.Get(ctx, tc.svc.Name, &metav1.GetOptions{})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if svc, _ := obj.(*api.Service); svc != nil {
				tc.expect.ObjectMeta = svc.ObjectMeta
			}
			if !reflect.DeepEqual(tc.expect, obj) {
				t.Fatalf("unexpected object: %s", diff.ObjectReflectDiff(tc.expect, obj))
			}
		})
	}

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
