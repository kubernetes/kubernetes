/*
Copyright 2022 The Kubernetes Authors.

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
	"testing"

	"github.com/google/go-cmp/cmp"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/apis/resource"
	_ "k8s.io/kubernetes/pkg/apis/resource/install"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, resource.GroupName)
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "resourceclaims",
	}
	resourceClaimStorage, statusStorage, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return resourceClaimStorage, statusStorage, server
}

func validNewClaim(name, ns string) *resource.ResourceClaim {
	claim := &resource.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
	}
	return claim
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	claim := validNewClaim("foo", metav1.NamespaceDefault)
	claim.ObjectMeta = metav1.ObjectMeta{}
	test.TestCreate(
		// valid
		claim,
		// invalid
		&resource.ResourceClaim{
			ObjectMeta: metav1.ObjectMeta{Name: "*BadName!"},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestUpdate(
		// valid
		validNewClaim("foo", metav1.NamespaceDefault),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*resource.ResourceClaim)
			if object.Labels == nil {
				object.Labels = map[string]string{}
			}
			object.Labels["foo"] = "bar"
			return object
		},
		// invalid update
		func(obj runtime.Object) runtime.Object {
			object := obj.(*resource.ResourceClaim)
			object.Name = "^%$#@#%"
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ReturnDeletedObject()
	test.TestDelete(validNewClaim("foo", metav1.NamespaceDefault))
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestGet(validNewClaim("foo", metav1.NamespaceDefault))
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestList(validNewClaim("foo", metav1.NamespaceDefault))
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestWatch(
		validNewClaim("foo", metav1.NamespaceDefault),
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

func TestUpdateStatus(t *testing.T) {
	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	ctx := genericapirequest.NewDefaultContext()

	key, _ := storage.KeyFunc(ctx, "foo")
	claimStart := validNewClaim("foo", metav1.NamespaceDefault)
	err := storage.Storage.Create(ctx, key, claimStart, nil, 0, false)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	claim := claimStart.DeepCopy()
	claim.Status.Allocation = &resource.AllocationResult{}
	_, _, err = statusStorage.Update(ctx, claim.Name, rest.DefaultUpdatedObjectInfo(claim), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	claimOut := obj.(*resource.ResourceClaim)
	// only compare relevant changes b/c of difference in metadata
	if !apiequality.Semantic.DeepEqual(claim.Status, claimOut.Status) {
		t.Errorf("unexpected object: %s", cmp.Diff(claim.Status, claimOut.Status))
	}
}

type testMetrics struct {
	attempts float64
	failures float64
}

func expectMetrics(t *testing.T, em testMetrics) {
	t.Helper()
	var m testMetrics
	var err error
	m.attempts, err = testutil.GetCounterMetricValue(resourceClaimUpdateStatusDevicesAttempts)
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", resourceClaimUpdateStatusDevicesAttempts.Name, err)
	}
	m.failures, err = testutil.GetCounterMetricValue(resourceClaimUpdateStatusDevicesFailures)
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", resourceClaimUpdateStatusDevicesFailures.Name, err)
	}
	if m != em {
		t.Fatalf("metrics error: expected %v, received %v", em, m)
	}
}

func TestUpdateStatusMetrics(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAResourceClaimDeviceStatus, true)
	// reset metrics since are stored globally
	resourceClaimUpdateStatusDevicesAttempts.Reset()
	resourceClaimUpdateStatusDevicesFailures.Reset()

	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	ctx := genericapirequest.NewDefaultContext()

	key, _ := storage.KeyFunc(ctx, "foo")
	claimStart := validNewClaim("foo", metav1.NamespaceDefault)
	claimStart.Spec = resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{Name: "request"},
			},
		},
	}
	err := storage.Storage.Create(ctx, key, claimStart, nil, 0, false)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expectMetrics(t, testMetrics{
		attempts: 0,
		failures: 0,
	})

	claimValid := claimStart.DeepCopy()
	claimValid.Status = resource.ResourceClaimStatus{
		Allocation: &resource.AllocationResult{
			Devices: resource.DeviceAllocationResult{
				Results: []resource.DeviceRequestAllocationResult{{
					Request: "request",
					Driver:  "driver-1",
					Pool:    "pool-1",
					Device:  "device-1",
				}},
			},
		},
		Devices: []resource.AllocatedDeviceStatus{
			{
				Driver: "driver-1",
				Pool:   "pool-1",
				Device: "device-1",
				NetworkData: &resource.NetworkDeviceData{
					IPs: []string{
						"2001:db8::1/64",
					},
				},
			},
		},
	}
	// success
	_, _, err = statusStorage.Update(ctx, claimValid.Name, rest.DefaultUpdatedObjectInfo(claimValid), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	expectMetrics(t, testMetrics{
		attempts: 1,
		failures: 0,
	})
	obj, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	claimOut := obj.(*resource.ResourceClaim)
	// only compare relevant changes b/c of difference in metadata
	if !apiequality.Semantic.DeepEqual(claimValid.Status, claimOut.Status) {
		t.Errorf("unexpected object: %s", cmp.Diff(claimValid.Status, claimOut.Status))
	}
	// failures
	// duplicate IPs and missing request name
	claimInvalid := claimStart.DeepCopy()
	claimInvalid.Status = resource.ResourceClaimStatus{
		Allocation: &resource.AllocationResult{
			Devices: resource.DeviceAllocationResult{
				Results: []resource.DeviceRequestAllocationResult{{
					Request: "", // must reference the Spec request
					Driver:  "driver-1",
					Pool:    "pool-1",
					Device:  "device-1",
				}},
			},
		},
		Devices: []resource.AllocatedDeviceStatus{
			{
				Driver: "driver-1",
				Pool:   "pool-1",
				Device: "device-1",
				NetworkData: &resource.NetworkDeviceData{
					IPs: []string{
						"2001:db8::1/64",
						"2001:db8::1/64", // can not have duplicates
					},
				},
			},
		},
	}
	_, _, err = statusStorage.Update(ctx, claimInvalid.Name, rest.DefaultUpdatedObjectInfo(claimInvalid), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err == nil {
		t.Fatalf("Unexpected success")
	}
	expectMetrics(t, testMetrics{
		attempts: 2,
		failures: 1,
	})

}
