/*
Copyright The Kubernetes Authors.

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
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/kubernetes/pkg/apis/resource"
	_ "k8s.io/kubernetes/pkg/apis/resource/install"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/utils/ptr"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, resource.Resource("resourcepoolstatusrequests"))
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "resourcepoolstatusrequests",
	}
	requestStorage, statusStorage, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return requestStorage, statusStorage, server
}

func validNewRequest(name string) *resource.ResourcePoolStatusRequest {
	return &resource.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: resource.ResourcePoolStatusRequestSpec{
			Driver: "gpu.example.com",
			// Defaulted only in the versioned API.
			Limit: ptr.To[int32](100),
		},
	}
}

func completeStatus() *resource.ResourcePoolStatusRequestStatus {
	return &resource.ResourcePoolStatusRequestStatus{
		PoolCount: ptr.To[int32](1),
		Pools: []resource.PoolStatus{{
			Driver:             "gpu.example.com",
			PoolName:           "pool-1",
			Generation:         1,
			ResourceSliceCount: ptr.To[int32](1),
			TotalDevices:       ptr.To[int32](4),
			AllocatedDevices:   ptr.To[int32](1),
			AvailableDevices:   ptr.To[int32](3),
			UnavailableDevices: ptr.To[int32](0),
		}},
		Conditions: []metav1.Condition{{
			Type:               "Complete",
			Status:             metav1.ConditionTrue,
			Reason:             "CalculationComplete",
			Message:            "Calculated status for 1 pools",
			LastTransitionTime: metav1.Time{Time: time.Now().Truncate(time.Second)},
		}},
	}
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	request := validNewRequest("foo")
	request.ObjectMeta = metav1.ObjectMeta{GenerateName: "foo"}
	test.TestCreate(
		// valid
		request,
		// invalid
		&resource.ResourcePoolStatusRequest{
			ObjectMeta: metav1.ObjectMeta{Name: "*BadName!"},
			Spec:       resource.ResourcePoolStatusRequestSpec{Driver: "gpu.example.com", Limit: ptr.To[int32](100)},
		},
		&resource.ResourcePoolStatusRequest{
			ObjectMeta: metav1.ObjectMeta{Name: "no-driver"},
			Spec:       resource.ResourcePoolStatusRequestSpec{Limit: ptr.To[int32](100)},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestUpdate(
		// valid
		validNewRequest("foo"),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*resource.ResourcePoolStatusRequest)
			object.Labels = map[string]string{"foo": "bar"}
			return object
		},
		// invalid updateFunc: the spec is immutable
		func(obj runtime.Object) runtime.Object {
			object := obj.(*resource.ResourcePoolStatusRequest)
			object.Spec.Driver = "other.example.com"
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope().ReturnDeletedObject()
	test.TestDelete(validNewRequest("foo"))
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestGet(validNewRequest("foo"))
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestList(validNewRequest("foo"))
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestWatch(
		validNewRequest("foo"),
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

// Status is ignored outside the status subresource.
func TestCreateAndUpdateIgnoreStatus(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	ctx := genericregistrytest.NewClusterScopeContext(storage.Store)

	request := validNewRequest("foo")
	request.Status = completeStatus()
	obj, err := storage.Create(ctx, request, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	created := obj.(*resource.ResourcePoolStatusRequest)
	if created.Status != nil {
		t.Fatalf("Expected status to be dropped on create, got %+v", created.Status)
	}

	update := created.DeepCopy()
	update.Status = completeStatus()
	obj, _, err = storage.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if updated := obj.(*resource.ResourcePoolStatusRequest); updated.Status != nil {
		t.Errorf("Expected status to be ignored on update of the main resource, got %+v", updated.Status)
	}
}

// The status is write-once.
func TestUpdateStatus(t *testing.T) {
	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	ctx := genericregistrytest.NewClusterScopeContext(storage.Store)
	statusCtx := genericregistrytest.NewClusterScopeContext(storage.Store, "status")

	key, _ := storage.KeyFunc(ctx, "foo")
	var created resource.ResourcePoolStatusRequest
	if err := storage.Storage.Create(ctx, key, validNewRequest("foo"), &created, 0, false); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	request := created.DeepCopy()
	request.Status = completeStatus()
	// Spec changes through the status subresource are ignored.
	request.Spec.Driver = "other.example.com"
	if _, _, err := statusStorage.Update(statusCtx, request.Name, rest.DefaultUpdatedObjectInfo(request), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	stored := obj.(*resource.ResourcePoolStatusRequest)
	if !apiequality.Semantic.DeepEqual(request.Status, stored.Status) {
		t.Errorf("Unexpected status: %s", cmp.Diff(request.Status, stored.Status))
	}
	if stored.Spec.Driver != "gpu.example.com" {
		t.Errorf("Spec.Driver = %q, want it unchanged by the status update", stored.Spec.Driver)
	}

	rewrite := stored.DeepCopy()
	rewrite.Status.Pools[0].AllocatedDevices = ptr.To[int32](2)
	_, _, err = statusStorage.Update(statusCtx, rewrite.Name, rest.DefaultUpdatedObjectInfo(rewrite), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err == nil || !strings.Contains(err.Error(), "field is immutable") {
		t.Errorf("Expected immutable error when rewriting a written status, got %v", err)
	}
}

func TestShortNames(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"rpsr"}
	registrytest.AssertShortNames(t, storage, expected)
}

func TestCategories(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"dra"}
	registrytest.AssertCategories(t, storage, expected)
}
