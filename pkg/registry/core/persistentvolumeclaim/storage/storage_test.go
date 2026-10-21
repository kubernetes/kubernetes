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
	"context"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "persistentvolumeclaims",
	}
	persistentVolumeClaimStorage, statusStorage, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return persistentVolumeClaimStorage, statusStorage, server
}

func validNewPersistentVolumeClaim(name, ns string) *api.PersistentVolumeClaim {
	pv := &api.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Resources: api.VolumeResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
			},
		},
		Status: api.PersistentVolumeClaimStatus{
			Phase: api.ClaimPending,
		},
	}
	return pv
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	pv := validNewPersistentVolumeClaim("foo", metav1.NamespaceDefault)
	pv.ObjectMeta = metav1.ObjectMeta{}
	test.TestCreate(
		// valid
		pv,
		// invalid
		&api.PersistentVolumeClaim{
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
		validNewPersistentVolumeClaim("foo", metav1.NamespaceDefault),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*api.PersistentVolumeClaim)
			object.Spec.VolumeName = "onlyVolumeNameUpdateAllowed"
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ReturnDeletedObject()
	test.TestDelete(validNewPersistentVolumeClaim("foo", metav1.NamespaceDefault))
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestGet(validNewPersistentVolumeClaim("foo", metav1.NamespaceDefault))
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestList(validNewPersistentVolumeClaim("foo", metav1.NamespaceDefault))
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestWatch(
		validNewPersistentVolumeClaim("foo", metav1.NamespaceDefault),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": "foo"},
			{"name": "foo"},
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
	ctx := genericregistrytest.NewNamespaceScopeContext(storage.Store, metav1.NamespaceDefault)
	statusCtx := genericregistrytest.NewNamespaceScopeContext(storage.Store, metav1.NamespaceDefault, "status")

	key, _ := storage.KeyFunc(ctx, "foo")
	pvcStart := validNewPersistentVolumeClaim("foo", metav1.NamespaceDefault)
	err := storage.Storage.Create(ctx, key, pvcStart, nil, 0, false)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	pvc := &api.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Resources: api.VolumeResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("3Gi"),
				},
			},
		},
		Status: api.PersistentVolumeClaimStatus{
			Phase: api.ClaimBound,
		},
	}

	_, _, err = statusStorage.Update(statusCtx, pvc.Name, rest.DefaultUpdatedObjectInfo(pvc), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	pvcOut := obj.(*api.PersistentVolumeClaim)
	// only compare relevant changes b/c of difference in metadata
	if !apiequality.Semantic.DeepEqual(pvc.Status, pvcOut.Status) {
		t.Errorf("unexpected object: %s", cmp.Diff(pvc.Status, pvcOut.Status))
	}
}

func TestShortNames(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"pvc"}
	registrytest.AssertShortNames(t, storage, expected)
}

func TestDefaultOnReadPvc(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	dataSource := api.TypedLocalObjectReference{
		Kind: "PersistentVolumeClaim",
		Name: "my-pvc",
	}
	dataSourceRef := api.TypedObjectReference{
		Kind: "PersistentVolumeClaim",
		Name: "my-pvc",
	}

	var tests = map[string]struct {
		dataSource    bool
		dataSourceRef bool
		want          bool
		wantRef       bool
	}{
		"empty ds": {},
		"volume ds": {
			dataSource: true,
			want:       true,
			wantRef:    true,
		},
		"volume ds ref": {
			dataSourceRef: true,
			want:          true,
			wantRef:       true,
		},
		"both data sources": {
			dataSource:    true,
			dataSourceRef: true,
			want:          true,
			wantRef:       true,
		},
	}

	for testName, test := range tests {
		t.Run(testName, func(t *testing.T) {
			pvc := new(api.PersistentVolumeClaim)
			if test.dataSource {
				pvc.Spec.DataSource = dataSource.DeepCopy()
			}
			if test.dataSourceRef {
				pvc.Spec.DataSourceRef = dataSourceRef.DeepCopy()
			}
			var expectDataSource *api.TypedLocalObjectReference
			if test.want {
				expectDataSource = &dataSource
			}
			var expectDataSourceRef *api.TypedObjectReference
			if test.wantRef {
				expectDataSourceRef = &dataSourceRef
			}

			// Method under test
			storage.defaultOnReadPvc(pvc)

			if !reflect.DeepEqual(pvc.Spec.DataSource, expectDataSource) {
				t.Errorf("data source does not match, test: %s, dataSource: %v, expected: %v",
					testName, test.dataSource, test.want)
			}
			if !reflect.DeepEqual(pvc.Spec.DataSourceRef, expectDataSourceRef) {
				t.Errorf("data source ref does not match, test: %s, dataSourceRef: %v, expected: %v",
					testName, test.dataSourceRef, test.wantRef)
			}
		})
	}
}

func TestDefaultOnReadWatchWithCacher(t *testing.T) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	defer server.Terminate(t)
	etcdStorage.EventsHistoryWindow = storagebackend.DefaultEventsHistoryWindow

	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage.ForResource(schema.GroupResource{Resource: "persistentvolumeclaims"}),
		Decorator:               genericregistry.StorageWithCacher(),
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "persistentvolumeclaims",
	}
	storage, statusStorage, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	defer storage.Store.DestroyFunc()

	ctx := genericregistrytest.NewNamespaceScopeContext(storage.Store, metav1.NamespaceDefault)
	statusCtx := genericregistrytest.NewNamespaceScopeContext(storage.Store, metav1.NamespaceDefault, "status")

	// Wait for Cacher to be ready and obtain the current ResourceVersion.
	var listRV string
	if err := wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		listObj, err := storage.List(ctx, &metainternalversion.ListOptions{ResourceVersion: "0"})
		if err != nil {
			return false, nil
		}
		listRV = listObj.(*api.PersistentVolumeClaimList).ResourceVersion
		return true, nil
	}); err != nil {
		t.Fatalf("failed waiting for cacher to be ready: %v", err)
	}

	w, err := storage.Watch(ctx, &metainternalversion.ListOptions{ResourceVersion: listRV})
	if err != nil {
		t.Fatalf("unexpected error starting watch: %v", err)
	}
	defer w.Stop()

	// Simulate a pre-existing PVC in etcd that has Spec.DataSource set without Spec.DataSourceRef.
	pvc := validNewPersistentVolumeClaim("foo", metav1.NamespaceDefault)
	pvc.Spec.DataSource = &api.TypedLocalObjectReference{
		Kind: "PersistentVolumeClaim",
		Name: "my-source-pvc",
	}
	key, err := storage.KeyFunc(ctx, pvc.Name)
	if err != nil {
		t.Fatalf("unexpected key error: %v", err)
	}
	out := &api.PersistentVolumeClaim{}
	if err := storage.Storage.Create(ctx, key, pvc, out, 0, false); err != nil {
		t.Fatalf("unexpected error creating PVC in storage: %v", err)
	}

	expectNormalizedWatchEvent := func(expectedType watch.EventType) *api.PersistentVolumeClaim {
		t.Helper()
		select {
		case event, ok := <-w.ResultChan():
			if !ok {
				t.Fatalf("watch channel closed unexpectedly waiting for %v", expectedType)
			}
			if event.Type != expectedType {
				t.Fatalf("expected watch event type %v, got %v (%#v)", expectedType, event.Type, event.Object)
			}
			obj := event.Object
			if co, ok := obj.(runtime.CacheableObject); ok {
				obj = co.GetObject()
			}
			gotPVC, ok := obj.(*api.PersistentVolumeClaim)
			if !ok {
				t.Fatalf("expected *api.PersistentVolumeClaim, got %T", obj)
			}
			if gotPVC.Spec.DataSourceRef == nil {
				t.Errorf("%v watch event: expected Spec.DataSourceRef to be populated by defaultOnRead, got nil", expectedType)
			}
			return gotPVC
		case <-time.After(wait.ForeverTestTimeout):
			t.Fatalf("timed out waiting for %v watch event", expectedType)
			return nil
		}
	}

	// 1. Live ADDED event from watch cache.
	createdPVC := expectNormalizedWatchEvent(watch.Added)

	// 2. Live MODIFIED event triggered via status update (normal API operation on a pre-existing PVC).
	updatedInput := createdPVC.DeepCopy()
	updatedInput.Status.Phase = api.ClaimBound
	if _, _, err := statusStorage.Update(statusCtx, updatedInput.Name, rest.DefaultUpdatedObjectInfo(updatedInput), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("unexpected error updating status: %v", err)
	}
	expectNormalizedWatchEvent(watch.Modified)

	// 3. Live DELETED event triggered via normal Delete.
	if _, _, err := storage.Delete(ctx, pvc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{}); err != nil {
		t.Fatalf("unexpected error deleting PVC: %v", err)
	}
	expectNormalizedWatchEvent(watch.Deleted)
}
