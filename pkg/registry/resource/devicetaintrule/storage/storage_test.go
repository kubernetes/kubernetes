/*
Copyright 2025 The Kubernetes Authors.

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
	"time"

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
	"k8s.io/kubernetes/pkg/apis/resource"
	_ "k8s.io/kubernetes/pkg/apis/resource/install"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, resource.Resource("devicetaintrules"))
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "devicetaintrules",
	}
	deviceTaintStorage, statusStorage, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return deviceTaintStorage, statusStorage, server
}

func validNewDeviceTaint(name string) *resource.DeviceTaintRule {
	return &resource.DeviceTaintRule{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: resource.DeviceTaintRuleSpec{
			Taint: resource.DeviceTaint{
				Key:       "example.com/taint",
				Effect:    resource.DeviceTaintEffectNoExecute,
				TimeAdded: &metav1.Time{Time: time.Now().Truncate(time.Second)}, // Must know in advance what will be stored.
			},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	patch := validNewDeviceTaint("foo")
	patch.ObjectMeta = metav1.ObjectMeta{GenerateName: "foo"}
	test.TestCreate(
		// valid
		patch,
		// invalid
		&resource.DeviceTaintRule{
			ObjectMeta: metav1.ObjectMeta{Name: "*BadName!"},
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
		validNewDeviceTaint("foo"),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*resource.DeviceTaintRule)
			object.Labels = map[string]string{"foo": "bar"}
			return object
		},
	)

}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope().ReturnDeletedObject()
	test.TestDelete(validNewDeviceTaint("foo"))
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestGet(validNewDeviceTaint("foo"))
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestList(validNewDeviceTaint("foo"))
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestWatch(
		validNewDeviceTaint("foo"),
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
	deviceTaintStart := validNewDeviceTaint("foo")
	err := storage.Storage.Create(ctx, key, deviceTaintStart, nil, 0, false)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	deviceTaint := deviceTaintStart.DeepCopy()
	deviceTaint.Status.Conditions = []metav1.Condition{{
		Type:               "EvicitionInProgress",
		Status:             metav1.ConditionTrue,
		Reason:             "PodsLeft",
		Message:            "100 pods left",
		LastTransitionTime: metav1.Time{Time: time.Now().Truncate(time.Second)},
	}}
	_, _, err = statusStorage.Update(ctx, deviceTaint.Name, rest.DefaultUpdatedObjectInfo(deviceTaint), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	deviceTaintOut := obj.(*resource.DeviceTaintRule)
	// only compare relevant changes b/c of difference in metadata
	if !apiequality.Semantic.DeepEqual(deviceTaint.Status, deviceTaintOut.Status) {
		t.Errorf("unexpected object: %s", cmp.Diff(deviceTaint.Status, deviceTaintOut.Status))
	}
}
