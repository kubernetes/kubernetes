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
	"k8s.io/kubernetes/pkg/apis/resource"
	_ "k8s.io/kubernetes/pkg/apis/resource/install"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, resource.GroupName)
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "podschedulingcontexts",
	}
	podSchedulingStorage, statusStorage, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return podSchedulingStorage, statusStorage, server
}

func validNewPodSchedulingContexts(name, ns string) *resource.PodSchedulingContext {
	schedulingCtx := &resource.PodSchedulingContext{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: resource.PodSchedulingContextSpec{
			SelectedNode: "worker",
		},
		Status: resource.PodSchedulingContextStatus{},
	}
	return schedulingCtx
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	schedulingCtx := validNewPodSchedulingContexts("foo", metav1.NamespaceDefault)
	schedulingCtx.ObjectMeta = metav1.ObjectMeta{}
	test.TestCreate(
		// valid
		schedulingCtx,
		// invalid
		&resource.PodSchedulingContext{
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
		validNewPodSchedulingContexts("foo", metav1.NamespaceDefault),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*resource.PodSchedulingContext)
			if object.Labels == nil {
				object.Labels = map[string]string{}
			}
			object.Labels["foo"] = "bar"
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ReturnDeletedObject()
	test.TestDelete(validNewPodSchedulingContexts("foo", metav1.NamespaceDefault))
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestGet(validNewPodSchedulingContexts("foo", metav1.NamespaceDefault))
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestList(validNewPodSchedulingContexts("foo", metav1.NamespaceDefault))
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestWatch(
		validNewPodSchedulingContexts("foo", metav1.NamespaceDefault),
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
	schedulingStart := validNewPodSchedulingContexts("foo", metav1.NamespaceDefault)
	err := storage.Storage.Create(ctx, key, schedulingStart, nil, 0, false)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	schedulingCtx := schedulingStart.DeepCopy()
	schedulingCtx.Status.ResourceClaims = append(schedulingCtx.Status.ResourceClaims,
		resource.ResourceClaimSchedulingStatus{
			Name: "my-claim",
		},
	)
	_, _, err = statusStorage.Update(ctx, schedulingCtx.Name, rest.DefaultUpdatedObjectInfo(schedulingCtx), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	schedulingOut := obj.(*resource.PodSchedulingContext)
	// only compare relevant changes b/c of difference in metadata
	if !apiequality.Semantic.DeepEqual(schedulingCtx.Status, schedulingOut.Status) {
		t.Errorf("unexpected object: %s", cmp.Diff(schedulingCtx.Status, schedulingOut.Status))
	}
}
