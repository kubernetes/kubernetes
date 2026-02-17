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
	"context"
	"testing"
	"time"

	"k8s.io/utils/ptr"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/kubernetes/pkg/apis/coordination"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	testing2 "k8s.io/utils/clock/testing"
)

const validUID = "8057f54d-455d-4b25-90c6-92a919cff10a"

func newStorage(t *testing.T) (*REST, *StatusREST, *etcd3testing.EtcdTestServer) {
	clock := testing2.NewFakePassiveClock(time.Now())
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, coordination.SchemeGroupVersion.WithResource("evictions").GroupResource())
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "evictions",
	}

	evictionStorage, evictionStatusStorage, err := NewREST(restOptions, clock)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return evictionStorage, evictionStatusStorage, server
}

func tester(t *testing.T, storage *REST) *genericregistrytest.Tester {
	test := genericregistrytest.New(t, storage.Store)
	requestInfo := &genericapirequest.RequestInfo{
		APIGroup:   "coordination.k8s.io",
		APIVersion: "v1alpha1",
		Resource:   "evictions",
	}
	test.SetRequestInfo(requestInfo)
	return test
}

func newValidEviction() *coordination.Eviction {
	return &coordination.Eviction{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: coordination.EvictionSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.EvictionPodReference{
					UID:  validUID,
					Name: "foo.pod",
				},
			},
		},
		Status: coordination.EvictionStatus{
			ObservedGeneration: ptr.To[int64](1),
			Requesters: []coordination.Requester{
				{Name: "requester-1.example.com/bar", Intent: coordination.RequesterIntentEviction},
				{Name: "requester-2.example.com/bar", Intent: coordination.RequesterIntentEviction},
			},
			TargetResponders: []coordination.TargetResponder{
				{Name: "responder1.example.com/bar", State: coordination.ResponderStateInactive},
			},
			Responders: []coordination.ResponderStatus{
				{Name: "responder1.example.com/bar"},
			},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage)
	validEviction := newValidEviction()
	validEviction.ObjectMeta = metav1.ObjectMeta{}

	invalidEviction := newValidEviction()
	invalidEviction.ObjectMeta = metav1.ObjectMeta{Name: "-foo"}
	test.TestCreate(
		validEviction,
		invalidEviction,
	)
}

func TestUpdate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage)
	validEviction := newValidEviction()
	test.TestUpdate(
		validEviction,
		func(obj runtime.Object) runtime.Object {
			object := obj.(*coordination.Eviction)
			object.ObjectMeta.Annotations = map[string]string{"foo": "bar"}
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*coordination.Eviction)
			object.Spec.Target.Pod.Name = "bar"
			return object
		},
	)
}
func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage)
	test.TestDelete(newValidEviction())
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage)
	test.TestGet(newValidEviction())
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage)
	test.TestList(newValidEviction())
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage)
	test.TestWatch(
		newValidEviction(),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"x": "y"},
		},
		// matching fields
		[]fields.Set{},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "xyz"},
			{"name": "foo"},
		},
	)
}

func TestStatusUpdate(t *testing.T) {
	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	eviction := newValidEviction()
	eviction.Status = coordination.EvictionStatus{}

	ctx := evictionContext()
	key, err := storage.KeyFunc(ctx, "foo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	result := &coordination.Eviction{}
	if err := storage.Storage.Create(ctx, key, eviction, result, 0, false); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Status.TargetResponders) != 0 {
		t.Errorf("we expected .status.targetResponders to be empty but it was %v", result.Status.TargetResponders)
	}
	evictionUpdate := newValidEviction()
	evictionUpdate.ObjectMeta = result.ObjectMeta
	evictionUpdate.Labels = map[string]string{"foo": "bar"}
	evictionUpdate.Spec.Target.Pod.Name = "bax"
	evictionUpdate.Status.TargetResponders = []coordination.TargetResponder{
		{Name: "responder1.example.com/bar", State: coordination.ResponderStateActive},
	}
	evictionUpdate.Status.Responders = []coordination.ResponderStatus{
		{Name: "responder1.example.com/bar", StartTime: ptr.To(metav1.Now())},
	}

	if _, _, err := statusStorage.Update(ctx, evictionUpdate.Name, rest.DefaultUpdatedObjectInfo(evictionUpdate), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, evictionUpdate.Name, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	result = obj.(*coordination.Eviction)
	if len(result.Labels) != 0 {
		t.Errorf("we expected .status.labels to be empty but it was %v", result.Labels)
	}
	if result.Spec.Target.Pod.Name != "foo.pod" {
		t.Errorf("we expected .spec.target.pod.name to not be updated but it was updated to %v", result.Spec.Target.Pod.Name)
	}
	if len(result.Status.TargetResponders) != 1 {
		t.Errorf("we expected .status.targetResponders to be updated to but it was %v", result.Status.TargetResponders)
	}
}

func TestGenerationNumber(t *testing.T) {
	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	eviction := newValidEviction()
	eviction.Generation = 100
	eviction.Status.ObservedGeneration = ptr.To[int64](10)
	ctx := evictionContext()
	resultObj, err := storage.Create(ctx, eviction, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	result, _ := resultObj.(*coordination.Eviction)

	// Generation initialization
	if result.Generation != 1 || result.Status.ObservedGeneration != nil {
		t.Fatalf("Unexpected generation number %v, status generation %v", result.Generation, result.Status.ObservedGeneration)
	}

	// Updates to status should not increment either spec or status generation numbers
	result.Status.Conditions = append(result.Status.Conditions, metav1.Condition{Type: "Test", Status: "True", LastTransitionTime: metav1.Now(), Reason: "Reason"})
	if _, _, err := statusStorage.Update(ctx, result.Name, rest.DefaultUpdatedObjectInfo(result), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	resultObj, err = storage.Get(ctx, result.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	result, _ = resultObj.(*coordination.Eviction)
	if result.Generation != 1 || result.Status.ObservedGeneration != nil {
		t.Fatalf("Unexpected generation number, spec: %v, status: %v", result.Generation, result.Status.ObservedGeneration)
	}
}

func evictionContext() context.Context {
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	ctx = genericapirequest.WithRequestInfo(ctx, &genericapirequest.RequestInfo{
		APIGroup:   "coordination.k8s.io",
		APIVersion: "v1alpha1",
		Resource:   "evictions",
	})
	return ctx
}
