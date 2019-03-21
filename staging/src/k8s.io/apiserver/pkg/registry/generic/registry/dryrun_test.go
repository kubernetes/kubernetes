/*
Copyright 2018 The Kubernetes Authors.

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

package registry

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/apitesting"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/storage"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/apiserver/pkg/storage/storagebackend/factory"
)

func NewDryRunnableTestStorage(t *testing.T) (DryRunnableStorage, func()) {
	server, sc := etcdtesting.NewUnsecuredEtcd3TestClientServer(t)
	sc.Codec = apitesting.TestStorageCodec(codecs, examplev1.SchemeGroupVersion)
	s, destroy, err := factory.Create(*sc)
	if err != nil {
		t.Fatalf("Error creating storage: %v", err)
	}
	return DryRunnableStorage{Storage: s, Codec: sc.Codec}, func() {
		destroy()
		server.Terminate(t)
	}
}

func UnstructuredOrDie(j string) *unstructured.Unstructured {
	m := map[string]interface{}{}
	err := json.Unmarshal([]byte(j), &m)
	if err != nil {
		panic(fmt.Errorf("Failed to unmarshal into Unstructured: %v", err))
	}
	return &unstructured.Unstructured{Object: m}
}

func TestDryRunCreateDoesntCreate(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod"}`)
	out := UnstructuredOrDie(`{}`)

	err := s.Create(context.Background(), "key", obj, out, 0, true)
	if err != nil {
		t.Fatalf("Failed to create new dry-run object: %v", err)
	}

	err = s.Get(context.Background(), "key", "", out, false)
	if e, ok := err.(*storage.StorageError); !ok || e.Code != storage.ErrCodeKeyNotFound {
		t.Errorf("Expected key to be not found, error: %v", err)
	}
}

func TestDryRunCreateReturnsObject(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod"}`)
	out := UnstructuredOrDie(`{}`)

	err := s.Create(context.Background(), "key", obj, out, 0, true)
	if err != nil {
		t.Fatalf("Failed to create new dry-run object: %v", err)
	}

	if !reflect.DeepEqual(obj, out) {
		t.Errorf("Returned object different from input object:\nExpected: %v\nGot: %v", obj, out)
	}
}

func TestDryRunCreateExistingObjectFails(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod"}`)
	out := UnstructuredOrDie(`{}`)

	err := s.Create(context.Background(), "key", obj, out, 0, false)
	if err != nil {
		t.Fatalf("Failed to create new object: %v", err)
	}

	err = s.Create(context.Background(), "key", obj, out, 0, true)
	if e, ok := err.(*storage.StorageError); !ok || e.Code != storage.ErrCodeKeyExists {
		t.Errorf("Expected KeyExists error: %v", err)
	}

}

func TestDryRunUpdateMissingObjectFails(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod"}`)

	updateFunc := func(input runtime.Object, res storage.ResponseMeta) (output runtime.Object, ttl *uint64, err error) {
		return input, nil, errors.New("UpdateFunction shouldn't be called")
	}

	err := s.GuaranteedUpdate(context.Background(), "key", obj, false, nil, updateFunc, true)
	if e, ok := err.(*storage.StorageError); !ok || e.Code != storage.ErrCodeKeyNotFound {
		t.Errorf("Expected key to be not found, error: %v", err)
	}
}

func TestDryRunUpdatePreconditions(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod", "metadata": {"uid": "my-uid"}}`)
	out := UnstructuredOrDie(`{}`)
	err := s.Create(context.Background(), "key", obj, out, 0, false)
	if err != nil {
		t.Fatalf("Failed to create new object: %v", err)
	}

	updateFunc := func(input runtime.Object, res storage.ResponseMeta) (output runtime.Object, ttl *uint64, err error) {
		u, ok := input.(*unstructured.Unstructured)
		if !ok {
			return input, nil, errors.New("Input object is not unstructured")
		}
		unstructured.SetNestedField(u.Object, "value", "field")
		return u, nil, nil
	}
	wrongID := types.UID("wrong-uid")
	myID := types.UID("my-uid")
	err = s.GuaranteedUpdate(context.Background(), "key", obj, false, &storage.Preconditions{UID: &wrongID}, updateFunc, true)
	if e, ok := err.(*storage.StorageError); !ok || e.Code != storage.ErrCodeInvalidObj {
		t.Errorf("Expected invalid object, error: %v", err)
	}

	err = s.GuaranteedUpdate(context.Background(), "key", obj, false, &storage.Preconditions{UID: &myID}, updateFunc, true)
	if err != nil {
		t.Fatalf("Failed to update with valid precondition: %v", err)
	}
}

func TestDryRunUpdateDoesntUpdate(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod"}`)
	created := UnstructuredOrDie(`{}`)

	err := s.Create(context.Background(), "key", obj, created, 0, false)
	if err != nil {
		t.Fatalf("Failed to create new object: %v", err)
	}

	updateFunc := func(input runtime.Object, res storage.ResponseMeta) (output runtime.Object, ttl *uint64, err error) {
		u, ok := input.(*unstructured.Unstructured)
		if !ok {
			return input, nil, errors.New("Input object is not unstructured")
		}
		unstructured.SetNestedField(u.Object, "value", "field")
		return u, nil, nil
	}

	err = s.GuaranteedUpdate(context.Background(), "key", obj, false, nil, updateFunc, true)
	if err != nil {
		t.Fatalf("Failed to dry-run update: %v", err)
	}
	out := UnstructuredOrDie(`{}`)
	err = s.Get(context.Background(), "key", "", out, false)
	if !reflect.DeepEqual(created, out) {
		t.Fatalf("Returned object %q different from expected %q", created, out)
	}
}

func TestDryRunUpdateReturnsObject(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod"}`)
	out := UnstructuredOrDie(`{}`)

	err := s.Create(context.Background(), "key", obj, out, 0, false)
	if err != nil {
		t.Fatalf("Failed to create new object: %v", err)
	}

	updateFunc := func(input runtime.Object, res storage.ResponseMeta) (output runtime.Object, ttl *uint64, err error) {
		u, ok := input.(*unstructured.Unstructured)
		if !ok {
			return input, nil, errors.New("Input object is not unstructured")
		}
		unstructured.SetNestedField(u.Object, "value", "field")
		return u, nil, nil
	}

	err = s.GuaranteedUpdate(context.Background(), "key", obj, false, nil, updateFunc, true)
	if err != nil {
		t.Fatalf("Failed to dry-run update: %v", err)
	}
	out = UnstructuredOrDie(`{"field": "value", "kind": "Pod", "metadata": {"resourceVersion": "2"}}`)
	if !reflect.DeepEqual(obj, out) {
		t.Fatalf("Returned object %#v different from expected %#v", obj, out)
	}
}

func TestDryRunDeleteDoesntDelete(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod"}`)
	out := UnstructuredOrDie(`{}`)

	err := s.Create(context.Background(), "key", obj, out, 0, false)
	if err != nil {
		t.Fatalf("Failed to create new object: %v", err)
	}

	err = s.Delete(context.Background(), "key", out, nil, true)
	if err != nil {
		t.Fatalf("Failed to dry-run delete the object: %v", err)
	}

	err = s.Get(context.Background(), "key", "", out, false)
	if err != nil {
		t.Fatalf("Failed to retrieve dry-run deleted object: %v", err)
	}
}

func TestDryRunDeleteMissingObjectFails(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	out := UnstructuredOrDie(`{}`)
	err := s.Delete(context.Background(), "key", out, nil, true)
	if e, ok := err.(*storage.StorageError); !ok || e.Code != storage.ErrCodeKeyNotFound {
		t.Errorf("Expected key to be not found, error: %v", err)
	}
}

func TestDryRunDeleteReturnsObject(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod"}`)
	out := UnstructuredOrDie(`{}`)

	err := s.Create(context.Background(), "key", obj, out, 0, false)
	if err != nil {
		t.Fatalf("Failed to create new object: %v", err)
	}

	out = UnstructuredOrDie(`{}`)
	expected := UnstructuredOrDie(`{"kind": "Pod", "metadata": {"resourceVersion": "2"}}`)
	err = s.Delete(context.Background(), "key", out, nil, true)
	if err != nil {
		t.Fatalf("Failed to delete with valid precondition: %v", err)
	}
	if !reflect.DeepEqual(expected, out) {
		t.Fatalf("Returned object %q doesn't match expected: %q", out, expected)
	}
}

func TestDryRunDeletePreconditions(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod", "metadata": {"uid": "my-uid"}}`)
	out := UnstructuredOrDie(`{}`)

	err := s.Create(context.Background(), "key", obj, out, 0, false)
	if err != nil {
		t.Fatalf("Failed to create new object: %v", err)
	}

	wrongID := types.UID("wrong-uid")
	myID := types.UID("my-uid")
	err = s.Delete(context.Background(), "key", out, &storage.Preconditions{UID: &wrongID}, true)
	if e, ok := err.(*storage.StorageError); !ok || e.Code != storage.ErrCodeInvalidObj {
		t.Errorf("Expected invalid object, error: %v", err)
	}

	err = s.Delete(context.Background(), "key", out, &storage.Preconditions{UID: &myID}, true)
	if err != nil {
		t.Fatalf("Failed to delete with valid precondition: %v", err)
	}
}
