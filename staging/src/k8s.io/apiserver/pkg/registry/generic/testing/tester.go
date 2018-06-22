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

package tester

import (
	"context"
	"fmt"
	"reflect"
	"sync"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/registry/rest/resttest"
	"k8s.io/apiserver/pkg/storage"
	etcdstorage "k8s.io/apiserver/pkg/storage/etcd"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
)

type Tester struct {
	tester         *resttest.Tester
	genericStorage *genericregistry.Store
}
type UpdateFunc func(runtime.Object) runtime.Object

// New creates a tester for the given rest.Storage.
//
// It assumes that the rest.Storage object embeds a *genericregistry.Store
// in its struct. Call the generic NewWithUnderlyingGenericStorage instead
// if this assumption does not hold.
//
// It mutates the generic storage.
func New(t *testing.T, storage rest.Storage) *Tester {
	if genericStore, ok := storage.(*genericregistry.Store); ok {
		return NewWithUnderlyingGenericStorage(t, storage, genericStore)
	}

	store := reflect.ValueOf(storage).Elem().FieldByName("Store")
	if !store.IsValid() {
		panic(fmt.Sprintf("storage %#v does not include an embedded *genericregistry.Store, please use NewWithUnderlyingGenericStorage instead", storage))
	}
	if store.Type() != reflect.TypeOf(&genericregistry.Store{}) {
		panic(fmt.Sprintf("storage %#v does not include an embedded *genericregistry.Store, but found a %T. Please use NewWithUnderlyingGenericStorage instead", storage, store.Interface()))
	}
	return NewWithUnderlyingGenericStorage(t, storage, store.Interface().(*genericregistry.Store))
}

func NewWithUnderlyingGenericStorage(t *testing.T, storage rest.Storage, genericStorage *genericregistry.Store) *Tester {
	i := &shallowCopyStorageInterceptor{Interface: genericStorage.Storage}
	genericStorage.Storage = i
	return &Tester{
		tester:         resttest.New(t, storage, i.lastReadStorageObject),
		genericStorage: genericStorage,
	}
}

func (t *Tester) TestNamespace() string {
	return t.tester.TestNamespace()
}

func (t *Tester) ClusterScope() *Tester {
	t.tester = t.tester.ClusterScope()
	return t
}

func (t *Tester) Namer(namer func(int) string) *Tester {
	t.tester = t.tester.Namer(namer)
	return t
}

func (t *Tester) AllowCreateOnUpdate() *Tester {
	t.tester = t.tester.AllowCreateOnUpdate()
	return t
}

func (t *Tester) GeneratesName() *Tester {
	t.tester = t.tester.GeneratesName()
	return t
}

func (t *Tester) ReturnDeletedObject() *Tester {
	t.tester = t.tester.ReturnDeletedObject()
	return t
}

func (t *Tester) TestCreate(valid runtime.Object, invalid ...runtime.Object) {
	t.tester.TestCreate(
		valid,
		t.createObject,
		t.getObject,
		invalid...,
	)
}

func (t *Tester) TestUpdate(valid runtime.Object, validUpdateFunc UpdateFunc, invalidUpdateFunc ...UpdateFunc) {
	var invalidFuncs []resttest.UpdateFunc
	for _, f := range invalidUpdateFunc {
		invalidFuncs = append(invalidFuncs, resttest.UpdateFunc(f))
	}
	t.tester.TestUpdate(
		valid,
		t.createObject,
		t.getObject,
		resttest.UpdateFunc(validUpdateFunc),
		invalidFuncs...,
	)
}

func (t *Tester) TestDelete(valid runtime.Object) {
	t.tester.TestDelete(
		valid,
		t.createObject,
		t.getObject,
		errors.IsNotFound,
	)
}

func (t *Tester) TestDeleteGraceful(valid runtime.Object, expectedGrace int64) {
	t.tester.TestDeleteGraceful(
		valid,
		t.createObject,
		t.getObject,
		expectedGrace,
	)
}

func (t *Tester) TestGet(valid runtime.Object) {
	t.tester.TestGet(valid)
}

func (t *Tester) TestList(valid runtime.Object) {
	t.tester.TestList(
		valid,
		t.setObjectsForList,
	)
}

func (t *Tester) TestWatch(valid runtime.Object, labelsPass, labelsFail []labels.Set, fieldsPass, fieldsFail []fields.Set) {
	t.tester.TestWatch(
		valid,
		t.emitObject,
		labelsPass,
		labelsFail,
		fieldsPass,
		fieldsFail,
		// TODO: This should be filtered, the registry should not be aware of this level of detail
		[]string{etcdstorage.EtcdCreate, etcdstorage.EtcdDelete},
	)
}

// Helper functions

func (t *Tester) getObject(ctx context.Context, obj runtime.Object) (runtime.Object, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return nil, err
	}

	result, err := t.genericStorage.Get(ctx, accessor.GetName(), &metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return result, nil
}

func (t *Tester) createObject(ctx context.Context, obj runtime.Object) error {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return err
	}
	key, err := t.genericStorage.KeyFunc(ctx, accessor.GetName())
	if err != nil {
		return err
	}
	return t.genericStorage.Storage.Create(ctx, key, obj, nil, 0)
}

func (t *Tester) setObjectsForList(objects []runtime.Object) []runtime.Object {
	key := t.genericStorage.KeyRootFunc(t.tester.TestContext())
	if _, err := t.genericStorage.DeleteCollection(t.tester.TestContext(), nil, nil); err != nil {
		t.tester.Errorf("unable to clear collection: %v", err)
		return nil
	}
	if err := storagetesting.CreateObjList(key, t.genericStorage.Storage, objects); err != nil {
		t.tester.Errorf("unexpected error: %v", err)
		return nil
	}
	return objects
}

func (t *Tester) emitObject(obj runtime.Object, action string) error {
	ctx := t.tester.TestContext()
	var err error

	switch action {
	case etcdstorage.EtcdCreate:
		err = t.createObject(ctx, obj)
	case etcdstorage.EtcdDelete:
		var accessor metav1.Object
		accessor, err = meta.Accessor(obj)
		if err != nil {
			return err
		}
		_, _, err = t.genericStorage.Delete(ctx, accessor.GetName(), nil)
	default:
		err = fmt.Errorf("unexpected action: %v", action)
	}

	return err
}

// shallowCopyStorageInterceptor intercepts Get and List by:
// - passing fresh runtime.Objects to the underlying etcd storage
// - shallow copying those filled runtime.Objects to the objects provided by the caller
//   (in the case of the List this is done for each list item).
// The interceptee is the last object passed down to the etcd storage.
type shallowCopyStorageInterceptor struct {
	storage.Interface

	lock        sync.RWMutex
	interceptee runtime.Object
}

// lastReadStorageObject returns the last shared object returned from the underlying storage.
// This is shallow copied to the consume of this storage.
func (s *shallowCopyStorageInterceptor) lastReadStorageObject() runtime.Object {
	s.lock.RLock()
	defer s.lock.RUnlock()
	return s.interceptee
}

// Get unmarshals json found at key into objPtr. On a not found error, will either
// return a zero object of the requested type, or an error, depending on ignoreNotFound.
// Treats empty responses and nil response nodes exactly like a not found error.
// The returned contents may be delayed, but it is guaranteed that they will
// be have at least 'resourceVersion'.
func (s *shallowCopyStorageInterceptor) Get(ctx context.Context, key string, resourceVersion string, objPtr runtime.Object, ignoreNotFound bool) error {
	interceptee := reflect.New(reflect.TypeOf(objPtr).Elem()).Interface().(runtime.Object)
	err := s.Interface.Get(ctx, key, resourceVersion, interceptee, ignoreNotFound)
	if err != nil {
		return err
	}
	objVal, err := conversion.EnforcePtr(objPtr)
	if err != nil {
		return err
	}
	objVal.Set(reflect.ValueOf(interceptee).Elem())

	s.lock.Lock()
	defer s.lock.Unlock()
	s.interceptee = interceptee

	return nil
}

// List unmarshalls jsons found at directory defined by key and opaque them
// into *List api object (an object that satisfies runtime.IsList definition).
// The returned contents may be delayed, but it is guaranteed that they will
// be have at least 'resourceVersion'.
func (s *shallowCopyStorageInterceptor) List(ctx context.Context, key string, resourceVersion string, p storage.SelectionPredicate, listObj runtime.Object) error {
	interceptee := reflect.New(reflect.TypeOf(listObj).Elem()).Interface().(runtime.Object)
	err := s.Interface.List(ctx, key, resourceVersion, p, interceptee)
	if err != nil {
		return err
	}

	listPtr, err := meta.GetItemsPtr(listObj)
	if err != nil {
		return err
	}
	listVal, err := conversion.EnforcePtr(listPtr)
	if err != nil || listVal.Kind() != reflect.Slice {
		return fmt.Errorf("need a pointer to slice, got %v", listVal.Kind())
	}

	intercepteeListPtr, err := meta.GetItemsPtr(interceptee)
	if err != nil {
		return err
	}
	intercepteeListVal, err := conversion.EnforcePtr(intercepteeListPtr)
	if err != nil || listVal.Kind() != reflect.Slice {
		return fmt.Errorf("need a pointer to slice, got %v", listVal.Kind())
	}

	listVal.Set(reflect.MakeSlice(listVal.Type(), intercepteeListVal.Len(), intercepteeListVal.Len()))
	for i := 0; i < intercepteeListVal.Len(); i++ {
		listVal.Index(i).Set(intercepteeListVal.Index(i))
	}

	s.lock.Lock()
	defer s.lock.Unlock()
	s.interceptee = interceptee

	return err
}
