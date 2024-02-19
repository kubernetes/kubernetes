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
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/registry/rest/resttest"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
)

type Tester struct {
	tester  *resttest.Tester
	storage *genericregistry.Store
}
type UpdateFunc func(runtime.Object) runtime.Object

func New(t *testing.T, storage *genericregistry.Store) *Tester {
	return &Tester{
		tester:  resttest.New(t, storage),
		storage: storage,
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
		[]string{"create", "delete"},
	)
}

// Helper functions

func (t *Tester) getObject(ctx context.Context, obj runtime.Object) (runtime.Object, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return nil, err
	}

	result, err := t.storage.Get(ctx, accessor.GetName(), &metav1.GetOptions{})
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
	key, err := t.storage.KeyFunc(ctx, accessor.GetName())
	if err != nil {
		return err
	}
	return t.storage.Storage.Create(ctx, key, obj, nil, 0, false)
}

func (t *Tester) setObjectsForList(objects []runtime.Object) []runtime.Object {
	key := t.storage.KeyRootFunc(t.tester.TestContext())
	if _, err := t.storage.DeleteCollection(t.tester.TestContext(), rest.ValidateAllObjectFunc, nil, nil); err != nil {
		t.tester.Errorf("unable to clear collection: %v", err)
		return nil
	}
	if err := storagetesting.CreateObjList(key, t.storage.Storage.Storage, objects); err != nil {
		t.tester.Errorf("unexpected error: %v", err)
		return nil
	}
	return objects
}

func (t *Tester) emitObject(obj runtime.Object, action string) error {
	ctx := t.tester.TestContext()
	var err error

	switch action {
	case "create":
		err = t.createObject(ctx, obj)
	case "delete":
		var accessor metav1.Object
		accessor, err = meta.Accessor(obj)
		if err != nil {
			return err
		}
		_, _, err = t.storage.Delete(ctx, accessor.GetName(), rest.ValidateAllObjectFunc, nil)
	default:
		err = fmt.Errorf("unexpected action: %v", action)
	}

	return err
}
