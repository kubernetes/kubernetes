/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package resttest

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/util"
)

type Tester struct {
	*testing.T
	storage             rest.Storage
	storageError        injectErrorFunc
	clusterScope        bool
	createOnUpdate      bool
	generatesName       bool
	returnDeletedObject bool
}

type injectErrorFunc func(err error)

func New(t *testing.T, storage rest.Storage, storageError injectErrorFunc) *Tester {
	return &Tester{
		T:            t,
		storage:      storage,
		storageError: storageError,
	}
}

func (t *Tester) withStorageError(err error, fn func()) {
	t.storageError(err)
	defer t.storageError(nil)
	fn()
}

func (t *Tester) ClusterScope() *Tester {
	t.clusterScope = true
	return t
}

func (t *Tester) AllowCreateOnUpdate() *Tester {
	t.createOnUpdate = true
	return t
}

func (t *Tester) GeneratesName() *Tester {
	t.generatesName = true
	return t
}

func (t *Tester) ReturnDeletedObject() *Tester {
	t.returnDeletedObject = true
	return t
}

// TestNamespace returns the namespace that will be used when creating contexts.
// Returns NamespaceNone for cluster-scoped objects.
func (t *Tester) TestNamespace() string {
	if t.clusterScope {
		return api.NamespaceNone
	}
	return "test"
}

// TestContext returns a namespaced context that will be used when making storage calls.
// Namespace is determined by TestNamespace()
func (t *Tester) TestContext() api.Context {
	if t.clusterScope {
		return api.NewContext()
	}
	return api.WithNamespace(api.NewContext(), t.TestNamespace())
}

func (t *Tester) getObjectMetaOrFail(obj runtime.Object) *api.ObjectMeta {
	meta, err := api.ObjectMetaFor(obj)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, obj)
	}
	return meta
}

func (t *Tester) setObjectMeta(obj runtime.Object, name string) {
	meta := t.getObjectMetaOrFail(obj)
	meta.Name = name
	if t.clusterScope {
		meta.Namespace = api.NamespaceNone
	} else {
		meta.Namespace = api.NamespaceValue(t.TestContext())
	}
	meta.GenerateName = ""
}

func copyOrDie(obj runtime.Object) runtime.Object {
	out, err := api.Scheme.Copy(obj)
	if err != nil {
		panic(err)
	}
	return out
}

type AssignFunc func([]runtime.Object) []runtime.Object
type EmitFunc func(runtime.Object, string) error
type GetFunc func(api.Context, runtime.Object) (runtime.Object, error)
type InitWatchFunc func()
type InjectErrFunc func(err error)
type IsErrorFunc func(err error) bool
type SetFunc func(api.Context, runtime.Object) error
type SetRVFunc func(uint64)
type UpdateFunc func(runtime.Object) runtime.Object

// Test creating an object.
func (t *Tester) TestCreate(valid runtime.Object, setFn SetFunc, getFn GetFunc, invalid ...runtime.Object) {
	t.testCreateHasMetadata(copyOrDie(valid))
	if !t.generatesName {
		t.testCreateGeneratesName(copyOrDie(valid))
		t.testCreateGeneratesNameReturnsServerTimeout(copyOrDie(valid))
	}
	t.testCreateEquals(copyOrDie(valid), getFn)
	t.testCreateAlreadyExisting(copyOrDie(valid), setFn)
	if t.clusterScope {
		t.testCreateDiscardsObjectNamespace(copyOrDie(valid))
		t.testCreateIgnoresContextNamespace(copyOrDie(valid))
		t.testCreateIgnoresMismatchedNamespace(copyOrDie(valid))
	} else {
		t.testCreateRejectsMismatchedNamespace(copyOrDie(valid))
	}
	t.testCreateInvokesValidation(invalid...)
}

// Test updating an object.
func (t *Tester) TestUpdate(valid runtime.Object, setFn SetFunc, setRVFn SetRVFunc, getFn GetFunc, updateFn UpdateFunc, invalidUpdateFn ...UpdateFunc) {
	t.testUpdateEquals(copyOrDie(valid), setFn, getFn, updateFn)
	t.testUpdateFailsOnVersionTooOld(copyOrDie(valid), setFn, setRVFn)
	t.testUpdateOnNotFound(copyOrDie(valid))
	if !t.clusterScope {
		t.testUpdateRejectsMismatchedNamespace(copyOrDie(valid), setFn)
	}
	t.testUpdateInvokesValidation(copyOrDie(valid), setFn, invalidUpdateFn...)
}

// Test deleting an object.
func (t *Tester) TestDelete(valid runtime.Object, setFn SetFunc, getFn GetFunc, isNotFoundFn IsErrorFunc) {
	t.testDeleteNonExist(copyOrDie(valid))
	t.testDeleteNoGraceful(copyOrDie(valid), setFn, getFn, isNotFoundFn)
}

// Test gracefully deleting an object.
func (t *Tester) TestDeleteGraceful(valid runtime.Object, setFn SetFunc, getFn GetFunc, expectedGrace int64) {
	t.testDeleteGracefulHasDefault(copyOrDie(valid), setFn, getFn, expectedGrace)
	t.testDeleteGracefulWithValue(copyOrDie(valid), setFn, getFn, expectedGrace)
	t.testDeleteGracefulUsesZeroOnNil(copyOrDie(valid), setFn, expectedGrace)
	t.testDeleteGracefulExtend(copyOrDie(valid), setFn, getFn, expectedGrace)
	t.testDeleteGracefulImmediate(copyOrDie(valid), setFn, getFn, expectedGrace)
}

// Test getting object.
func (t *Tester) TestGet(valid runtime.Object) {
	t.testGetFound(copyOrDie(valid))
	t.testGetNotFound(copyOrDie(valid))
	t.testGetMimatchedNamespace(copyOrDie(valid))
	if !t.clusterScope {
		t.testGetDifferentNamespace(copyOrDie(valid))
	}
}

// Test listing objects.
func (t *Tester) TestList(valid runtime.Object, assignFn AssignFunc, setRVFn SetRVFunc) {
	t.testListError()
	t.testListFound(copyOrDie(valid), assignFn)
	t.testListNotFound(assignFn, setRVFn)
	t.testListMatchLabels(copyOrDie(valid), assignFn)
}

// Test watching objects.
func (t *Tester) TestWatch(
	valid runtime.Object, initWatchFn InitWatchFunc, injectErrFn InjectErrFunc, emitFn EmitFunc,
	labelsPass, labelsFail []labels.Set, fieldsPass, fieldsFail []fields.Set, actions []string) {
	t.testWatch(initWatchFn, injectErrFn)
	t.testWatchLabels(copyOrDie(valid), initWatchFn, emitFn, labelsPass, labelsFail, actions)
	t.testWatchFields(copyOrDie(valid), initWatchFn, emitFn, fieldsPass, fieldsFail, actions)
}

// =============================================================================
// Creation tests.

func (t *Tester) testCreateAlreadyExisting(obj runtime.Object, setFn SetFunc) {
	ctx := t.TestContext()

	foo := copyOrDie(obj)
	t.setObjectMeta(foo, "foo1")
	if err := setFn(ctx, foo); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	_, err := t.storage.(rest.Creater).Create(ctx, foo)
	if !errors.IsAlreadyExists(err) {
		t.Errorf("expected already exists err, got %v", err)
	}
}

func (t *Tester) testCreateEquals(obj runtime.Object, getFn GetFunc) {
	ctx := t.TestContext()

	foo := copyOrDie(obj)
	t.setObjectMeta(foo, "foo2")

	created, err := t.storage.(rest.Creater).Create(ctx, foo)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	got, err := getFn(ctx, foo)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Set resource version which might be unset in created object.
	createdMeta := t.getObjectMetaOrFail(created)
	gotMeta := t.getObjectMetaOrFail(got)
	createdMeta.ResourceVersion = gotMeta.ResourceVersion

	if e, a := created, got; !api.Semantic.DeepEqual(e, a) {
		t.Errorf("unexpected obj: %#v, expected %#v", e, a)
	}
}

func (t *Tester) testCreateDiscardsObjectNamespace(valid runtime.Object) {
	objectMeta := t.getObjectMetaOrFail(valid)

	// Ignore non-empty namespace in object meta
	objectMeta.Namespace = "not-default"

	// Ideally, we'd get an error back here, but at least verify the namespace wasn't persisted
	created, err := t.storage.(rest.Creater).Create(t.TestContext(), copyOrDie(valid))
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	createdObjectMeta := t.getObjectMetaOrFail(created)
	if createdObjectMeta.Namespace != api.NamespaceNone {
		t.Errorf("Expected empty namespace on created object, got '%v'", createdObjectMeta.Namespace)
	}
}

func (t *Tester) testCreateGeneratesName(valid runtime.Object) {
	objectMeta := t.getObjectMetaOrFail(valid)
	objectMeta.Name = ""
	objectMeta.GenerateName = "test-"

	_, err := t.storage.(rest.Creater).Create(t.TestContext(), valid)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if objectMeta.Name == "test-" || !strings.HasPrefix(objectMeta.Name, "test-") {
		t.Errorf("unexpected name: %#v", valid)
	}
}

func (t *Tester) testCreateGeneratesNameReturnsServerTimeout(valid runtime.Object) {
	objectMeta := t.getObjectMetaOrFail(valid)
	objectMeta.Name = ""
	objectMeta.GenerateName = "test-"
	t.withStorageError(errors.NewAlreadyExists("kind", "thing"), func() {
		_, err := t.storage.(rest.Creater).Create(t.TestContext(), valid)
		if err == nil || !errors.IsServerTimeout(err) {
			t.Fatalf("Unexpected error: %v", err)
		}
	})
}

func (t *Tester) testCreateHasMetadata(valid runtime.Object) {
	objectMeta := t.getObjectMetaOrFail(valid)
	objectMeta.Name = ""
	objectMeta.GenerateName = "test-"
	objectMeta.Namespace = t.TestNamespace()

	obj, err := t.storage.(rest.Creater).Create(t.TestContext(), valid)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if obj == nil {
		t.Fatalf("Unexpected object from result: %#v", obj)
	}
	if !api.HasObjectMetaSystemFieldValues(objectMeta) {
		t.Errorf("storage did not populate object meta field values")
	}
}

func (t *Tester) testCreateIgnoresContextNamespace(valid runtime.Object) {
	// Ignore non-empty namespace in context
	ctx := api.WithNamespace(api.NewContext(), "not-default2")

	// Ideally, we'd get an error back here, but at least verify the namespace wasn't persisted
	created, err := t.storage.(rest.Creater).Create(ctx, copyOrDie(valid))
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	createdObjectMeta := t.getObjectMetaOrFail(created)
	if createdObjectMeta.Namespace != api.NamespaceNone {
		t.Errorf("Expected empty namespace on created object, got '%v'", createdObjectMeta.Namespace)
	}
}

func (t *Tester) testCreateIgnoresMismatchedNamespace(valid runtime.Object) {
	objectMeta := t.getObjectMetaOrFail(valid)

	// Ignore non-empty namespace in object meta
	objectMeta.Namespace = "not-default"
	ctx := api.WithNamespace(api.NewContext(), "not-default2")

	// Ideally, we'd get an error back here, but at least verify the namespace wasn't persisted
	created, err := t.storage.(rest.Creater).Create(ctx, copyOrDie(valid))
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	createdObjectMeta := t.getObjectMetaOrFail(created)
	if createdObjectMeta.Namespace != api.NamespaceNone {
		t.Errorf("Expected empty namespace on created object, got '%v'", createdObjectMeta.Namespace)
	}
}

func (t *Tester) testCreateInvokesValidation(invalid ...runtime.Object) {
	for i, obj := range invalid {
		ctx := t.TestContext()
		_, err := t.storage.(rest.Creater).Create(ctx, obj)
		if !errors.IsInvalid(err) {
			t.Errorf("%d: Expected to get an invalid resource error, got %v", i, err)
		}
	}
}

func (t *Tester) testCreateRejectsMismatchedNamespace(valid runtime.Object) {
	objectMeta := t.getObjectMetaOrFail(valid)
	objectMeta.Namespace = "not-default"

	_, err := t.storage.(rest.Creater).Create(t.TestContext(), valid)
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if !strings.Contains(err.Error(), "does not match the namespace sent on the request") {
		t.Errorf("Expected 'does not match the namespace sent on the request' error, got '%v'", err.Error())
	}
}

func (t *Tester) testCreateResetsUserData(valid runtime.Object) {
	objectMeta := t.getObjectMetaOrFail(valid)
	now := unversioned.Now()
	objectMeta.UID = "bad-uid"
	objectMeta.CreationTimestamp = now

	obj, err := t.storage.(rest.Creater).Create(t.TestContext(), valid)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if obj == nil {
		t.Fatalf("Unexpected object from result: %#v", obj)
	}
	if objectMeta.UID == "bad-uid" || objectMeta.CreationTimestamp == now {
		t.Errorf("ObjectMeta did not reset basic fields: %#v", objectMeta)
	}
}

// =============================================================================
// Update tests.

func (t *Tester) testUpdateEquals(obj runtime.Object, setFn SetFunc, getFn GetFunc, updateFn UpdateFunc) {
	ctx := t.TestContext()

	foo := copyOrDie(obj)
	t.setObjectMeta(foo, "foo2")
	if err := setFn(ctx, foo); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	toUpdate, err := getFn(ctx, foo)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	toUpdate = updateFn(toUpdate)
	updated, created, err := t.storage.(rest.Updater).Update(ctx, toUpdate)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if created {
		t.Errorf("unexpected creation")
	}
	got, err := getFn(ctx, foo)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	// Set resource version which might be unset in created object.
	updatedMeta := t.getObjectMetaOrFail(updated)
	gotMeta := t.getObjectMetaOrFail(got)
	updatedMeta.ResourceVersion = gotMeta.ResourceVersion

	if e, a := updated, got; !api.Semantic.DeepEqual(e, a) {
		t.Errorf("unexpected obj: %#v, expected %#v", e, a)
	}
}

func (t *Tester) testUpdateFailsOnVersionTooOld(obj runtime.Object, setFn SetFunc, setRVFn SetRVFunc) {
	ctx := t.TestContext()

	foo := copyOrDie(obj)
	t.setObjectMeta(foo, "foo3")

	setRVFn(10)
	if err := setFn(ctx, foo); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	older := copyOrDie(foo)
	olderMeta := t.getObjectMetaOrFail(older)
	olderMeta.ResourceVersion = "8"

	_, _, err := t.storage.(rest.Updater).Update(t.TestContext(), older)
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if !errors.IsConflict(err) {
		t.Errorf("Expected Conflict error, got '%v'", err)
	}
}

func (t *Tester) testUpdateInvokesValidation(obj runtime.Object, setFn SetFunc, invalidUpdateFn ...UpdateFunc) {
	ctx := t.TestContext()

	foo := copyOrDie(obj)
	t.setObjectMeta(foo, "foo4")
	if err := setFn(ctx, foo); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	for _, update := range invalidUpdateFn {
		toUpdate := update(copyOrDie(foo))
		got, created, err := t.storage.(rest.Updater).Update(t.TestContext(), toUpdate)
		if got != nil || created {
			t.Errorf("expected nil object and no creation for object: %v", toUpdate)
		}
		if !errors.IsInvalid(err) && !errors.IsBadRequest(err) {
			t.Errorf("expected invalid or bad request error, got %v", err)
		}
	}
}

func (t *Tester) testUpdateOnNotFound(obj runtime.Object) {
	t.setObjectMeta(obj, "foo")
	_, created, err := t.storage.(rest.Updater).Update(t.TestContext(), obj)
	if t.createOnUpdate {
		if err != nil {
			t.Errorf("creation allowed on updated, but got an error: %v", err)
		}
		if !created {
			t.Errorf("creation allowed on update, but object not created")
		}
	} else {
		if err == nil {
			t.Errorf("Expected an error, but we didn't get one")
		} else if !errors.IsNotFound(err) {
			t.Errorf("Expected NotFound error, got '%v'", err)
		}
	}
}

func (t *Tester) testUpdateRejectsMismatchedNamespace(obj runtime.Object, setFn SetFunc) {
	ctx := t.TestContext()

	foo := copyOrDie(obj)
	t.setObjectMeta(foo, "foo1")
	if err := setFn(ctx, foo); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	objectMeta := t.getObjectMetaOrFail(obj)
	objectMeta.Name = "foo1"
	objectMeta.Namespace = "not-default"

	obj, updated, err := t.storage.(rest.Updater).Update(t.TestContext(), obj)
	if obj != nil || updated {
		t.Errorf("expected nil object and not updated")
	}
	if err == nil {
		t.Errorf("expected an error, but didn't get one")
	} else if !strings.Contains(err.Error(), "does not match the namespace sent on the request") {
		t.Errorf("expected 'does not match the namespace sent on the request' error, got '%v'", err.Error())
	}
}

// =============================================================================
// Deletion tests.

func (t *Tester) testDeleteNoGraceful(obj runtime.Object, setFn SetFunc, getFn GetFunc, isNotFoundFn IsErrorFunc) {
	ctx := t.TestContext()

	foo := copyOrDie(obj)
	t.setObjectMeta(foo, "foo1")
	if err := setFn(ctx, foo); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	objectMeta := t.getObjectMetaOrFail(foo)
	obj, err := t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, api.NewDeleteOptions(10))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !t.returnDeletedObject {
		if status, ok := obj.(*unversioned.Status); !ok {
			t.Errorf("expected status of delete, got %v", status)
		} else if status.Status != unversioned.StatusSuccess {
			t.Errorf("expected success, got: %v", status.Status)
		}
	}

	_, err = getFn(ctx, foo)
	if err == nil || !isNotFoundFn(err) {
		t.Errorf("unexpected error: %v", err)
	}
}

func (t *Tester) testDeleteNonExist(obj runtime.Object) {
	objectMeta := t.getObjectMetaOrFail(obj)

	t.withStorageError(tools.EtcdErrorNotFound, func() {
		_, err := t.storage.(rest.GracefulDeleter).Delete(t.TestContext(), objectMeta.Name, nil)
		if err == nil || !errors.IsNotFound(err) {
			t.Errorf("unexpected error: %v", err)
		}
	})
}

// =============================================================================
// Graceful Deletion tests.

func (t *Tester) testDeleteGracefulHasDefault(obj runtime.Object, setFn SetFunc, getFn GetFunc, expectedGrace int64) {
	ctx := t.TestContext()

	foo := copyOrDie(obj)
	t.setObjectMeta(foo, "foo1")
	if err := setFn(ctx, foo); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	objectMeta := t.getObjectMetaOrFail(foo)
	_, err := t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, &api.DeleteOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if _, err := getFn(ctx, foo); err != nil {
		t.Fatalf("did not gracefully delete resource", err)
	}

	object, err := t.storage.(rest.Getter).Get(ctx, objectMeta.Name)
	if err != nil {
		t.Fatalf("unexpected error, object should exist: %v", err)
	}
	objectMeta = t.getObjectMetaOrFail(object)
	if objectMeta.DeletionTimestamp == nil || objectMeta.DeletionGracePeriodSeconds == nil || *objectMeta.DeletionGracePeriodSeconds != expectedGrace {
		t.Errorf("unexpected deleted meta: %#v", objectMeta)
	}
}

func (t *Tester) testDeleteGracefulWithValue(obj runtime.Object, setFn SetFunc, getFn GetFunc, expectedGrace int64) {
	ctx := t.TestContext()

	foo := copyOrDie(obj)
	t.setObjectMeta(foo, "foo2")
	if err := setFn(ctx, foo); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	objectMeta := t.getObjectMetaOrFail(foo)
	_, err := t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, api.NewDeleteOptions(expectedGrace+2))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if _, err := getFn(ctx, foo); err != nil {
		t.Fatalf("did not gracefully delete resource", err)
	}

	object, err := t.storage.(rest.Getter).Get(ctx, objectMeta.Name)
	if err != nil {
		t.Errorf("unexpected error, object should exist: %v", err)
	}
	objectMeta = t.getObjectMetaOrFail(object)
	if objectMeta.DeletionTimestamp == nil || objectMeta.DeletionGracePeriodSeconds == nil || *objectMeta.DeletionGracePeriodSeconds != expectedGrace+2 {
		t.Errorf("unexpected deleted meta: %#v", objectMeta)
	}
}

func (t *Tester) testDeleteGracefulExtend(obj runtime.Object, setFn SetFunc, getFn GetFunc, expectedGrace int64) {
	ctx := t.TestContext()

	foo := copyOrDie(obj)
	t.setObjectMeta(foo, "foo3")
	if err := setFn(ctx, foo); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	objectMeta := t.getObjectMetaOrFail(foo)
	_, err := t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, api.NewDeleteOptions(expectedGrace))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if _, err := getFn(ctx, foo); err != nil {
		t.Fatalf("did not gracefully delete resource", err)
	}

	// second delete duration is ignored
	_, err = t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, api.NewDeleteOptions(expectedGrace+2))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	object, err := t.storage.(rest.Getter).Get(ctx, objectMeta.Name)
	if err != nil {
		t.Errorf("unexpected error, object should exist: %v", err)
	}
	objectMeta = t.getObjectMetaOrFail(object)
	if objectMeta.DeletionTimestamp == nil || objectMeta.DeletionGracePeriodSeconds == nil || *objectMeta.DeletionGracePeriodSeconds != expectedGrace {
		t.Errorf("unexpected deleted meta: %#v", objectMeta)
	}
}

func (t *Tester) testDeleteGracefulImmediate(obj runtime.Object, setFn SetFunc, getFn GetFunc, expectedGrace int64) {
	ctx := t.TestContext()

	foo := copyOrDie(obj)
	t.setObjectMeta(foo, "foo4")
	if err := setFn(ctx, foo); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	objectMeta := t.getObjectMetaOrFail(foo)
	_, err := t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, api.NewDeleteOptions(expectedGrace))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if _, err := getFn(ctx, foo); err != nil {
		t.Fatalf("did not gracefully delete resource", err)
	}

	// second delete is immediate, resource is deleted
	out, err := t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, api.NewDeleteOptions(0))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	_, err = t.storage.(rest.Getter).Get(ctx, objectMeta.Name)
	if !errors.IsNotFound(err) {
		t.Errorf("unexpected error, object should be deleted immediately: %v", err)
	}
	objectMeta = t.getObjectMetaOrFail(out)
	if objectMeta.DeletionTimestamp == nil || objectMeta.DeletionGracePeriodSeconds == nil || *objectMeta.DeletionGracePeriodSeconds != 0 {
		t.Errorf("unexpected deleted meta: %#v", objectMeta)
	}
}

func (t *Tester) testDeleteGracefulUsesZeroOnNil(obj runtime.Object, setFn SetFunc, expectedGrace int64) {
	ctx := t.TestContext()

	foo := copyOrDie(obj)
	t.setObjectMeta(foo, "foo5")
	if err := setFn(ctx, foo); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	objectMeta := t.getObjectMetaOrFail(foo)
	_, err := t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if _, err := t.storage.(rest.Getter).Get(ctx, objectMeta.Name); !errors.IsNotFound(err) {
		t.Errorf("unexpected error, object should not exist: %v", err)
	}
}

// =============================================================================
// Get tests.

// testGetDifferentNamespace ensures same-name objects in different namespaces do not clash
func (t *Tester) testGetDifferentNamespace(obj runtime.Object) {
	if t.clusterScope {
		t.Fatalf("the test does not work in in cluster-scope")
	}

	objMeta := t.getObjectMetaOrFail(obj)
	objMeta.Name = "foo5"

	ctx1 := api.WithNamespace(api.NewContext(), "bar3")
	objMeta.Namespace = api.NamespaceValue(ctx1)
	_, err := t.storage.(rest.Creater).Create(ctx1, obj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	ctx2 := api.WithNamespace(api.NewContext(), "bar4")
	objMeta.Namespace = api.NamespaceValue(ctx2)
	_, err = t.storage.(rest.Creater).Create(ctx2, obj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	got1, err := t.storage.(rest.Getter).Get(ctx1, objMeta.Name)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	got1Meta := t.getObjectMetaOrFail(got1)
	if got1Meta.Name != objMeta.Name {
		t.Errorf("unexpected name of object: %#v, expected: %s", got1, objMeta.Name)
	}
	if got1Meta.Namespace != api.NamespaceValue(ctx1) {
		t.Errorf("unexpected namespace of object: %#v, expected: %s", got1, api.NamespaceValue(ctx1))
	}

	got2, err := t.storage.(rest.Getter).Get(ctx2, objMeta.Name)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	got2Meta := t.getObjectMetaOrFail(got2)
	if got2Meta.Name != objMeta.Name {
		t.Errorf("unexpected name of object: %#v, expected: %s", got2, objMeta.Name)
	}
	if got2Meta.Namespace != api.NamespaceValue(ctx2) {
		t.Errorf("unexpected namespace of object: %#v, expected: %s", got2, api.NamespaceValue(ctx2))
	}
}

func (t *Tester) testGetFound(obj runtime.Object) {
	ctx := t.TestContext()
	t.setObjectMeta(obj, "foo1")

	existing, err := t.storage.(rest.Creater).Create(ctx, obj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	existingMeta := t.getObjectMetaOrFail(existing)

	got, err := t.storage.(rest.Getter).Get(ctx, "foo1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	gotMeta := t.getObjectMetaOrFail(got)
	gotMeta.ResourceVersion = existingMeta.ResourceVersion
	if e, a := existing, got; !api.Semantic.DeepEqual(e, a) {
		t.Errorf("unexpected obj: %#v, expected %#v", e, a)
	}
}

func (t *Tester) testGetMimatchedNamespace(obj runtime.Object) {
	ctx1 := api.WithNamespace(api.NewContext(), "bar1")
	ctx2 := api.WithNamespace(api.NewContext(), "bar2")
	objMeta := t.getObjectMetaOrFail(obj)
	objMeta.Name = "foo4"
	objMeta.Namespace = api.NamespaceValue(ctx1)
	_, err := t.storage.(rest.Creater).Create(ctx1, obj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	_, err = t.storage.(rest.Getter).Get(ctx2, "foo4")
	if t.clusterScope {
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	} else {
		if !errors.IsNotFound(err) {
			t.Errorf("unexpected error returned: %#v", err)
		}
	}
}

func (t *Tester) testGetNotFound(obj runtime.Object) {
	ctx := t.TestContext()
	t.setObjectMeta(obj, "foo2")
	_, err := t.storage.(rest.Creater).Create(ctx, obj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	_, err = t.storage.(rest.Getter).Get(ctx, "foo3")
	if !errors.IsNotFound(err) {
		t.Errorf("unexpected error returned: %#v", err)
	}
}

// =============================================================================
// List tests.

func listToItems(listObj runtime.Object) ([]runtime.Object, error) {
	v, err := conversion.EnforcePtr(listObj)
	if err != nil {
		return nil, fmt.Errorf("unexpected error: %v", err)
	}
	items := v.FieldByName("Items")
	if !items.IsValid() {
		return nil, fmt.Errorf("unexpected Items field in %v", listObj)
	}
	if items.Type().Kind() != reflect.Slice {
		return nil, fmt.Errorf("unexpected Items field type: %v", items.Type().Kind())
	}
	result := make([]runtime.Object, items.Len())
	for i := 0; i < items.Len(); i++ {
		result[i] = items.Index(i).Addr().Interface().(runtime.Object)
	}
	return result, nil
}

func (t *Tester) testListError() {
	ctx := t.TestContext()

	storageError := fmt.Errorf("test error")
	t.withStorageError(storageError, func() {
		_, err := t.storage.(rest.Lister).List(ctx, labels.Everything(), fields.Everything())
		if err != storageError {
			t.Errorf("unexpected error: %v", err)
		}
	})
}

func (t *Tester) testListFound(obj runtime.Object, assignFn AssignFunc) {
	ctx := t.TestContext()

	foo1 := copyOrDie(obj)
	t.setObjectMeta(foo1, "foo1")
	foo2 := copyOrDie(obj)
	t.setObjectMeta(foo2, "foo2")

	existing := assignFn([]runtime.Object{foo1, foo2})

	listObj, err := t.storage.(rest.Lister).List(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	items, err := listToItems(listObj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(items) != len(existing) {
		t.Errorf("unexpected number of items: %v", len(items))
	}
	if !api.Semantic.DeepEqual(existing, items) {
		t.Errorf("expected: %#v, got: %#v", existing, items)
	}
}

func (t *Tester) testListMatchLabels(obj runtime.Object, assignFn AssignFunc) {
	ctx := t.TestContext()
	testLabels := map[string]string{"key": "value"}

	foo1 := copyOrDie(obj)
	t.setObjectMeta(foo1, "foo1")
	foo2 := copyOrDie(obj)
	foo2Meta := t.getObjectMetaOrFail(foo2)
	foo2Meta.Name = "foo2"
	foo2Meta.Namespace = api.NamespaceValue(ctx)
	foo2Meta.Labels = testLabels

	existing := assignFn([]runtime.Object{foo1, foo2})
	filtered := []runtime.Object{existing[1]}

	selector := labels.SelectorFromSet(labels.Set(testLabels))
	listObj, err := t.storage.(rest.Lister).List(ctx, selector, fields.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	items, err := listToItems(listObj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(items) != len(filtered) {
		t.Errorf("unexpected number of items: %v", len(items))
	}
	if !api.Semantic.DeepEqual(filtered, items) {
		t.Errorf("expected: %#v, got: %#v", filtered, items)
	}
}

func (t *Tester) testListNotFound(assignFn AssignFunc, setRVFn SetRVFunc) {
	ctx := t.TestContext()

	setRVFn(uint64(123))
	_ = assignFn([]runtime.Object{})

	listObj, err := t.storage.(rest.Lister).List(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	items, err := listToItems(listObj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(items) != 0 {
		t.Errorf("unexpected items: %v", items)
	}

	meta, err := api.ListMetaFor(listObj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if meta.ResourceVersion != "123" {
		t.Errorf("unexpected resource version: %d", meta.ResourceVersion)
	}
}

// =============================================================================
// Watching tests.

func (t *Tester) testWatch(initWatchFn InitWatchFunc, injectErrFn InjectErrFunc) {
	ctx := t.TestContext()
	watcher, err := t.storage.(rest.Watcher).Watch(ctx, labels.Everything(), fields.Everything(), "1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	initWatchFn()

	select {
	case _, ok := <-watcher.ResultChan():
		if !ok {
			t.Errorf("watch channel should be open")
		}
	default:
	}

	injectErrFn(nil)
	if _, ok := <-watcher.ResultChan(); ok {
		t.Errorf("watch channel should be closed")
	}
	watcher.Stop()
}

func (t *Tester) testWatchFields(obj runtime.Object, initWatchFn InitWatchFunc, emitFn EmitFunc, fieldsPass, fieldsFail []fields.Set, actions []string) {
	ctx := t.TestContext()

	for _, field := range fieldsPass {
		for _, action := range actions {
			watcher, err := t.storage.(rest.Watcher).Watch(ctx, labels.Everything(), field.AsSelector(), "1")
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			initWatchFn()
			if err := emitFn(obj, action); err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			select {
			case _, ok := <-watcher.ResultChan():
				if !ok {
					t.Errorf("watch channel should be open")
				}
			case <-time.After(util.ForeverTestTimeout):
				t.Errorf("unexpected timeout from result channel")
			}
			watcher.Stop()
		}
	}

	for _, field := range fieldsFail {
		for _, action := range actions {
			watcher, err := t.storage.(rest.Watcher).Watch(ctx, labels.Everything(), field.AsSelector(), "1")
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			initWatchFn()
			if err := emitFn(obj, action); err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			select {
			case <-watcher.ResultChan():
				t.Errorf("unexpected result from result channel")
			case <-time.After(time.Millisecond * 500):
				// expected case
			}
			watcher.Stop()
		}
	}
}

func (t *Tester) testWatchLabels(obj runtime.Object, initWatchFn InitWatchFunc, emitFn EmitFunc, labelsPass, labelsFail []labels.Set, actions []string) {
	ctx := t.TestContext()

	for _, label := range labelsPass {
		for _, action := range actions {
			watcher, err := t.storage.(rest.Watcher).Watch(ctx, label.AsSelector(), fields.Everything(), "1")
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			initWatchFn()
			if err := emitFn(obj, action); err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			select {
			case _, ok := <-watcher.ResultChan():
				if !ok {
					t.Errorf("watch channel should be open")
				}
			case <-time.After(util.ForeverTestTimeout):
				t.Errorf("unexpected timeout from result channel")
			}
			watcher.Stop()
		}
	}

	for _, label := range labelsFail {
		for _, action := range actions {
			watcher, err := t.storage.(rest.Watcher).Watch(ctx, label.AsSelector(), fields.Everything(), "1")
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			initWatchFn()
			if err := emitFn(obj, action); err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			select {
			case <-watcher.ResultChan():
				t.Errorf("unexpected result from result channel")
			case <-time.After(time.Millisecond * 500):
				// expected case
			}
			watcher.Stop()
		}
	}
}
