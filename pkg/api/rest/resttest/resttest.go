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

	"github.com/coreos/go-etcd/etcd"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/util"
)

type Tester struct {
	*testing.T
	storage       rest.Storage
	storageError  injectErrorFunc
	clusterScope  bool
	generatesName bool
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

func (t *Tester) GeneratesName() *Tester {
	t.generatesName = true
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

func copyOrDie(obj runtime.Object) runtime.Object {
	out, err := api.Scheme.Copy(obj)
	if err != nil {
		panic(err)
	}
	return out
}

type AssignFunc func([]runtime.Object) []runtime.Object
type GetFunc func(api.Context, runtime.Object) (runtime.Object, error)
type SetFunc func(api.Context, runtime.Object) error
type SetRVFunc func(uint64)

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
func (t *Tester) TestUpdate(valid runtime.Object, existing, older runtime.Object) {
	t.testUpdateFailsOnNotFound(copyOrDie(valid))
	t.testUpdateFailsOnVersion(copyOrDie(older))
}

// Test deleting an object.
// TODO(wojtek-t): Change it to use AssignFunc instead.
func (t *Tester) TestDelete(createFn func() runtime.Object, wasGracefulFn func() bool, invalid ...runtime.Object) {
	t.TestDeleteNonExist(createFn)
	t.TestDeleteNoGraceful(createFn, wasGracefulFn)
	t.TestDeleteInvokesValidation(invalid...)
	// TODO: Test delete namespace mismatch rejection
	// once #5684 is fixed.
}

// Test graceful deletion.
// TODO(wojtek-t): Change it to use AssignFunc instead.
func (t *Tester) TestDeleteGraceful(createFn func() runtime.Object, expectedGrace int64, wasGracefulFn func() bool) {
	t.TestDeleteGracefulHasDefault(createFn(), expectedGrace, wasGracefulFn)
	t.TestDeleteGracefulWithValue(createFn(), expectedGrace, wasGracefulFn)
	t.TestDeleteGracefulUsesZeroOnNil(createFn(), 0)
	t.TestDeleteGracefulExtend(createFn(), expectedGrace, wasGracefulFn)
	t.TestDeleteGracefulImmediate(createFn(), expectedGrace, wasGracefulFn)
}

// Test getting object.
func (t *Tester) TestGet(obj runtime.Object) {
	t.testGetFound(obj)
	t.testGetNotFound(obj)
	t.testGetMimatchedNamespace(obj)
	if !t.clusterScope {
		t.testGetDifferentNamespace(obj)
	}
}

// Test listing object.
func (t *Tester) TestList(obj runtime.Object, assignFn AssignFunc, setRVFn SetRVFunc) {
	t.testListError()
	t.testListFound(obj, assignFn)
	t.testListNotFound(assignFn, setRVFn)
	t.testListMatchLabels(obj, assignFn)
}

// =============================================================================
// Creation tests.

func (t *Tester) testCreateAlreadyExisting(obj runtime.Object, setFn SetFunc) {
	ctx := t.TestContext()

	foo := copyOrDie(obj)
	fooMeta := t.getObjectMetaOrFail(foo)
	fooMeta.Name = "foo1"
	fooMeta.Namespace = api.NamespaceValue(ctx)
	fooMeta.GenerateName = ""
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
	fooMeta := t.getObjectMetaOrFail(foo)
	fooMeta.Name = "foo2"
	fooMeta.Namespace = api.NamespaceValue(ctx)
	fooMeta.GenerateName = ""

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
	now := util.Now()
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

func (t *Tester) testUpdateFailsOnNotFound(valid runtime.Object) {
	_, _, err := t.storage.(rest.Updater).Update(t.TestContext(), valid)
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if !errors.IsNotFound(err) {
		t.Errorf("Expected NotFound error, got '%v'", err)
	}
}

func (t *Tester) testUpdateFailsOnVersion(older runtime.Object) {
	_, _, err := t.storage.(rest.Updater).Update(t.TestContext(), older)
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if !errors.IsConflict(err) {
		t.Errorf("Expected Conflict error, got '%v'", err)
	}
}

// =============================================================================
// Deletion tests.

func (t *Tester) TestDeleteInvokesValidation(invalid ...runtime.Object) {
	for i, obj := range invalid {
		objectMeta := t.getObjectMetaOrFail(obj)
		ctx := t.TestContext()
		_, err := t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, nil)
		if !errors.IsInvalid(err) {
			t.Errorf("%d: Expected to get an invalid resource error, got %v", i, err)
		}
	}
}

func (t *Tester) TestDeleteNonExist(createFn func() runtime.Object) {
	existing := createFn()
	objectMeta := t.getObjectMetaOrFail(existing)
	context := t.TestContext()

	t.withStorageError(&etcd.EtcdError{ErrorCode: tools.EtcdErrorCodeNotFound}, func() {
		_, err := t.storage.(rest.GracefulDeleter).Delete(context, objectMeta.Name, nil)
		if err == nil || !errors.IsNotFound(err) {
			t.Fatalf("Unexpected error: %v", err)
		}
	})
}

// =============================================================================
// Graceful Deletion tests.

func (t *Tester) TestDeleteNoGraceful(createFn func() runtime.Object, wasGracefulFn func() bool) {
	existing := createFn()
	objectMeta := t.getObjectMetaOrFail(existing)
	ctx := api.WithNamespace(t.TestContext(), objectMeta.Namespace)
	_, err := t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, api.NewDeleteOptions(10))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if _, err := t.storage.(rest.Getter).Get(ctx, objectMeta.Name); !errors.IsNotFound(err) {
		t.Errorf("unexpected error, object should not exist: %v", err)
	}
	if wasGracefulFn() {
		t.Errorf("resource should not support graceful delete")
	}
}

func (t *Tester) TestDeleteGracefulHasDefault(existing runtime.Object, expectedGrace int64, wasGracefulFn func() bool) {
	objectMeta := t.getObjectMetaOrFail(existing)
	ctx := api.WithNamespace(t.TestContext(), objectMeta.Namespace)
	_, err := t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, &api.DeleteOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !wasGracefulFn() {
		t.Errorf("did not gracefully delete resource")
		return
	}
	object, err := t.storage.(rest.Getter).Get(ctx, objectMeta.Name)
	if err != nil {
		t.Errorf("unexpected error, object should exist: %v", err)
		return
	}
	objectMeta, err = api.ObjectMetaFor(object)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, object)
	}
	if objectMeta.DeletionTimestamp == nil {
		t.Errorf("did not set deletion timestamp")
	}
	if objectMeta.DeletionGracePeriodSeconds == nil {
		t.Fatalf("did not set deletion grace period seconds")
	}
	if *objectMeta.DeletionGracePeriodSeconds != expectedGrace {
		t.Errorf("actual grace period does not match expected: %d", *objectMeta.DeletionGracePeriodSeconds)
	}
}

func (t *Tester) TestDeleteGracefulWithValue(existing runtime.Object, expectedGrace int64, wasGracefulFn func() bool) {
	objectMeta, err := api.ObjectMetaFor(existing)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, existing)
	}

	ctx := api.WithNamespace(t.TestContext(), objectMeta.Namespace)
	_, err = t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, api.NewDeleteOptions(expectedGrace+2))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !wasGracefulFn() {
		t.Errorf("did not gracefully delete resource")
	}
	object, err := t.storage.(rest.Getter).Get(ctx, objectMeta.Name)
	if err != nil {
		t.Errorf("unexpected error, object should exist: %v", err)
	}
	objectMeta, err = api.ObjectMetaFor(object)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, object)
	}
	if objectMeta.DeletionTimestamp == nil {
		t.Errorf("did not set deletion timestamp")
	}
	if objectMeta.DeletionGracePeriodSeconds == nil {
		t.Fatalf("did not set deletion grace period seconds")
	}
	if *objectMeta.DeletionGracePeriodSeconds != expectedGrace+2 {
		t.Errorf("actual grace period does not match expected: %d", *objectMeta.DeletionGracePeriodSeconds)
	}
}

func (t *Tester) TestDeleteGracefulExtend(existing runtime.Object, expectedGrace int64, wasGracefulFn func() bool) {
	objectMeta, err := api.ObjectMetaFor(existing)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, existing)
	}

	ctx := api.WithNamespace(t.TestContext(), objectMeta.Namespace)
	_, err = t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, api.NewDeleteOptions(expectedGrace))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !wasGracefulFn() {
		t.Errorf("did not gracefully delete resource")
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
	objectMeta, err = api.ObjectMetaFor(object)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, object)
	}
	if objectMeta.DeletionTimestamp == nil {
		t.Errorf("did not set deletion timestamp")
	}
	if objectMeta.DeletionGracePeriodSeconds == nil {
		t.Fatalf("did not set deletion grace period seconds")
	}
	if *objectMeta.DeletionGracePeriodSeconds != expectedGrace {
		t.Errorf("actual grace period does not match expected: %d", *objectMeta.DeletionGracePeriodSeconds)
	}
}

func (t *Tester) TestDeleteGracefulImmediate(existing runtime.Object, expectedGrace int64, wasGracefulFn func() bool) {
	objectMeta, err := api.ObjectMetaFor(existing)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, existing)
	}

	ctx := api.WithNamespace(t.TestContext(), objectMeta.Namespace)
	_, err = t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, api.NewDeleteOptions(expectedGrace))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !wasGracefulFn() {
		t.Errorf("did not gracefully delete resource")
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
	objectMeta, err = api.ObjectMetaFor(out)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}
	if objectMeta.DeletionTimestamp == nil || objectMeta.DeletionGracePeriodSeconds == nil || *objectMeta.DeletionGracePeriodSeconds != 0 {
		t.Errorf("unexpected deleted meta: %#v", objectMeta)
	}
}

func (t *Tester) TestDeleteGracefulUsesZeroOnNil(existing runtime.Object, expectedGrace int64) {
	objectMeta := t.getObjectMetaOrFail(existing)
	ctx := api.WithNamespace(t.TestContext(), objectMeta.Namespace)
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
	objMeta := t.getObjectMetaOrFail(obj)
	objMeta.Name = "foo1"
	objMeta.Namespace = api.NamespaceValue(ctx)

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
	objMeta := t.getObjectMetaOrFail(obj)
	objMeta.Name = "foo2"
	objMeta.Namespace = api.NamespaceValue(ctx)
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
	foo1Meta := t.getObjectMetaOrFail(foo1)
	foo1Meta.Name = "foo1"
	foo1Meta.Namespace = api.NamespaceValue(ctx)
	foo2 := copyOrDie(obj)
	foo2Meta := t.getObjectMetaOrFail(foo2)
	foo2Meta.Name = "foo2"
	foo2Meta.Namespace = api.NamespaceValue(ctx)

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
	foo1Meta := t.getObjectMetaOrFail(foo1)
	foo1Meta.Name = "foo1"
	foo1Meta.Namespace = api.NamespaceValue(ctx)
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
