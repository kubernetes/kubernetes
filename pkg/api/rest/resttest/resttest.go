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
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
)

type Tester struct {
	*testing.T
	storage      rest.Storage
	storageError injectErrorFunc
	clusterScope bool
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

func copyOrDie(obj runtime.Object) runtime.Object {
	out, err := api.Scheme.Copy(obj)
	if err != nil {
		panic(err)
	}
	return out
}

func (t *Tester) TestCreate(valid runtime.Object, invalid ...runtime.Object) {
	t.TestCreateHasMetadata(copyOrDie(valid))
	t.TestCreateGeneratesName(copyOrDie(valid))
	t.TestCreateGeneratesNameReturnsServerTimeout(copyOrDie(valid))
	if t.clusterScope {
		t.TestCreateDiscardsObjectNamespace(copyOrDie(valid))
		t.TestCreateIgnoresContextNamespace(copyOrDie(valid))
		t.TestCreateIgnoresMismatchedNamespace(copyOrDie(valid))
	} else {
		t.TestCreateRejectsMismatchedNamespace(copyOrDie(valid))
	}
	t.TestCreateInvokesValidation(invalid...)
}

func (t *Tester) TestCreateResetsUserData(valid runtime.Object) {
	objectMeta, err := api.ObjectMetaFor(valid)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, valid)
	}

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

func (t *Tester) TestCreateHasMetadata(valid runtime.Object) {
	objectMeta, err := api.ObjectMetaFor(valid)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, valid)
	}

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

func (t *Tester) TestCreateGeneratesName(valid runtime.Object) {
	objectMeta, err := api.ObjectMetaFor(valid)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, valid)
	}

	objectMeta.Name = ""
	objectMeta.GenerateName = "test-"

	_, err = t.storage.(rest.Creater).Create(t.TestContext(), valid)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if objectMeta.Name == "test-" || !strings.HasPrefix(objectMeta.Name, "test-") {
		t.Errorf("unexpected name: %#v", valid)
	}
}

func (t *Tester) TestCreateGeneratesNameReturnsServerTimeout(valid runtime.Object) {
	objectMeta, err := api.ObjectMetaFor(valid)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, valid)
	}

	objectMeta.Name = ""
	objectMeta.GenerateName = "test-"
	t.withStorageError(errors.NewAlreadyExists("kind", "thing"), func() {
		_, err := t.storage.(rest.Creater).Create(t.TestContext(), valid)
		if err == nil || !errors.IsServerTimeout(err) {
			t.Fatalf("Unexpected error: %v", err)
		}
	})
}

func (t *Tester) TestCreateInvokesValidation(invalid ...runtime.Object) {
	for i, obj := range invalid {
		ctx := t.TestContext()
		_, err := t.storage.(rest.Creater).Create(ctx, obj)
		if !errors.IsInvalid(err) {
			t.Errorf("%d: Expected to get an invalid resource error, got %v", i, err)
		}
	}
}

func (t *Tester) TestCreateRejectsMismatchedNamespace(valid runtime.Object) {
	objectMeta, err := api.ObjectMetaFor(valid)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, valid)
	}

	objectMeta.Namespace = "not-default"

	_, err = t.storage.(rest.Creater).Create(t.TestContext(), valid)
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if !strings.Contains(err.Error(), "does not match the namespace sent on the request") {
		t.Errorf("Expected 'does not match the namespace sent on the request' error, got '%v'", err.Error())
	}
}

func (t *Tester) TestCreateDiscardsObjectNamespace(valid runtime.Object) {
	objectMeta, err := api.ObjectMetaFor(valid)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, valid)
	}

	// Ignore non-empty namespace in object meta
	objectMeta.Namespace = "not-default"

	// Ideally, we'd get an error back here, but at least verify the namespace wasn't persisted
	created, err := t.storage.(rest.Creater).Create(t.TestContext(), copyOrDie(valid))
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	createdObjectMeta, err := api.ObjectMetaFor(created)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, created)
	}
	if createdObjectMeta.Namespace != api.NamespaceNone {
		t.Errorf("Expected empty namespace on created object, got '%v'", createdObjectMeta.Namespace)
	}
}

func (t *Tester) TestCreateIgnoresContextNamespace(valid runtime.Object) {
	// Ignore non-empty namespace in context
	ctx := api.WithNamespace(api.NewContext(), "not-default2")

	// Ideally, we'd get an error back here, but at least verify the namespace wasn't persisted
	created, err := t.storage.(rest.Creater).Create(ctx, copyOrDie(valid))
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	createdObjectMeta, err := api.ObjectMetaFor(created)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, created)
	}
	if createdObjectMeta.Namespace != api.NamespaceNone {
		t.Errorf("Expected empty namespace on created object, got '%v'", createdObjectMeta.Namespace)
	}
}

func (t *Tester) TestCreateIgnoresMismatchedNamespace(valid runtime.Object) {
	objectMeta, err := api.ObjectMetaFor(valid)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, valid)
	}

	// Ignore non-empty namespace in object meta
	objectMeta.Namespace = "not-default"
	ctx := api.WithNamespace(api.NewContext(), "not-default2")

	// Ideally, we'd get an error back here, but at least verify the namespace wasn't persisted
	created, err := t.storage.(rest.Creater).Create(ctx, copyOrDie(valid))
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	createdObjectMeta, err := api.ObjectMetaFor(created)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, created)
	}
	if createdObjectMeta.Namespace != api.NamespaceNone {
		t.Errorf("Expected empty namespace on created object, got '%v'", createdObjectMeta.Namespace)
	}
}

func (t *Tester) TestUpdate(valid runtime.Object, existing, older runtime.Object) {
	t.TestUpdateFailsOnNotFound(copyOrDie(valid))
	t.TestUpdateFailsOnVersion(copyOrDie(older))
}

func (t *Tester) TestUpdateFailsOnNotFound(valid runtime.Object) {
	_, _, err := t.storage.(rest.Updater).Update(t.TestContext(), valid)
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if !errors.IsNotFound(err) {
		t.Errorf("Expected NotFound error, got '%v'", err)
	}
}

func (t *Tester) TestUpdateFailsOnVersion(older runtime.Object) {
	_, _, err := t.storage.(rest.Updater).Update(t.TestContext(), older)
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if !errors.IsConflict(err) {
		t.Errorf("Expected Conflict error, got '%v'", err)
	}
}

func (t *Tester) TestDeleteInvokesValidation(invalid ...runtime.Object) {
	for i, obj := range invalid {
		objectMeta, err := api.ObjectMetaFor(obj)
		if err != nil {
			t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, obj)
		}
		ctx := t.TestContext()
		_, err = t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, nil)
		if !errors.IsInvalid(err) {
			t.Errorf("%d: Expected to get an invalid resource error, got %v", i, err)
		}
	}
}

func (t *Tester) TestDelete(createFn func() runtime.Object, wasGracefulFn func() bool, invalid ...runtime.Object) {
	t.TestDeleteNonExist(createFn)
	t.TestDeleteNoGraceful(createFn, wasGracefulFn)
	t.TestDeleteInvokesValidation(invalid...)
	// TODO: Test delete namespace mismatch rejection
	// once #5684 is fixed.
}

func (t *Tester) TestDeleteNonExist(createFn func() runtime.Object) {
	existing := createFn()
	objectMeta, err := api.ObjectMetaFor(existing)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, existing)
	}
	context := t.TestContext()

	t.withStorageError(&etcd.EtcdError{ErrorCode: tools.EtcdErrorCodeNotFound}, func() {
		_, err := t.storage.(rest.GracefulDeleter).Delete(context, objectMeta.Name, nil)
		if err == nil || !errors.IsNotFound(err) {
			t.Fatalf("Unexpected error: %v", err)
		}
	})
}

func (t *Tester) TestDeleteGraceful(createFn func() runtime.Object, expectedGrace int64, wasGracefulFn func() bool) {
	t.TestDeleteGracefulHasDefault(createFn(), expectedGrace, wasGracefulFn)
	t.TestDeleteGracefulUsesZeroOnNil(createFn(), 0)
}

func (t *Tester) TestDeleteNoGraceful(createFn func() runtime.Object, wasGracefulFn func() bool) {
	existing := createFn()
	objectMeta, err := api.ObjectMetaFor(existing)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, existing)
	}
	ctx := api.WithNamespace(t.TestContext(), objectMeta.Namespace)
	_, err = t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, api.NewDeleteOptions(10))
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
	objectMeta, err := api.ObjectMetaFor(existing)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, existing)
	}

	ctx := api.WithNamespace(t.TestContext(), objectMeta.Namespace)
	_, err = t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, &api.DeleteOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if _, err := t.storage.(rest.Getter).Get(ctx, objectMeta.Name); err != nil {
		t.Errorf("unexpected error, object should exist: %v", err)
	}
	if !wasGracefulFn() {
		t.Errorf("did not gracefully delete resource")
	}
}

func (t *Tester) TestDeleteGracefulUsesZeroOnNil(existing runtime.Object, expectedGrace int64) {
	objectMeta, err := api.ObjectMetaFor(existing)
	if err != nil {
		t.Fatalf("object does not have ObjectMeta: %v\n%#v", err, existing)
	}

	ctx := api.WithNamespace(t.TestContext(), objectMeta.Namespace)
	_, err = t.storage.(rest.GracefulDeleter).Delete(ctx, objectMeta.Name, nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if _, err := t.storage.(rest.Getter).Get(ctx, objectMeta.Name); !errors.IsNotFound(err) {
		t.Errorf("unexpected error, object should exist: %v", err)
	}
}
