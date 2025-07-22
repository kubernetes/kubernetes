/*
Copyright 2024 The Kubernetes Authors.

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
	"fmt"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/apis/example"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"

	"k8s.io/utils/ptr"
)

type result struct {
	deleted bool
	err     error
}

type deleteWant struct {
	deleted  bool
	checkErr func(err error) bool
}

var (
	wantNoError     = func(err error) bool { return err == nil }
	wantErrContains = func(shouldContain string) func(error) bool {
		return func(err error) bool {
			return err != nil && strings.Contains(err.Error(), shouldContain)
		}
	}
)

func (w deleteWant) verify(t *testing.T, got result) {
	t.Helper()

	if !w.checkErr(got.err) {
		t.Errorf("Unexpected failure with the deletion operation, got: %v", got.err)
	}
	if w.deleted != got.deleted {
		t.Errorf("Expected deleted to be: %t, but got: %t", w.deleted, got.deleted)
	}

}

func TestUnsafeDeletePrecondition(t *testing.T) {
	option := func(enabled bool) *metav1.DeleteOptions {
		return &metav1.DeleteOptions{
			IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](enabled),
		}
	}

	const (
		unsafeDeleteNotAllowed = "ignoreStoreReadErrorWithClusterBreakingPotential: Invalid value: true: is exclusively used to delete corrupt object(s), try again by removing this option"
		internalErr            = "Internal error occurred: initialization error, expected normal deletion flow to be used"
	)

	tests := []struct {
		name    string
		err     error
		opts    *metav1.DeleteOptions
		invoked int
		want    deleteWant
	}{
		{
			name: "option nil, should throw internal error",
			opts: nil,
			want: deleteWant{checkErr: wantErrContains(internalErr)},
		},
		{
			name: "option empty, should throw internal error",
			opts: &metav1.DeleteOptions{},
			want: deleteWant{checkErr: wantErrContains(internalErr)},
		},
		{
			name: "option false, should throw internal error",
			opts: option(false),
			want: deleteWant{checkErr: wantErrContains(internalErr)},
		},
		{
			name: "option true, object readable, should throw invalid error",
			opts: option(true),
			want: deleteWant{
				checkErr: wantErrContains(unsafeDeleteNotAllowed),
			},
		},
		{
			name: "option true, object not readable with unexpected error, should throw invalid error",
			opts: option(true),
			err:  fmt.Errorf("unexpected error"),
			want: deleteWant{
				checkErr: wantErrContains(unsafeDeleteNotAllowed),
			},
		},
		{
			name: "option true, object not readable with storage internal error, should throw invalid error",
			opts: option(true),
			err:  storage.NewInternalError(fmt.Errorf("unexpected error")),
			want: deleteWant{
				checkErr: wantErrContains(unsafeDeleteNotAllowed),
			},
		},
		{
			name: "option true, object not readable with corrupt object error, unsafe-delete should trigger",
			opts: option(true),
			err:  storage.NewCorruptObjError("foo", fmt.Errorf("object not decodable")),
			want: deleteWant{
				deleted:  true,
				checkErr: wantNoError,
			},
			invoked: 1,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
			destroyFunc, registry := NewTestGenericStoreRegistry(t)
			defer destroyFunc()

			object := &example.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec:       example.PodSpec{NodeName: "machine"},
			}
			_, err := registry.Create(ctx, object, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Unexpected error from Create: %v", err)
			}

			// wrap the storage so it returns the expected error
			cs := &corruptStorage{
				Interface: registry.Storage.Storage,
				err:       test.err,
			}
			registry.Storage.Storage = cs
			deleter := NewCorruptObjectDeleter(registry)

			_, deleted, err := deleter.Delete(ctx, "foo", rest.ValidateAllObjectFunc, test.opts)

			got := result{deleted: deleted, err: err}
			test.want.verify(t, got)
			if want, got := test.invoked, cs.unsafeDeleteInvoked; want != got {
				t.Errorf("Expected unsafe-delete to be invoked %d time(s), but got: %d", want, got)
			}
		})
	}
}

func TestUnsafeDeleteWithCorruptObject(t *testing.T) {
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()

	object := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec:       example.PodSpec{NodeName: "machine"},
	}
	// a) prerequisite: try deleting the object, we expect a not found error
	_, _, err := registry.Delete(ctx, object.Name, rest.ValidateAllObjectFunc, nil)
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}

	// b) create the target object
	_, err = registry.Create(ctx, object, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// c) wrap the storage to return corrupt object error
	cs := &corruptStorage{
		Interface: registry.Storage.Storage,
		err:       storage.NewCorruptObjError("key", fmt.Errorf("untransformable")),
	}
	registry.Storage.Storage = cs

	got := result{}
	// d) try deleting the traget object
	_, got.deleted, got.err = registry.Delete(ctx, object.Name, rest.ValidateAllObjectFunc, nil)
	want := deleteWant{checkErr: errors.IsInternalError}
	want.verify(t, got)

	// e) set up an unsafe-deleter
	deleter := NewCorruptObjectDeleter(registry)

	// f) try to delete the object, but don't set the delete option just yet
	_, got.deleted, got.err = deleter.Delete(ctx, object.Name, rest.ValidateAllObjectFunc, nil)
	want.verify(t, got)

	// g) this time, set the delete option to ignore store read error
	_, got.deleted, got.err = deleter.Delete(ctx, object.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{
		IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](true),
	})
	want = deleteWant{
		deleted:  true,
		checkErr: wantNoError,
	}
	want.verify(t, got)
	if want, got := 1, cs.unsafeDeleteInvoked; want != got {
		t.Errorf("Expected unsafe-delete to be invoked %d time(s), but got: %d", want, got)
	}
}

func TestUnsafeDeleteWithUnexpectedError(t *testing.T) {
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	// TODO: inject a corrupt transformer
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()

	object := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec:       example.PodSpec{NodeName: "machine"},
	}
	// a) create the target object
	_, err := registry.Create(ctx, object, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// b) wrap the storage to return corrupt object error
	cs := &corruptStorage{
		Interface: registry.Storage.Storage,
		err:       storage.NewInternalError(fmt.Errorf("unexpected error")),
	}
	registry.Storage.Storage = cs

	// c) try deleting the object using normal deletion flow
	got := result{}
	_, got.deleted, got.err = registry.Delete(ctx, object.Name, rest.ValidateAllObjectFunc, nil)
	want := deleteWant{checkErr: errors.IsInternalError}
	want.verify(t, got)

	// d) set up a corrupt object deleter for the registry
	deleter := NewCorruptObjectDeleter(registry)

	// e) try deleting with unsafe-delete
	_, got.deleted, got.err = deleter.Delete(ctx, object.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{
		IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](true),
	})
	want = deleteWant{
		checkErr: wantErrContains("is exclusively used to delete corrupt object(s), try again by removing this option"),
	}
	want.verify(t, got)
	if want, got := 0, cs.unsafeDeleteInvoked; want != got {
		t.Errorf("Expected unsafe-delete to be invoked %d time(s), but got: %d", want, got)
	}
}

func TestUnsafeDeleteWithAdmissionShouldBeSkipped(t *testing.T) {
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()

	// a) create the target object
	object := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec:       example.PodSpec{NodeName: "machine"},
	}
	_, err := registry.Create(ctx, object, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// b) wrap the storage layer to return corrupt object error
	registry.Storage.Storage = &corruptStorage{
		Interface: registry.Storage.Storage,
		err:       storage.NewCorruptObjError("key", fmt.Errorf("untransformable")),
	}

	// c) set up a corrupt object deleter for the registry
	deleter := NewCorruptObjectDeleter(registry)

	// d) try unsafe delete, but pass a validation that always fails
	var admissionInvoked int
	_, deleted, err := deleter.Delete(ctx, object.Name, func(_ context.Context, _ runtime.Object) error {
		admissionInvoked++
		return fmt.Errorf("admission was not skipped")
	}, &metav1.DeleteOptions{
		IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](true),
	})

	if err != nil {
		t.Errorf("Unexpected error from Delete: %v", err)
	}
	if want, got := true, deleted; want != got {
		t.Errorf("Expected deleted: %t, but got: %t", want, got)
	}
	if want, got := 0, admissionInvoked; want != got {
		t.Errorf("Expected admission to be invoked %d time(s), but got: %d", want, got)
	}
}

type corruptStorage struct {
	storage.Interface
	err                 error
	unsafeDeleteInvoked int
}

func (s *corruptStorage) Get(ctx context.Context, key string, opts storage.GetOptions, objPtr runtime.Object) error {
	if s.err != nil {
		return s.err
	}
	return s.Interface.Get(ctx, key, opts, objPtr)
}

func (s *corruptStorage) Delete(ctx context.Context, key string, out runtime.Object, preconditions *storage.Preconditions, deleteValidation storage.ValidateObjectFunc, cachedExistingObject runtime.Object, opts storage.DeleteOptions) error {
	if opts.IgnoreStoreReadError {
		s.unsafeDeleteInvoked++
	}
	return s.Interface.Delete(ctx, key, out, preconditions, deleteValidation, cachedExistingObject, opts)
}
