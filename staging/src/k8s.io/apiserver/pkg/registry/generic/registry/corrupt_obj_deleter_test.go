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
	"path"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/apis/example"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/utils/ptr"
)

// fakeStorage is a test double for storage.Interface that intercepts Delete calls, records the
// arguments, and returns a pre-configured error. By default it mimics the etcd3 store handling
// a corrupt object: the deletion validator runs after the corruption check and before the
// delete, a validator error aborts the delete. With bypassValidator set, it simulates a buggy
// backend that deletes without consulting the deletion validator.
type fakeStorage struct {
	storage.Interface
	deleteErr       error
	bypassValidator bool

	deleteCalled            bool
	key                     string
	expectTransformOrDecode bool
	preconditions           *storage.Preconditions
	cachedExistingObject    runtime.Object
}

func (s *fakeStorage) Delete(ctx context.Context, key string, out runtime.Object, preconditions *storage.Preconditions, deleteValidation storage.ValidateObjectFunc, cachedExistingObject runtime.Object, opts storage.DeleteOptions) error {
	s.deleteCalled = true
	s.key = key
	s.expectTransformOrDecode = opts.ExpectTransformOrDecodeError
	s.preconditions = preconditions
	s.cachedExistingObject = cachedExistingObject
	if s.deleteErr != nil {
		return s.deleteErr
	}
	if s.bypassValidator {
		return nil
	}
	// the object is nil for a corrupt object, it cannot be decoded
	return deleteValidation(ctx, nil)
}

func TestCorruptObjectDeleterDelete(t *testing.T) {
	for _, test := range []struct {
		name               string
		deleteErr          error
		bypassValidator    bool
		opts               *metav1.DeleteOptions
		expectDeleteCalled bool
		wantDeleted        bool
		wantErr            func(error) bool
	}{
		{
			name:    "options nil, should return internal error",
			opts:    nil,
			wantErr: errors.IsInternalError,
		},
		{
			name:    "options empty, should return internal error",
			opts:    &metav1.DeleteOptions{},
			wantErr: errors.IsInternalError,
		},
		{
			name:    "option false, should return internal error",
			opts:    &metav1.DeleteOptions{IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To(false)},
			wantErr: errors.IsInternalError,
		},
		{
			name:               "option true, object decodable, store returns InvalidObj",
			opts:               &metav1.DeleteOptions{IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To(true)},
			deleteErr:          storage.NewInvalidObjError("/pods/foo", "object is decodable"),
			expectDeleteCalled: true,
			wantErr:            errors.IsConflict,
		},
		{
			name:               "option true, object not decodable, delete succeeds",
			opts:               &metav1.DeleteOptions{IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To(true)},
			deleteErr:          nil,
			expectDeleteCalled: true,
			wantDeleted:        true,
			wantErr:            func(err error) bool { return err == nil },
		},
		{
			name:               "option true, object not found, store returns NotFound",
			opts:               &metav1.DeleteOptions{IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To(true)},
			deleteErr:          storage.NewKeyNotFoundError("/pods/foo", 0),
			expectDeleteCalled: true,
			wantErr:            errors.IsNotFound,
		},
		{
			name:               "option true, dry run, object not decodable, store honors the validator",
			opts:               &metav1.DeleteOptions{IgnoreStoreReadErrorWithClusterBreakingPotential: new(true), DryRun: []string{"All"}},
			expectDeleteCalled: true,
			wantDeleted:        true,
			wantErr:            func(err error) bool { return err == nil },
		},
		{
			name:               "option true, dry run, object decodable, store returns InvalidObj",
			opts:               &metav1.DeleteOptions{IgnoreStoreReadErrorWithClusterBreakingPotential: new(true), DryRun: []string{"All"}},
			deleteErr:          storage.NewInvalidObjError("/pods/foo", "object is decodable"),
			expectDeleteCalled: true,
			wantErr:            errors.IsConflict,
		},
		{
			name:               "option true, dry run, object not found, store returns NotFound",
			opts:               &metav1.DeleteOptions{IgnoreStoreReadErrorWithClusterBreakingPotential: new(true), DryRun: []string{"All"}},
			deleteErr:          storage.NewKeyNotFoundError("/pods/foo", 0),
			expectDeleteCalled: true,
			wantErr:            errors.IsNotFound,
		},
		{
			name:               "option true, dry run, store deletes without invoking the validator",
			opts:               &metav1.DeleteOptions{IgnoreStoreReadErrorWithClusterBreakingPotential: new(true), DryRun: []string{"All"}},
			bypassValidator:    true,
			expectDeleteCalled: true,
			wantErr:            errors.IsInternalError,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
			fs := &fakeStorage{deleteErr: test.deleteErr, bypassValidator: test.bypassValidator}
			const podPrefix = "/pods/"
			store := &Store{
				NewFunc:                   func() runtime.Object { return &example.Pod{} },
				DefaultQualifiedResource:  schema.GroupResource{Resource: "pods"},
				SingularQualifiedResource: schema.GroupResource{Resource: "pod"},
				KeyRootFunc:               func(ctx context.Context) string { return podPrefix },
				KeyFunc: func(ctx context.Context, name string) (string, error) {
					if _, ok := genericapirequest.NamespaceFrom(ctx); !ok {
						return "", fmt.Errorf("namespace is required")
					}
					return path.Join(podPrefix, name), nil
				},
				Storage: DryRunnableStorage{Storage: fs},
			}
			deleter := NewCorruptObjectDeleter(store)

			// the unsafe deleter must ignore the caller's admission validator, it
			// passes its own validator to the storage instead: admit-everything for
			// a regular delete, a hardcoded-fail sentinel for dry run. This func
			// fails the test if the deleter ever wires admission back in.
			obj, deleted, err := deleter.Delete(ctx, "foo", func(context.Context, runtime.Object) error {
				t.Fatal("caller-provided admission was invoked")
				return nil
			}, test.opts)

			if obj != nil {
				t.Errorf("Expected nil object, but got %v", obj)
			}
			if !test.wantErr(err) {
				t.Errorf("Unexpected error: %v", err)
			}
			if test.wantDeleted != deleted {
				t.Errorf("Expected deleted to be %t, but got %t", test.wantDeleted, deleted)
			}
			if test.expectDeleteCalled != fs.deleteCalled {
				t.Errorf("Expected storage Delete called=%t, but got %t", test.expectDeleteCalled, fs.deleteCalled)
			}
			if fs.deleteCalled {
				if want, got := "/pods/foo", fs.key; want != got {
					t.Errorf("Expected storage Delete to be called with key %q, but got %q", want, got)
				}
				if !fs.expectTransformOrDecode {
					t.Error("Expected storage Delete to be called with ExpectTransformOrDecodeError=true")
				}
				if fs.preconditions != nil {
					t.Error("Expected storage Delete to be called with nil preconditions")
				}
				if fs.cachedExistingObject != nil {
					t.Error("Expected storage Delete to be called with nil cachedExistingObject")
				}
			}
		})
	}
}
