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
// arguments, and returns a pre-configured error.
type fakeStorage struct {
	storage.Interface
	deleteErr error

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
	return s.deleteErr
}

func TestCorruptObjectDeleterDelete(t *testing.T) {
	for _, test := range []struct {
		name               string
		deleteErr          error
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
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
			fs := &fakeStorage{deleteErr: test.deleteErr}
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
