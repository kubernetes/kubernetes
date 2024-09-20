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

func TestUnsafeDeletePrecondition(t *testing.T) {
	tests := []struct {
		name    string
		err     error
		opts    *metav1.DeleteOptions
		invoked int
	}{
		{
			name: "no error, want: not invoked",
			opts: &metav1.DeleteOptions{
				IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](true),
			},
		},
		{
			name: "corrupt object error, options is nil, want: not invoked",
			err:  storage.NewCorruptObjError("foo", fmt.Errorf("object not decodable")),
			opts: nil,
		},
		{
			name: "corrupt object error, IgnoreStoreReadErrorWithClusterBreakingPotential is nil, want: not invoked",
			err:  storage.NewCorruptObjError("foo", fmt.Errorf("object not decodable")),
			opts: nil,
		},
		{
			name: "corrupt object error, IgnoreStoreReadErrorWithClusterBreakingPotential is false, want: not invoked",
			err:  storage.NewCorruptObjError("foo", fmt.Errorf("object not decodable")),
			opts: &metav1.DeleteOptions{
				IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](false),
			},
		},
		{
			name: "error is not corrupt object, IgnoreStoreReadErrorWithClusterBreakingPotential is true, want: not invoked",
			err:  fmt.Errorf("unexpected error"),
			opts: &metav1.DeleteOptions{
				IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](true),
			},
		},
		{
			name: "internal storage error, IgnoreStoreReadErrorWithClusterBreakingPotential is true, want: not invoked",
			err:  storage.NewInternalError(fmt.Errorf("unexpected error")),
			opts: &metav1.DeleteOptions{
				IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](true),
			},
		},
		{
			name: "corrupt object error, IgnoreStoreReadErrorWithClusterBreakingPotential is true, want: invoked",
			err:  storage.NewCorruptObjError("key", fmt.Errorf("object not decodable")),
			opts: &metav1.DeleteOptions{
				IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](true),
			},
			invoked: 1,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
			destroyFunc, registry := NewTestGenericStoreRegistry(t)
			defer destroyFunc()

			// wrap the storage so it returns the error we want to
			cs := &corruptStorage{
				Interface: registry.Storage.Storage,
				err:       test.err,
			}
			registry.Storage.Storage = cs
			deleter := NewCorruptObjectDeleter(registry)

			_, _, err := deleter.Delete(testContext, "foo", rest.ValidateAllObjectFunc, test.opts)
			if err != nil {
				t.Logf("Registry Delete returned error: %v", err)
			}

			if want, got := test.invoked, cs.deleteInvoked; want != got {
				t.Errorf("Expected unsafe delete to be invoked %d time(s), but got: %d", want, got)
			}
		})
	}
}

func TestUnsafeDeleteWithCorruptObject(t *testing.T) {
	podA := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec:       example.PodSpec{NodeName: "machine"},
	}

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()

	// a) prerequisite: try deleting the object, we expect a not found error
	_, _, err := registry.Delete(testContext, podA.Name, rest.ValidateAllObjectFunc, nil)
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}

	// b) create the target object
	_, err = registry.Create(testContext, podA, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// c) wrap the storage to return corrupt object error
	cs := &corruptStorage{
		Interface: registry.Storage.Storage,
		err:       storage.NewCorruptObjError("key", fmt.Errorf("unexpected error")),
	}
	registry.Storage.Storage = cs

	// d) try deleting the traget object
	_, wasDeleted, err := registry.Delete(testContext, podA.Name, rest.ValidateAllObjectFunc, nil)
	if !errors.IsInternalError(err) {
		t.Errorf("Unexpected failure with the normal deletion flow, got: %v", err)
	}
	if wasDeleted {
		t.Errorf("Unexpected, normal deletion flow did not fail")
	}

	// e) set up a corrupt object deleter for the registry
	deleter := NewCorruptObjectDeleter(registry)

	// f) try to delete the target object now, but note that the user
	// has not set the ignore store read error option yet
	_, wasDeleted, err = deleter.Delete(testContext, podA.Name, rest.ValidateAllObjectFunc, nil)
	if !errors.IsInternalError(err) {
		t.Errorf("Unexpected error: %v", err)
	}
	if wasDeleted {
		t.Errorf("Unexpected, the user did not set the delete option")
	}

	// g) this time, set the delete option to ignore store read error
	_, wasDeleted, err = deleter.Delete(testContext, podA.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{
		IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](true),
	})
	if err != nil {
		t.Errorf("Expected the corrupt object deletion flow to have worked, but got: %v", err)
	}
	if !wasDeleted {
		t.Errorf("Expected the corrupt object deletion flow to have worked")
	}
	if want, got := 1, cs.deleteInvoked; want != got {
		t.Errorf("Expected unsafe delete to be invoked %d time(s), but got: %d", want, got)
	}
}

func TestUnsafeDeleteWithReadableObject(t *testing.T) {
	podA := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec:       example.PodSpec{NodeName: "machine"},
	}

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()

	// a) prerequisite: try deleting the object, we expect a not found error
	_, _, err := registry.Delete(testContext, podA.Name, rest.ValidateAllObjectFunc, nil)
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}

	// b) create the target object
	_, err = registry.Create(testContext, podA, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// c) wrap the storage to return corrupt object error
	cs := &corruptStorage{
		Interface: registry.Storage.Storage,
		err:       storage.NewInternalError(fmt.Errorf("unexpected error")),
	}
	registry.Storage.Storage = cs

	// d) try deleting the traget object using the registry
	_, wasDeleted, err := registry.Delete(testContext, podA.Name, rest.ValidateAllObjectFunc, nil)
	if !errors.IsInternalError(err) {
		t.Errorf("Unexpected failure with the normal deletion flow, got: %v", err)
	}
	if wasDeleted {
		t.Errorf("Unexpected, normal deletion flow did not fail")
	}

	// e) set up a corrupt object deleter for the registry
	deleter := NewCorruptObjectDeleter(registry)

	// g) this time, set the delete option to ignore store read error
	_, _, err = deleter.Delete(testContext, podA.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{
		IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](true),
	})
	if want := "is exclusively used to delete corrupt object(s), try again by removing this option"; err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("Expected the error to contain: %q, but got: %v", want, err)
	}
	if want, got := 0, cs.deleteInvoked; want != got {
		t.Errorf("Expected unsafe delete to be invoked %d time(s), but got: %d", want, got)
	}
}

type corruptStorage struct {
	storage.Interface
	err           error
	deleteInvoked int
}

func (s *corruptStorage) Get(ctx context.Context, key string, opts storage.GetOptions, objPtr runtime.Object) error {
	if s.err != nil {
		return s.err
	}
	return s.Interface.Get(ctx, key, opts, objPtr)
}

func (s *corruptStorage) Delete(ctx context.Context, key string, out runtime.Object, preconditions *storage.Preconditions, deleteValidation storage.ValidateObjectFunc, cachedExistingObject runtime.Object, opts storage.DeleteOptions) error {
	s.deleteInvoked++
	return s.Interface.Delete(ctx, key, out, preconditions, deleteValidation, cachedExistingObject, opts)
}
