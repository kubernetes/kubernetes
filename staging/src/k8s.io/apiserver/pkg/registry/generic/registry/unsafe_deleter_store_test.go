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

func TestShouldAllowUnsafeCorruptObjectDeletion(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		opts     *metav1.DeleteOptions
		expected bool
	}{
		{
			name: "no error",
			opts: &metav1.DeleteOptions{
				IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](true),
			},
		},
		{
			name: "corrupt object error, options is nil",
			err:  storage.NewCorruptObjError("foo", "", nil),
			opts: nil,
		},
		{
			name: "corrupt object error, IgnoreStoreReadErrorWithClusterBreakingPotential is nil",
			err:  storage.NewCorruptObjError("foo", "", nil),
			opts: nil,
		},
		{
			name: "corrupt object error, IgnoreStoreReadErrorWithClusterBreakingPotential is false",
			err:  storage.NewCorruptObjError("foo", "", nil),
			opts: &metav1.DeleteOptions{
				IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](false),
			},
		},
		{
			name: "error does not represent corrupt object, IgnoreStoreReadErrorWithClusterBreakingPotential is true",
			err:  fmt.Errorf("unexpected error"),
			opts: &metav1.DeleteOptions{
				IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](true),
			},
		},
		{
			name: "internal storage error, IgnoreStoreReadErrorWithClusterBreakingPotential is true",
			err:  storage.NewInternalError(fmt.Errorf("unexpected error")),
			opts: &metav1.DeleteOptions{
				IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](true),
			},
		},
		{
			name: "corrupt object error, IgnoreStoreReadErrorWithClusterBreakingPotential is true",
			err:  storage.NewCorruptObjError("foo", "", nil),
			opts: &metav1.DeleteOptions{
				IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](true),
			},
			expected: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			unsafe := &corruptObjectDeleter{}
			if want, got := test.expected, unsafe.IsCandidateForUnsafeDeletion(test.err, test.opts); want != got {
				t.Errorf("expected %t, but got: %t", want, got)
			}
		})
	}
}

func TestCorruptObjectDelete(t *testing.T) {
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

	// c) wire the storage to return corrupt object error
	cs := &corruptStorage{
		Interface: registry.Storage.Storage,
		err:       storage.NewCorruptObjError("key", "", fmt.Errorf("unexpected error")),
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
	registry.CorruptObjDeleter = NewCorruptObjectDeleter(registry)

	// f) try to delete the target object now, but note that the user
	// has not set the ignore store read error option yet
	_, wasDeleted, err = registry.Delete(testContext, podA.Name, rest.ValidateAllObjectFunc, nil)
	if !errors.IsInternalError(err) {
		t.Errorf("Unexpected error: %v", err)
	}
	if wasDeleted {
		t.Errorf("Unexpected, the user did not set the delete option")
	}

	// g) this time, set the delete option to ignore store read error
	_, wasDeleted, err = registry.Delete(testContext, podA.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{
		IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](true),
	})
	if err != nil {
		t.Errorf("Expected the corrupt object deletion flow to have worked, but got: %v", err)
	}
	if !wasDeleted {
		t.Errorf("Expected the corrupt object deletion flow to have worked")
	}
}

type corruptStorage struct {
	storage.Interface
	err error
}

func (s *corruptStorage) Get(ctx context.Context, key string, opts storage.GetOptions, objPtr runtime.Object) error {
	if s.err != nil {
		return s.err
	}
	return s.Interface.Get(ctx, key, opts, objPtr)
}
