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
	"errors"
	"fmt"
	"strconv"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	storeerr "k8s.io/apiserver/pkg/storage/errors"

	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

// the corrupt object deleter has the same interface as rest.GracefulDeleter
var _ rest.GracefulDeleter = &corruptObjectDeleter{}

// NewCorruptObjectDeleter returns a deleter that can perform unsafe deletion
// of corrupt objects, it can be invoked to delete corrupt objects only after
// the normal deletion flow fails with either a or b:
// a) the data from the storage fails to transform properly (eg. decryption failure)
// b) failure to decode the object
//
// NOTE: it skips precondition checks, finalizer constraints, and any
// post deletion hooks.
//
// WARNING: This may break the cluster if the resource being deleted has dependencies
func NewCorruptObjectDeleter(store *Store) *corruptObjectDeleter {
	return &corruptObjectDeleter{
		GracefulDeleter:          store,
		KeyFunc:                  store.KeyFunc,
		NewFunc:                  store.NewFunc,
		DefaultQualifiedResource: store.DefaultQualifiedResource,
		Storage:                  store.Storage.Storage,
	}
}

// corruptObjectDeleter implements unsafe object deletion flow
type corruptObjectDeleter struct {
	rest.GracefulDeleter

	KeyFunc                  func(ctx context.Context, name string) (string, error)
	NewFunc                  func() runtime.Object
	DefaultQualifiedResource schema.GroupResource
	// NOTE: not holding the DryRunnableStorage wrapper,
	// directly using the storage interface
	Storage storage.Interface
}

// Delete performs an unsafe deletion of the given resource from the storage
//
// NOTE: This function should NEVER be used for any normal deletion
// flow, it is exclusively used when the delete option
// 'IgnoreStoreReadErrorWithClusterBreakingPotential' is enabled by the user.
func (s *corruptObjectDeleter) Delete(ctx context.Context, name string, deleteValidation rest.ValidateObjectFunc, opts *metav1.DeleteOptions) (runtime.Object, bool, error) {
	const optionName = "ignoreStoreReadErrorWithClusterBreakingPotential"
	if opts == nil ||
		opts.IgnoreStoreReadErrorWithClusterBreakingPotential == nil ||
		!*opts.IgnoreStoreReadErrorWithClusterBreakingPotential {
		// developer error, unsafe deleter should be invoked only when
		// IgnoreStoreReadErrorWithClusterBreakingPotential is true
		return nil, false, apierrors.NewInternalError(fmt.Errorf("expected %s to be enabled", optionName))
	}

	qualifiedResource := s.qualifiedResourceFromContext(ctx)
	key, err := s.KeyFunc(ctx, name)
	if err != nil {
		return nil, false, err
	}
	obj := s.NewFunc()

	// TODO: what if Get returns the object from cache, in this case the
	// apiserver must restart for unsafe deletion to work
	err = s.Storage.Get(ctx, key, storage.GetOptions{}, obj)
	if err == nil || !storage.IsCorruptObject(err) {
		// TODO: The Invalid error should have a field for Resource.
		// After that field is added, we should fill the Resource and
		// leave the Kind field empty. See the discussion in #18526.
		qualifiedKind := schema.GroupKind{Group: qualifiedResource.Group, Kind: qualifiedResource.Resource}
		fieldErrList := field.ErrorList{
			field.Invalid(field.NewPath(optionName), true, "it is exclusively used to delete corrupt object(s), try again by removing this option"),
		}
		return nil, false, apierrors.NewInvalid(qualifiedKind, name, fieldErrList)
	}

	var (
		preconditions *storage.Preconditions
		internalErr   storage.InternalError
	)
	// if we have the resource version of the object then we pin it to the
	// preconditions, otherwise we drop preconditions entirely.
	if errors.As(err, &internalErr) && internalErr.ResourceVersion != 0 {
		preconditions = &storage.Preconditions{
			ResourceVersion: ptr.To[string](strconv.FormatInt(internalErr.ResourceVersion, 10)),
		}
	}

	klog.V(1).InfoS("Going to perform unsafe object deletion", "object", klog.KRef(genericapirequest.NamespaceValue(ctx), name))
	out := s.NewFunc()
	storageOpts := storage.DeleteOptions{IgnoreStoreReadError: true}
	// keep the admission
	if err := s.Storage.Delete(ctx, key, out, preconditions, storage.ValidateObjectFunc(deleteValidation), nil, storageOpts); err != nil {
		if storage.IsNotFound(err) {
			// the DELETE succeeded, but we don't have the object sine it's
			// not retrievable from the storage, so we send a nil objct
			return nil, false, nil
		}
		return nil, false, storeerr.InterpretDeleteError(err, qualifiedResource, name)
	}
	// the DELETE succeeded, but we don't have the object sine it's
	// not retrievable from the storage, so we send a nil objct
	return nil, true, nil
}

func (s *corruptObjectDeleter) qualifiedResourceFromContext(ctx context.Context) schema.GroupResource {
	if info, ok := genericapirequest.RequestInfoFrom(ctx); ok {
		return schema.GroupResource{Group: info.APIGroup, Resource: info.Resource}
	}
	// some implementations access storage directly and thus the context has no RequestInfo
	return s.DefaultQualifiedResource
}
