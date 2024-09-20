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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	storeerr "k8s.io/apiserver/pkg/storage/errors"

	"k8s.io/klog/v2"
)

var _ rest.GracefulDeleter = &corruptObjectDeleter{}
var _ rest.CorruptObjectDeleter = &corruptObjectDeleter{}

func NewCorruptObjectDeleter(store *Store) *corruptObjectDeleter {
	return &corruptObjectDeleter{
		KeyFunc:                  store.KeyFunc,
		NewFunc:                  store.NewFunc,
		DefaultQualifiedResource: store.DefaultQualifiedResource,
		Storage:                  store.Storage,
	}
}

type corruptObjectDeleter struct {
	KeyFunc                  func(ctx context.Context, name string) (string, error)
	NewFunc                  func() runtime.Object
	DefaultQualifiedResource schema.GroupResource
	Storage                  DryRunnableStorage
}

// Delete will delete the given resource from the storage, it will bypass:
// a) any safety check, or validation
// b) finalization constraints
// c) skip after delete hooks
// It assumes that the given object is corrupt - its data from the storage can
// not be transformed successfully, or it can not be decoded into a valid object.
// it will disregard these errors and go ahead with the unsafe deletion flow.
//
// NOTE: This function should NEVER be used for any normal deletion flow,
// it is exclusively used when the delete option
// IgnoreStoreReadErrorWithClusterBreakingPotential is specified by the user.
//
// WARNING: This will break the cluster if the resource has dependencies.
// Use only when you have assessed the ripple effects of the deletion.
// WARNING: Vendors will most likely consider using this option to be
// breaking the support of their product.
func (s *corruptObjectDeleter) Delete(ctx context.Context, name string, deleteValidation rest.ValidateObjectFunc, options *metav1.DeleteOptions) (runtime.Object, bool, error) {
	key, err := s.KeyFunc(ctx, name)
	if err != nil {
		return nil, false, err
	}
	out := s.NewFunc()
	klog.V(1).InfoS("Going to delete object from registry, triggered by delete option IgnoreStoreReadErrorWithClusterBreakingPotential", "object", klog.KRef(genericapirequest.NamespaceValue(ctx), name))

	storageOptions := storage.DeleteOptions{IgnoreStoreReadError: *options.IgnoreStoreReadErrorWithClusterBreakingPotential}

	// a) Using the rest.ValidateAllObjectFunc because the request is a DELETE
	// request and has already passed the admission for the DELETE verb.
	// b) dry-run is false
	// c) storage option is set to allow deletion if the object is corrupt
	if err := s.Storage.Delete(ctx, key, out, nil, storage.ValidateObjectFunc(deleteValidation), false, nil, storageOptions); err != nil {
		// Deletion is racy, i.e., there could be multiple
		// requests to remove all finalizers from the object,
		// so we ignore the NotFound error.
		if storage.IsNotFound(err) {
			// the DELETE succeeded, but we don't have the object sine it's
			// not retrievable from the storage, so we send a nil objct
			return nil, false, nil
		}
		return nil, false, storeerr.InterpretDeleteError(err, s.qualifiedResourceFromContext(ctx), name)
	}
	// the DELETE succeeded, but we don't have the object sine it's
	// not retrievable from the storage, so we send a nil objct
	return nil, true, nil
}

// shouldAllowUnsafeCorruptObjectDeletion returns true if and only if:
// a) the feature AllowUnsafeCorruptObjectDeletion is enabled
// b) the user has set the delete option
// 'IgnoreStoreReadErrorWithClusterBreakingPotential' to true
// c) the given store read error represents a corrupt object
// otherwise, it returns false
func (s *corruptObjectDeleter) IsCandidateForUnsafeDeletion(err error, options *metav1.DeleteOptions) bool {
	if options == nil {
		return false
	}
	if ignore := options.IgnoreStoreReadErrorWithClusterBreakingPotential; ignore != nil && *ignore && storage.IsCorruptObject(err) {
		return true
	}
	return false
}

func (s *corruptObjectDeleter) qualifiedResourceFromContext(ctx context.Context) schema.GroupResource {
	if info, ok := genericapirequest.RequestInfoFrom(ctx); ok {
		return schema.GroupResource{Group: info.APIGroup, Resource: info.Resource}
	}
	// some implementations access storage directly and thus the context has no RequestInfo
	return s.DefaultQualifiedResource
}
